from flask import Flask, render_template, request, jsonify
import os, re
import joblib
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Safe import for Google News RSS parser
try:
    import feedparser
except Exception as e:
    print(f"⚠️ feedparser not available: {e}")
    feedparser = None


app = Flask(__name__)

# ----------------------------
# 1) Load model & vectorizer
# ----------------------------
MODEL_PATH = "news_model.pkl"
VECT_PATH  = "vectorizer.pkl"

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"⚠️ Could not load {path}: {e}")
        return None

model = safe_load(MODEL_PATH)
vectorizer = safe_load(VECT_PATH)

# Determine positive/real class index safely
def positive_index(clf):
    try:
        classes = list(clf.classes_)
        if 1 in classes:
            return classes.index(1)
        low = [str(c).lower() for c in classes]
        if "real" in low:
            return low.index("real")
        return None
    except Exception:
        return None

POS_IDX = positive_index(model) if model is not None else None

# ----------------------------
# 2) Evidence (Live + Fallback)
# ----------------------------
# (Removed unused NEWS_API_KEY ✅)

DF, VEC, TFIDF_MATRIX = None, None, None
try:
    DF = pd.read_csv("data.csv")
    DF["title"] = DF["title"].fillna("")
    DF["text"]  = DF["text"].fillna("")
    DF["_full_text"] = (DF["title"] + " " + DF["text"]).str.slice(0, 5000)
    VEC = TfidfVectorizer(max_features=4000, stop_words="english")
    TFIDF_MATRIX = VEC.fit_transform(DF["_full_text"])
except Exception as e:
    print(f"⚠️ Fallback dataset unavailable: {e}")

def find_local_evidence(news_text, top_k=4):
    if VEC is None or TFIDF_MATRIX is None or DF is None or DF.empty:
        return []
    input_vec = VEC.transform([news_text])
    sims = cosine_similarity(input_vec, TFIDF_MATRIX).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    rows = []
    for i in top_idx:
        row = DF.iloc[i]
        rows.append({
            "headline": row.get("title", "Untitled"),
            "source": row.get("source", "Archive"),
            "url": "#",
            "publishedAt": str(row.get("date", "")),
            "image": None,
            "badge": "Archive",
            "similarity": float(sims[i])
        })
    return rows

# ----------------------------
# Google News RSS Evidence
# ----------------------------
STOPWORDS = {...}  # keep your existing stopwords set

def _extract_keywords(text, cap=10):
    import re
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z]+", text)
    lower = [t.lower() for t in tokens]
    keep, seen = [], set()
    for i, tok in enumerate(tokens):
        lo = lower[i]
        if len(lo) <= 2 or lo in STOPWORDS:
            continue
        if lo not in seen:
            keep.append(lo)
            seen.add(lo)
        if len(keep) >= cap:
            break
    return keep or [w for w in re.findall(r"[A-Za-z]+", text.lower()) if w not in STOPWORDS][:cap]

from urllib.parse import urlparse

def fetch_live_evidence(query, max_results=6):
    if feedparser is None:
        return find_local_evidence(query)
    keywords = _extract_keywords(query, cap=10)
    keyword_query = " ".join(keywords) if keywords else query
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(keyword_query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, timeout=6, headers=headers)
        if r.status_code != 200 or not r.text:
            return find_local_evidence(query)
        feed = feedparser.parse(r.text)
        out, seen_links = [], set()
        for entry in getattr(feed, "entries", [])[:max_results*2]:
            link = entry.get("link") or "#"
            if link in seen_links:
                continue
            seen_links.add(link)

            image_url = None
            if hasattr(entry, "media_content") and entry.media_content:
                image_url = entry.media_content[0].get("url", None)
            elif hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
                image_url = entry.media_thumbnail[0].get("url", None)

            if not image_url and link and link != "#":
                try:
                    domain = urlparse(link).netloc
                    if domain:
                        image_url = f"https://www.google.com/s2/favicons?sz=128&domain={domain}"
                except Exception:
                    pass

            card = {
                "headline": entry.get("title", "Untitled"),
                "source": getattr(entry.source, "title", "Google News") if hasattr(entry, "source") else "Google News",
                "url": link,
                "publishedAt": entry.get("published", ""),
                "image": image_url,
                "badge": "Live"
            }
            out.append(card)
            if len(out) >= max_results:
                break
        return out
    except Exception as e:
        print(f"⚠️ Google News fetch error: {e}")
        return find_local_evidence(query)

# ----------------------------
# 3) Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = ""
    if request.is_json:
        body = request.get_json(silent=True) or {}
        news_text = (body.get("text") or "").strip()
    else:
        news_text = (request.form.get("newsText") or "").strip()

    if not news_text:
        return jsonify({"error": "No news text provided"}), 400

    verdict, real_prob, fake_prob = "Unknown", 0.5, 0.5
    if model is not None and vectorizer is not None:
        try:
            x = vectorizer.transform([news_text])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                if POS_IDX is None:
                    real_prob = float(max(probs))
                    fake_prob = float(1.0 - real_prob)
                else:
                    real_prob = float(probs[POS_IDX])
                    fake_prob = float(1.0 - real_prob)
                verdict = "Real" if real_prob >= fake_prob else "Fake"
            else:
                y = model.predict(x)[0]
                verdict = "Real" if str(y).lower() in ["real", "1"] else "Fake"
                real_prob = 0.75 if verdict == "Real" else 0.25
                fake_prob = 1.0 - real_prob
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            verdict = "Error"
    else:
        verdict = "Model not loaded"

    evidence_cards = fetch_live_evidence(news_text)

    return jsonify({
        "verdict": verdict,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "probability": real_prob if verdict == "Real" else fake_prob,
        "evidence": evidence_cards,
    })

# ----------------------------
# 4) Entry Point
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
