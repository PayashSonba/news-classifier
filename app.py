from flask import Flask, render_template, request, jsonify
import os, re
import joblib
import json
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

model = None
vectorizer = None

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
        # Prefer label "1" or "real" if present
        if 1 in classes:
            return classes.index(1)
        # case-insensitive 'real'
        low = [str(c).lower() for c in classes]
        if "real" in low:
            return low.index("real")
        # otherwise pick the max prob as "real" (fallback)
        return None
    except Exception:
        return None

POS_IDX = positive_index(model) if model is not None else None

# ----------------------------
# 2) Evidence (Live + Fallback)
# ----------------------------
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")  # set env var or replace string

# Load local dataset for fallback (lightweight TF-IDF to keep old laptop happy)
DF = None
VEC = None
TFIDF_MATRIX = None
try:
    DF = pd.read_csv("data.csv")
    DF["title"] = DF["title"].fillna("")
    DF["text"]  = DF["text"].fillna("")
    DF["_full_text"] = (DF["title"] + " " + DF["text"]).str.slice(0, 5000)  # cap to reduce memory
    VEC = TfidfVectorizer(max_features=4000, stop_words="english")  # slightly smaller
    TFIDF_MATRIX = VEC.fit_transform(DF["_full_text"])
except Exception as e:
    print(f"⚠️ Fallback dataset unavailable: {e}")

def find_local_evidence(news_text, top_k=4):
    """Return card-ready evidence from the local dataset (Archive)."""
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
            "url": "#",  # no external URL in local data
            "publishedAt": str(row.get("date", "")),
            "image": None,  # no image in local
            "badge": "Archive",
            "similarity": float(sims[i])
        })
    return rows


# ----------------------------
# Google News RSS Evidence (keyword-based)
# ----------------------------
STOPWORDS = {
    "the","is","am","are","was","were","be","been","to","of","and","in","on","at","a","an",
    "for","by","with","about","into","after","before","from","this","that","these","those",
    "as","it","its","their","his","her","you","your","we","our","they","them","i","will","shall",
    "not","no","but","or","if","so","than","then","because","while","over","under","between","within"
}

def _extract_keywords(text, cap=10):
    """Heuristic: keep meaningful words, prefer Proper Nouns and uniques; cap length."""
    import re
    if not text:
        return []
    # Keep words (letters only), track original tokens for capitalization signal
    tokens = re.findall(r"[A-Za-z][A-Za-z]+", text)
    lower = [t.lower() for t in tokens]
    keep = []
    seen = set()
    for i, tok in enumerate(tokens):
        lo = lower[i]
        if len(lo) <= 2 or lo in STOPWORDS:
            continue
        # prefer proper nouns (has uppercase anywhere besides first char) or all-caps acronyms
        is_proper = tok[0].isupper()
        is_acronym = tok.isupper() and len(tok) > 2
        if (is_proper or is_acronym or True) and lo not in seen:
            keep.append(lo)
            seen.add(lo)
        if len(keep) >= cap:
            break
    # Fallback: if nothing, split by spaces and pick first few words
    if not keep:
        words = re.findall(r"[A-Za-z]+", text.lower())
        keep = [w for w in words if w not in STOPWORDS and len(w) > 2][:cap]
    return keep



from urllib.parse import urlparse

def fetch_live_evidence(query, max_results=6):
    """Fetch live evidence from Google News RSS using extracted keywords and include images or favicons."""
    if feedparser is None:
        return []
    keywords = _extract_keywords(query, cap=10)
    keyword_query = " ".join(keywords) if keywords else query
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(keyword_query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, timeout=6, headers=headers)
        if r.status_code != 200 or not r.text:
            return []
        feed = feedparser.parse(r.text)
        out = []
        seen_links = set()
        for entry in getattr(feed, "entries", [])[:max_results*2]:
            link = entry.get("link") or "#"
            if link in seen_links:
                continue
            seen_links.add(link)

            # Extract image if available
            image_url = None
            if hasattr(entry, "media_content") and entry.media_content:
                try:
                    image_url = entry.media_content[0].get("url")
                except Exception:
                    image_url = None
            elif hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
                try:
                    image_url = entry.media_thumbnail[0].get("url")
                except Exception:
                    image_url = None

            # ✅ fallback to site favicon if no image
            if not image_url and link and link != "#":
                try:
                    domain = urlparse(link).netloc
                    if domain:
                        image_url = f"https://www.google.com/s2/favicons?sz=128&domain={domain}"
                except Exception:
                    image_url = None

            source = None
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source = entry.source.title

            card = {
                "headline": entry.get("title", "Untitled"),
                "source": source or "Google News",
                "url": link,
                "publishedAt": entry.get("published", ""),
                "image": image_url,   # ✅ favicon fallback ensures every card has an image
                "badge": "Live"
            }
            out.append(card)
            if len(out) >= max_results:
                break
        return out
    except Exception as e:
        print(f"⚠️ Google News fetch error: {e}")
        return []

    keywords = _extract_keywords(query, cap=10)
    keyword_query = " ".join(keywords) if keywords else query
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(keyword_query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, timeout=6, headers=headers)
        if r.status_code != 200 or not r.text:
            return []
        feed = feedparser.parse(r.text)
        out = []
        seen_links = set()
        for entry in getattr(feed, "entries", [])[:max_results*2]:
            link = entry.get("link") or "#"
            if link in seen_links:
                continue
            seen_links.add(link)

            # Extract image if available
            image_url = None
            if hasattr(entry, "media_content") and entry.media_content:
                try:
                    image_url = entry.media_content[0].get("url")
                except Exception:
                    image_url = None
            elif hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
                try:
                    image_url = entry.media_thumbnail[0].get("url")
                except Exception:
                    image_url = None

            source = None
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source = entry.source.title

            card = {
                "headline": entry.get("title", "Untitled"),
                "source": source or "Google News",
                "url": link,
                "publishedAt": entry.get("published", ""),
                "image": image_url,   # ✅ now includes image if present
                "badge": "Live"
            }
            out.append(card)
            if len(out) >= max_results:
                break
        return out
    except Exception as e:
        print(f"⚠️ Google News fetch error: {e}")
        return []

    keywords = _extract_keywords(query, cap=10)
    keyword_query = " ".join(keywords) if keywords else query
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(keyword_query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"}
        r = requests.get(url, timeout=6, headers=headers)
        # If request failed, return no evidence gracefully
        if r.status_code != 200 or not r.text:
            return []
        feed = feedparser.parse(r.text)
        out = []
        seen_links = set()
        for entry in getattr(feed, "entries", [])[:max_results*2]:  # pull extras then dedupe
            link = entry.get("link") or "#"
            if link in seen_links:
                continue
            seen_links.add(link)
            # Source
            source = None
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source = entry.source.title
            # Build card
            card = {
                "headline": entry.get("title", "Untitled"),
                "source": source or "Google News",
                "url": link,
                "publishedAt": entry.get("published", ""),
                "image": None,
                "badge": "Live"
            }
            out.append(card)
            if len(out) >= max_results:
                break
        return out
    except Exception as e:
        print(f"⚠️ Google News fetch error: {e}")
        return []

    except Exception:
        # Network/Quota issues — fallback
        return find_local_evidence(query)

# ----------------------------
# 3) Lightweight location hint
# ----------------------------
LOCATIONS = [
    "India","United States","USA","UK","Canada","Australia","New Zealand","Germany","France","Italy",
    "Delhi","Mumbai","Chennai","Bengaluru","Kolkata","Hyderabad","Pune","Gujarat","Uttar Pradesh",
    "New York","Washington","California","Texas","London","Sydney","Melbourne","Toronto","Vancouver"
]
def detect_location(text):
    for loc in LOCATIONS:
        if re.search(rf"\b{re.escape(loc)}\b", text, flags=re.IGNORECASE):
            return loc
    return None

# ----------------------------
# 4) Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Accept JSON {"text": "..."} or form field "newsText"
    news_text = ""
    if request.is_json:
        body = request.get_json(silent=True) or {}
        news_text = (body.get("text") or "").strip()
    else:
        news_text = (request.form.get("newsText") or "").strip()

    if not news_text:
        return jsonify({"error": "No news text provided"}), 400

    # Vectorize & predict
    real_prob = 0.5
    fake_prob = 0.5
    verdict = "Unknown"

    if model is not None and vectorizer is not None:
        try:
            x = vectorizer.transform([news_text])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                if POS_IDX is None:
                    # If we can't identify "real" class, treat highest prob as "real"
                    real_prob = float(max(probs))
                    fake_prob = float(1.0 - real_prob)
                else:
                    real_prob = float(probs[POS_IDX])
                    fake_prob = float(1.0 - real_prob)
                verdict = "Real" if real_prob >= fake_prob else "Fake"
            else:
                y = model.predict(x)[0]
                verdict = "Real" if str(y).lower() == "real" or str(y) == "1" else "Fake"
                real_prob = 0.75 if verdict == "Real" else 0.25
                fake_prob = 1.0 - real_prob
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            verdict = "Error"
    else:
        verdict = "Model not loaded"
        real_prob = 0.5
        fake_prob = 0.5

    # Evidence (Live -> Fallback)
    evidence_cards = fetch_live_evidence(news_text)

    # Location hint
    location = detect_location(news_text)

    return jsonify({
        "verdict": verdict,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "probability": real_prob if verdict == "Real" else fake_prob,  # for meter
        "evidence": evidence_cards,
        "location": location
    })

if __name__ == "__main__":
    # Optional: open browser automatically (Windows-friendly)
    try:
        import webbrowser
        webbrowser.open("http://127.0.0.1:5000", new=2)
    except Exception:
        pass
    port = int(os.environ.get("PORT", 5000))  # Render gives PORT env variable
    app.run(host="0.0.0.0", port=port, debug=False)

