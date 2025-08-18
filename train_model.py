import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load your dataset
df = pd.read_csv("data.csv")

# Ensure text column exists
if "_full_text" not in df.columns:
    # Merge title & text if needed
    df["_full_text"] = df["title"] + " " + df["text"]

# Vectorize
vec = TfidfVectorizer(max_features=5000)
X = vec.fit_transform(df["_full_text"])
y = df["label"]  # Assuming 1 = Real, 0 = Fake

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "news_model.pkl")
joblib.dump(vec, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
