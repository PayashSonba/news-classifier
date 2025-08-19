# 📰 News Classifier – Fake vs Real News Detection  

[![Made with Flask](https://img.shields.io/badge/Made%20with-Flask-blue)](https://flask.palletsprojects.com/)  
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)](https://scikit-learn.org/)  
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-green)](https://render.com/)  

🔗 **Live Demo:** [https://your-app.onrender.com](https://news-classifier-mglr.onrender.com)  

---

## 🚀 About the Project
The **News Classifier** is a web app that helps you check if a news article is **Real or Fake**.  
It uses a trained Machine Learning model (Logistic Regression with TF-IDF features) and enhances predictions with **Google News RSS evidence**.  

---

## ✨ Features
- 🧠 ML-powered fake news detection  
- 📊 Confidence score (Real vs Fake probability)  
- 🔎 Evidence cards from **Google News RSS** + fallback to local dataset  
- 🌍 Simple, clean Flask-based web interface  
- ☁️ Deployed live on **Render**
  
---

## ⚙️ Tech Stack
- **Frontend:** HTML, CSS, JavaScript (custom styled evidence cards ✨)  
- **Backend:** Flask (Python)  
- **ML Model:** Logistic Regression + TF-IDF Vectorizer  
- **Other Tools:** Pandas, Scikit-learn, Joblib  

---

## 📂 Project Structure
```bash
News_Classifier/
│── app.py               # Flask web app  
│── train_model.py       # ML training script  
│── news_model.pkl       # Trained ML model (ignored in GitHub)
├── vectorizer.pkl       # Saved TF-IDF vectorizer used with the model  
│── data.csv             # Dataset (ignored in GitHub)  
│── templates/           # HTML templates (index.html, result.html)  
│── static/              # CSS, JS, images  
│── requirements.txt     # Project dependencies
├── Procfile             # Start command for Render
├── .gitignore           # Specifies files/folders ignored by Git
│── README.md            # You are here
```

<hr>

## 📊 Model Training

The repository includes **train_model.py**, which allows you to retrain the model if you have a dataset.  

### Steps to Retrain:
1. Place your dataset file (CSV) in the project folder.  
   - Make sure it has at least these columns:  
     - `title` (news headline)  
     - `text` (news article content)  
     - `label` (Fake = 0, Real = 1)  

2. Run the training script:
   ```bash
   python train_model.py

---

 Once training is done, app.py will automatically use news_model.pkl for predictions.

⚠️ Note:
The dataset used to build this project is not included in the repository (to keep the repo lightweight).
You can use any news dataset (e.g., Kaggle Fake News Dataset) or your own data.

