# 📰 News Classifier  

An AI-powered web application that detects **Fake vs Real News** using **Machine Learning (NLP + Logistic Regression)**.  
Built with **Python, Flask, and Scikit-learn** 🚀  

---

## 🌟 Features
✅ Classifies news as **Real** or **Fake**  
✅ Confidence scores for predictions  
✅ User-friendly Flask web interface  
✅ Evidence cards with professional UI  
✅ Trained on a merged dataset of real and fake news articles  

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
│── data.csv             # Dataset (ignored in GitHub)  
│── templates/           # HTML templates (index.html, result.html)  
│── static/              # CSS, JS, images  
│── requirements.txt     # Project dependencies  
│── README.md            # You are here

---

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

# Once training is done, app.py will automatically use news_model.pkl for predictions.

⚠️ Note:
The dataset used to build this project is not included in the repository (to keep the repo lightweight).
You can use any news dataset (e.g., Kaggle Fake News Dataset) or your own data.

