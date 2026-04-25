# Phishing URL Detector 🎣

A machine learning web application designed to detect and classify malicious phishing URLs. This project features a custom Natural Language Processing (NLP) pipeline and a Logistic Regression classification model, served through a Flask REST API with a clean front-end interface.

## 🧠 Technical Architecture

### 1. Machine Learning & NLP
* **Algorithm**: Logistic Regression (Binary Classification: 'Good' vs 'Bad').
* **Dataset**: Trained on a dataset of 549,346 labeled URLs.
* **Feature Extraction**: 
  * Tokenized using NLTK `RegexpTokenizer`.
  * Noise reduction via English stop-word removal.
  * Root word extraction using NLTK `PorterStemmer`.
  * Vectorized into numerical arrays using strictly defined `CountVectorizer` rules.

### 2. Web Stack
* **Backend**: Python / Flask. Exposes a `/predict` POST endpoint that receives URLs, runs them through the exact training preprocessing pipeline, and returns the model's prediction.
* **Frontend**: Vanilla HTML, CSS, and JavaScript. Uses the asynchronous `fetch()` API to communicate with the Flask backend in real-time without page reloads.

## 📁 Project Structure
```text
phishing_url_detector/
├── app.py                 # Flask server and API endpoint logic
├── phishing.pkl           # Serialized Logistic Regression model
├── vectorizer.pkl         # Serialized CountVectorizer vocabulary
├── README.md              # Project documentation
└── templates/
    └── index.html         # Frontend user interface