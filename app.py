from flask import Flask, request, jsonify, render_template
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK data is available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Load the vectorizer and the classification model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('phishing.pkl', 'rb') as f:
    model = pickle.load(f)

# Recreate the exact preprocessing logic from your notebook
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

def process_tokens(tokens):
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    # CountVectorizer expects strings, so we join the stemmed tokens back together
    return ' '.join(stemmed_tokens)

@app.route('/')
def home():
    # Serves the frontend interface
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # 1. Tokenize the incoming URL
    tokens = tokenizer.tokenize(url)
    
    # 2. Apply lowercasing, stop-word removal, and stemming
    processed_text = process_tokens(tokens)
    
    # 3. Transform the processed text using the loaded CountVectorizer
    vectorized_url = vectorizer.transform([processed_text])
    
    # 4. Predict the class ('good' or 'bad')
    prediction = model.predict(vectorized_url)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)