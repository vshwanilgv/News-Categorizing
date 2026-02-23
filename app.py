from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
import nltk
import pandas as pd
from scipy.sparse import hstack
import numpy as np

# Download stopwords if not already present
try:
    stop_words = set(stopwords.words('english')) - {"not", "no", "never"}
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

app = Flask(__name__)

# Load trained models and vectorizers
try:
    with open('models/svm_industry_model.pkl', 'rb') as f:
        svm_industry = pickle.load(f)
    with open('models/svm_sentiment_model.pkl', 'rb') as f:
        svm_sentiment = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/label_encoder_industry.pkl', 'rb') as f:
        le_industry = pickle.load(f)
    with open('models/label_encoder_sentiment.pkl', 'rb') as f:
        le_sentiment = pickle.load(f)
    
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    print("Please train models first using train_and_save_models.py")
    MODELS_LOADED = False

def clean_text(text):
    """Clean and preprocess text"""
    # Lowercase
    text = text.lower()
    # Remove punctuation & special chars
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED:
        return jsonify({
            'error': 'Models not loaded. Please train models first using train_and_save_models.py'
        }), 500
    
    try:
        data = request.get_json()
        headline = data.get('headline', '')
        description = data.get('description', '')
        
        if not headline and not description:
            return jsonify({'error': 'Please provide at least a headline or description'}), 400
        
        # Combine headline and description
        combined_text = f"{headline} {description}".strip()
        
        # Clean text
        clean_combined = clean_text(combined_text)
        
        # Vectorize
        X_text = tfidf.transform([clean_combined])
        
        # Predict industry
        industry_pred = int(svm_industry.predict(X_text)[0])
        industry_label = str(le_industry.inverse_transform([industry_pred])[0])
        
        # Get prediction probabilities (decision function for SVM)
        industry_scores = svm_industry.decision_function(X_text)
        if industry_scores.ndim > 1:
            industry_scores = industry_scores[0]
        
        # Apply softmax to convert scores to probabilities
        exp_scores = np.exp(industry_scores - np.max(industry_scores))  # For numerical stability
        industry_probs = exp_scores / np.sum(exp_scores)
        
        # Get top 3 industries
        top_3_idx = np.argsort(industry_probs)[-3:][::-1]
        top_industries = []
        for idx in top_3_idx:
            idx = int(idx)
            industry_name = str(le_industry.classes_[idx])
            confidence_val = float(industry_probs[idx]) * 100.0
            top_industries.append({
                'industry': industry_name,
                'confidence': confidence_val
            })
        
        # Predict sentiment
        sentiment_pred = int(svm_sentiment.predict(X_text)[0])
        sentiment_label = str(le_sentiment.inverse_transform([sentiment_pred])[0])
        
        # Calculate sentiment confidence
        sentiment_scores = svm_sentiment.decision_function(X_text)
        # Extract scalar value properly - ensure it's a single float
        if isinstance(sentiment_scores, np.ndarray):
            if sentiment_scores.ndim > 1:
                sentiment_score = float(sentiment_scores[0][0])
            else:
                sentiment_score = float(sentiment_scores[0])
        else:
            sentiment_score = float(sentiment_scores)
        
        # Convert to probability-like score (0-100)
        sentiment_confidence = float(min(abs(sentiment_score) * 10.0, 100.0))
        
        return jsonify({
            'success': True,
            'industry': str(industry_label),
            'sentiment': str(sentiment_label),
            'sentiment_confidence': round(sentiment_confidence, 2),
            'top_industries': top_industries,
            'cleaned_text': str(clean_combined[:200] + '...' if len(clean_combined) > 200 else clean_combined)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': MODELS_LOADED
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
