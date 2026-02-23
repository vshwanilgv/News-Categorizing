# 📰 News Categorization AI - Frontend Application

A machine learning-powered web application for categorizing Sri Lankan news articles by industry and analyzing sentiment.

## 🌟 Features

- **Industry Classification**: Automatically categorizes news articles into various industries (Financial Services, Energy & Utilities, Agriculture, etc.)
- **Sentiment Analysis**: Determines whether the news sentiment is POSITIVE or NEGATIVE
- **Confidence Scores**: Shows prediction confidence for better transparency
- **Top Predictions**: Displays top 3 industry predictions with confidence bars
- **Clean UI**: Modern, responsive design that works on all devices
- **Real-time Analysis**: Get instant results as you submit articles

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train and save the models:**
   ```bash
   python train_and_save_models.py
   ```
   
   This script will:
   - Load the `sri_lankan_news_article_data.csv` dataset
   - Clean and preprocess the data
   - Train SVM models for industry and sentiment classification
   - Save the trained models in the `models/` directory
   
   **Note**: Make sure you have the `sri_lankan_news_article_data.csv` file in the same directory.

3. **Start the Flask application:**
   ```bash
   python app.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## 📊 Model Performance

The application uses two LinearSVC models:
- **Industry Classification**: Multi-class classification with balanced class weights
- **Sentiment Analysis**: Binary classification (POSITIVE/NEGATIVE)

Both models use TF-IDF vectorization with:
- Max features: 5000
- N-gram range: (1, 3)
- Min document frequency: 3

## 🏗️ Project Structure

```
News Categorizing/
├── app.py                          # Flask web application
├── train_and_save_models.py        # Model training script
├── ml_project_214034c.py           # Original ML notebook code
├── requirements.txt                # Python dependencies
├── sri_lankan_news_article_data.csv  # Dataset (you need this)
├── models/                         # Trained models directory
│   ├── svm_industry_model.pkl
│   ├── svm_sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder_industry.pkl
│   └── label_encoder_sentiment.pkl
├── templates/
│   └── index.html                  # Frontend HTML
└── static/
    └── css/
        └── style.css               # Styling
```

## 🎯 Usage

1. **Enter a news headline** (required)
2. **Optionally add a description** for better accuracy
3. **Click "Analyze Article"**
4. **View the results:**
   - Primary industry category
   - Sentiment (POSITIVE/NEGATIVE) with confidence
   - Top 3 industry predictions
   - Processed text preview

## 🔧 API Endpoints

### `POST /predict`
Analyzes a news article and returns predictions.

**Request Body:**
```json
{
  "headline": "Central Bank raises interest rates",
  "description": "The Central Bank announced a policy rate increase..."
}
```

**Response:**
```json
{
  "success": true,
  "industry": "Financial Services",
  "sentiment": "NEGATIVE",
  "sentiment_confidence": 78.5,
  "top_industries": [
    {"industry": "Financial Services", "confidence": 85.2},
    {"industry": "Energy & Utilities", "confidence": 8.1},
    {"industry": "Agriculture", "confidence": 3.4}
  ],
  "cleaned_text": "central bank raises interest rates..."
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true
}
```

## 📝 Industry Categories

The model classifies news into the following categories:
- Financial Services
- Energy & Utilities
- Agriculture
- Food Industry
- Consumer Sector
- Capital Goods
- Commercial & Professional Services
- Transportation
- Media
- Other

## 🛠️ Technologies Used

- **Backend**: Flask (Python)
- **ML Libraries**: scikit-learn, NLTK, imbalanced-learn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Fonts**: Google Fonts (Inter)

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is open source and available for educational purposes.

## 🙋‍♂️ Support

If you encounter any issues:
1. Make sure you've installed all dependencies
2. Ensure the dataset file exists
3. Check that models are trained and saved properly
4. Verify Flask is running on port 5000

---

**Built with ❤️ using Machine Learning**
