"""
Train and save models for the News Categorization application
Run this script first to train models before starting the Flask app
"""

import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
import pickle
import os

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load data
print("Loading data...")
df = pd.read_csv("sri_lankan_news_article_data.csv")

# Data cleaning
print("Cleaning data...")
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['combined_text'])
df = df[df['industry'] != "Here's the list of industries for the news items you've provided, using the expanded list of categories:"]

# Industry mapping
industry_mapping = {
    "Banks": "Financial Services",
    "Diversified Financials": "Financial Services",
    "Insurance": "Financial Services",
    "Food Products": "Food Industry",
    "Food, Beverage & Tobacco": "Food Industry",
    "Food & Staples Retailing": "Food Industry",
    "Real Estate Management & Development": "Real Estate",
    "Real Estate Management&Development": "Real Estate",
    "Energy": "Energy & Utilities",
    "Utilities": "Energy & Utilities",
    "Consumer Services": "Consumer Sector",
    "Consumer Goods Industry": "Consumer Sector",
    "Consumer Durables & Apparel": "Consumer Sector",
    "Consumer Discretionary": "Consumer Sector",
    "Household & Personal Products": "Consumer Sector",
    "Retailing": "Consumer Sector",
    "Agricultural Raw Materials": "Agriculture"
}

df["industry_clean"] = df["industry"].replace(industry_mapping)

additional_merges = {
    "Health Care Equipment & Services": "Commercial & Professional Services",
    "Software & Services": "Commercial & Professional Services",
    "Telecommunication Services": "Commercial & Professional Services",
    "Real Estate": "Capital Goods",
    "Automobiles & Components": "Capital Goods",
    "Materials": "Capital Goods"
}

df["industry_clean"] = df["industry_clean"].replace(additional_merges)

# Combine headline and description
df['headline_desc'] = df['Headline'] + " " + df['News Description'].fillna('')

# Clean text
print("Preprocessing text...")
df['clean_text'] = df['headline_desc'].apply(clean_text)

# Balance dataset - downsample "Other" category
df_other = df[df["industry_clean"] == "Other"]
df_non_other = df[df["industry_clean"] != "Other"]
if len(df_other) > 200:
    df_other_downsampled = df_other.sample(n=200, random_state=42)
    df = pd.concat([df_non_other, df_other_downsampled])
    df = df.sample(frac=1, random_state=42)

# TF-IDF Vectorization
print("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000, min_df=3, ngram_range=(1,3))
X_text = tfidf.fit_transform(df['clean_text'])

# Prepare labels
le_industry = LabelEncoder()
y_industry = le_industry.fit_transform(df['industry_clean'])

le_sentiment = LabelEncoder()
y_sentiment = le_sentiment.fit_transform(df['sentiment'])

# Apply SMOTE for industry classification
print("Balancing classes with SMOTE...")
smote = SMOTE(random_state=42)
X_balanced, y_industry_balanced = smote.fit_resample(X_text, y_industry)

# Split data for industry classification
print("Splitting data for industry classification...")
X_train_ind, X_temp_ind, y_train_ind, y_temp_ind = train_test_split(
    X_balanced, y_industry_balanced, test_size=0.3, random_state=42, stratify=y_industry_balanced
)
X_val_ind, X_test_ind, y_val_ind, y_test_ind = train_test_split(
    X_temp_ind, y_temp_ind, test_size=0.5, random_state=42, stratify=y_temp_ind
)

# Train Industry Classification Model
print("Training industry classification model...")
svm_industry = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
svm_industry.fit(X_train_ind, y_train_ind)

# Evaluate industry model
from sklearn.metrics import accuracy_score
y_val_pred_ind = svm_industry.predict(X_val_ind)
val_acc_ind = accuracy_score(y_val_ind, y_val_pred_ind)
print(f"Industry Model - Validation Accuracy: {val_acc_ind:.4f}")

# Split data for sentiment analysis
print("Splitting data for sentiment classification...")
X_train_sent, X_temp_sent, y_train_sent, y_temp_sent = train_test_split(
    X_text, y_sentiment, test_size=0.3, random_state=42, stratify=y_sentiment
)
X_val_sent, X_test_sent, y_val_sent, y_test_sent = train_test_split(
    X_temp_sent, y_temp_sent, test_size=0.5, random_state=42, stratify=y_temp_sent
)

# Train Sentiment Classification Model
print("Training sentiment classification model...")
svm_sentiment = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42)
svm_sentiment.fit(X_train_sent, y_train_sent)

# Evaluate sentiment model
y_val_pred_sent = svm_sentiment.predict(X_val_sent)
val_acc_sent = accuracy_score(y_val_sent, y_val_pred_sent)
print(f"Sentiment Model - Validation Accuracy: {val_acc_sent:.4f}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save models
print("Saving models...")
with open('models/svm_industry_model.pkl', 'wb') as f:
    pickle.dump(svm_industry, f)

with open('models/svm_sentiment_model.pkl', 'wb') as f:
    pickle.dump(svm_sentiment, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('models/label_encoder_industry.pkl', 'wb') as f:
    pickle.dump(le_industry, f)

with open('models/label_encoder_sentiment.pkl', 'wb') as f:
    pickle.dump(le_sentiment, f)

print("\n✓ Models saved successfully!")
print(f"  - Industry classes: {list(le_industry.classes_)}")
print(f"  - Sentiment classes: {list(le_sentiment.classes_)}")
print(f"  - Industry validation accuracy: {val_acc_ind:.2%}")
print(f"  - Sentiment validation accuracy: {val_acc_sent:.2%}")
print("\nYou can now run 'python app.py' to start the web application!")
