"""
Extract sample news articles from the dataset for the browse page
"""
import pandas as pd
import json
import os

try:
    # Load the dataset
    df = pd.read_csv("sri_lankan_news_article_data.csv")
    
    # Clean data
    df = df.drop_duplicates()
    df = df[df['industry'] != "Here's the list of industries for the news items you've provided, using the expanded list of categories:"]
    
    # Industry mapping (same as training script)
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
    
    # Sample 50 articles per category (or all if less than 50)
    sampled_data = []
    for category in df['industry_clean'].unique():
        category_df = df[df['industry_clean'] == category]
        sample_size = min(30, len(category_df))  # 30 per category
        sample = category_df.sample(n=sample_size, random_state=42)
        sampled_data.append(sample)
    
    df_sample = pd.concat(sampled_data, ignore_index=True)
    
    # Prepare the data for JSON
    news_data = []
    for idx, row in df_sample.iterrows():
        news_data.append({
            'id': int(idx),
            'headline': str(row.get('Headline', '')),
            'description': str(row.get('News Description', ''))[:300],  # Truncate long descriptions
            'category': str(row['industry_clean']),
            'sentiment': str(row.get('sentiment', 'NEUTRAL')),
            'date': str(row.get('Date', ''))
        })
    
    # Save to JSON file
    os.makedirs('static/data', exist_ok=True)
    with open('static/data/news_sample.json', 'w', encoding='utf-8') as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Successfully saved {len(news_data)} news articles")
    print(f"  Categories: {df_sample['industry_clean'].nunique()}")
    print(f"  File: static/data/news_sample.json")
    
except FileNotFoundError:
    print("Warning: sri_lankan_news_article_data.csv not found")
    print("Creating sample mock data instead...")
    
    # Create mock data
    mock_data = [
        {
            'id': 1,
            'headline': 'Central Bank raises interest rates to combat inflation',
            'description': 'The Central Bank of Sri Lanka announced a 50 basis point increase in policy rates.',
            'category': 'Financial Services',
            'sentiment': 'NEGATIVE',
            'date': '2024-01-15'
        },
        {
            'id': 2,
            'headline': 'New tech startup launches AI platform',
            'description': 'A local technology company unveiled an innovative artificial intelligence solution.',
            'category': 'Commercial & Professional Services',
            'sentiment': 'POSITIVE',
            'date': '2024-01-16'
        },
        {
            'id': 3,
            'headline': 'Rice production expected to decline',
            'description': 'Farmers report significant crop losses due to the ongoing drought.',
            'category': 'Agriculture',
            'sentiment': 'NEGATIVE',
            'date': '2024-01-17'
        }
    ]
    
    os.makedirs('static/data', exist_ok=True)
    with open('static/data/news_sample.json', 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Created {len(mock_data)} mock news articles")
