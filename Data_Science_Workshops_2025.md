# 🎨 Data Science Workshops - ปี 2025
**โปรเจกต์จริง สนุก และนำไปใช้งานได้ทันที!**

---

## ส่วนที่ 4: Workshops และโปรเจกต์จริง

### Workshop 1: วิเคราะห์ความรู้สึกในโซเชียลมีเดีย 😊😢😡

#### 🎯 เป้าหมาย
สร้างระบบวิเคราะห์ความรู้สึก (Sentiment Analysis) จากความคิดเห็นในโซเชียลมีเดีย เพื่อรู้ว่าลูกค้าคิดอย่างไรกับสินค้า/บริการ

#### 📚 ความรู้ที่ใช้
- Text preprocessing
- TF-IDF vectorization
- Machine Learning classification
- Hugging Face Transformers

#### 💻 โค้ดเต็ม

```python
# workshop1_sentiment_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import string
from transformers import pipeline

# ===== ส่วนที่ 1: เตรียมข้อมูล =====

print("😊 Workshop 1: Sentiment Analysis")
print("=" * 60)

# สร้างข้อมูลตัวอย่าง (ในงานจริงใช้ Twitter API หรือ web scraping)
data = {
    'text': [
        # Positive
        "I love this product! Best purchase ever!",
        "Amazing quality, highly recommend!",
        "Excellent service, will buy again!",
        "Fantastic! Exceeded my expectations!",
        "Great value for money, very satisfied!",
        "Outstanding quality, love it!",
        "Super happy with this purchase!",
        "Brilliant product, works perfectly!",
        "Absolutely love it, 5 stars!",
        "Best decision ever made!",
        # Negative
        "Terrible quality, waste of money!",
        "Worst product ever, don't buy!",
        "Poor service, very disappointed!",
        "Awful experience, not recommended!",
        "Bad quality, broke after one day!",
        "Horrible, complete waste!",
        "Very disappointed, poor value!",
        "Terrible service, avoid this!",
        "Worst purchase of my life!",
        "Extremely poor quality!",
        # Neutral
        "It's okay, nothing special.",
        "Average product, works as expected.",
        "Decent quality for the price.",
        "Not bad, not great either.",
        "Standard quality, no complaints.",
        "Fair enough, does the job.",
        "Acceptable quality, okay service.",
        "Medium quality, as described.",
        "Normal product, nothing fancy.",
        "Moderate satisfaction, it's fine."
    ],
    'sentiment': ['positive'] * 10 + ['negative'] * 10 + ['neutral'] * 10
}

df = pd.DataFrame(data)
print(f"\n📊 Dataset: {len(df)} reviews")
print(f"\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# ===== ส่วนที่ 2: Text Preprocessing =====

def preprocess_text(text):
    """ทำความสะอาดข้อความ"""
    # แปลงเป็นตัวพิมพ์เล็ก
    text = text.lower()

    # ลบ URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # ลบ mentions และ hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # ลบเครื่องหมายวรรคตอน
    text = text.translate(str.maketrans('', '', string.punctuation))

    # ลบช่องว่างเกิน
    text = ' '.join(text.split())

    return text

df['text_clean'] = df['text'].apply(preprocess_text)

print("\n🧹 Text Preprocessing Example:")
print(f"Original: {df['text'].iloc[0]}")
print(f"Cleaned:  {df['text_clean'].iloc[0]}")

# ===== ส่วนที่ 3: Feature Extraction =====

print("\n📝 Feature Extraction with TF-IDF...")

vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text_clean'])
y = df['sentiment']

print(f"Feature matrix shape: {X.shape}")
print(f"Top features: {vectorizer.get_feature_names_out()[:10]}")

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ===== ส่วนที่ 4: Train Models =====

print("\n🤖 Training Models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    print(f"Accuracy: {accuracy:.4f}")

# ===== ส่วนที่ 5: Evaluation =====

print("\n📊 Evaluation Results")
print("=" * 60)

best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, best_predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=sorted(df['sentiment'].unique()),
           yticklabels=sorted(df['sentiment'].unique()))
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('sentiment_confusion_matrix.png')
plt.show()

# ===== ส่วนที่ 6: Deep Learning with Transformers =====

print("\n🤗 Using Pretrained Transformer Model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

test_texts = [
    "This product is absolutely amazing!",
    "Terrible experience, very disappointed.",
    "It's okay, nothing special."
]

print("\n🔍 Predictions with Transformer:")
for text in test_texts:
    result = sentiment_pipeline(text)[0]
    print(f"\nText: {text}")
    print(f"Sentiment: {result['label']} (confidence: {result['score']:.4f})")

# ===== ส่วนที่ 7: Interactive Prediction =====

def predict_sentiment(text, model=best_model, vectorizer=vectorizer):
    """ทำนายความรู้สึกจากข้อความ"""
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    return {
        'text': text,
        'sentiment': prediction,
        'confidence': max(probabilities),
        'probabilities': dict(zip(model.classes_, probabilities))
    }

# ทดสอบ
print("\n🎯 Interactive Predictions:")
print("=" * 60)

sample_reviews = [
    "This is the best thing I've ever bought! Highly recommend!",
    "Don't waste your money, total garbage!",
    "It works, but nothing impressive."
]

for review in sample_reviews:
    result = predict_sentiment(review)
    print(f"\n📝 Review: {result['text']}")
    print(f"😊 Sentiment: {result['sentiment'].upper()}")
    print(f"💯 Confidence: {result['confidence']:.2%}")
    print(f"📊 Probabilities: {result['probabilities']}")

# ===== ส่วนที่ 8: Visualization =====

print("\n📈 Creating Visualizations...")

# Sentiment distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution
df['sentiment'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red', 'gray'])
axes[0].set_title('Sentiment Distribution (Training Data)')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Model comparison
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
axes[1].bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
axes[1].set_title('Model Accuracy Comparison')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1])
for i, v in enumerate(accuracies):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png')
plt.show()

print("\n✅ Workshop 1 Completed!")
print("Saved: sentiment_confusion_matrix.png")
print("Saved: sentiment_analysis_results.png")
```

#### 🎯 แบบฝึกหัด

**Challenge 1: ขยายข้อมูล**
- ดาวน์โหลดข้อมูลจริงจาก Twitter API หรือ Kaggle
- เพิ่มข้อมูลให้มีอย่างน้อย 1,000 reviews

**Challenge 2: ปรับปรุงโมเดล**
- ลองใช้ BERT, RoBERTa แทน DistilBERT
- Fine-tune model กับข้อมูลของตัวเอง
- เปรียบเทียบผลลัพธ์

**Challenge 3: Dashboard**
- สร้าง web dashboard ด้วย Streamlit
- Real-time sentiment analysis
- แสดงกราฟและสถิติแบบ interactive

---

### Workshop 2: สร้างระบบแนะนำสินค้า 🛒

#### 🎯 เป้าหมาย
สร้าง Recommendation System แบบ Collaborative Filtering เพื่อแนะนำสินค้าให้ลูกค้า (เหมือน Netflix, Amazon)

#### 📚 ความรู้ที่ใช้
- Collaborative Filtering
- Matrix Factorization
- Cosine Similarity
- Neural Collaborative Filtering

#### 💻 โค้ดเต็ม

```python
# workshop2_recommendation_system.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

print("🛒 Workshop 2: Recommendation System")
print("=" * 60)

# ===== ส่วนที่ 1: สร้างข้อมูล =====

np.random.seed(42)

# จำลองข้อมูล: users, products, ratings
n_users = 100
n_products = 50
n_ratings = 2000

users = [f"User_{i:03d}" for i in range(1, n_users + 1)]
products = [f"Product_{i:02d}" for i in range(1, n_products + 1)]

# สร้าง ratings (1-5)
ratings_data = {
    'user_id': np.random.choice(users, n_ratings),
    'product_id': np.random.choice(products, n_ratings),
    'rating': np.random.randint(1, 6, n_ratings)
}

df_ratings = pd.DataFrame(ratings_data)
df_ratings = df_ratings.drop_duplicates(subset=['user_id', 'product_id'])

print(f"\n📊 Dataset:")
print(f"Users: {df_ratings['user_id'].nunique()}")
print(f"Products: {df_ratings['product_id'].nunique()}")
print(f"Ratings: {len(df_ratings)}")
print(f"\nRating Distribution:")
print(df_ratings['rating'].value_counts().sort_index())

# ===== ส่วนที่ 2: Exploratory Analysis =====

print("\n📈 Exploratory Analysis")

# User activity
user_activity = df_ratings.groupby('user_id').size()
print(f"\nAverage ratings per user: {user_activity.mean():.2f}")
print(f"Most active user: {user_activity.idxmax()} ({user_activity.max()} ratings)")

# Product popularity
product_popularity = df_ratings.groupby('product_id').size()
print(f"\nAverage ratings per product: {product_popularity.mean():.2f}")
print(f"Most popular product: {product_popularity.idxmax()} ({product_popularity.max()} ratings)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Rating distribution
df_ratings['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Rating Distribution')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Count')

# 2. User activity distribution
axes[0, 1].hist(user_activity, bins=20, color='lightcoral', edgecolor='black')
axes[0, 1].set_title('User Activity Distribution')
axes[0, 1].set_xlabel('Number of Ratings')
axes[0, 1].set_ylabel('Number of Users')

# 3. Product popularity distribution
axes[1, 0].hist(product_popularity, bins=20, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Product Popularity Distribution')
axes[1, 0].set_xlabel('Number of Ratings')
axes[1, 0].set_ylabel('Number of Products')

# 4. Average rating by product
avg_rating_by_product = df_ratings.groupby('product_id')['rating'].mean().sort_values(ascending=False).head(10)
axes[1, 1].barh(range(len(avg_rating_by_product)), avg_rating_by_product.values, color='orange')
axes[1, 1].set_yticks(range(len(avg_rating_by_product)))
axes[1, 1].set_yticklabels(avg_rating_by_product.index)
axes[1, 1].set_title('Top 10 Highest Rated Products')
axes[1, 1].set_xlabel('Average Rating')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('recommendation_eda.png')
plt.show()

# ===== ส่วนที่ 3: User-Item Matrix =====

print("\n🔢 Creating User-Item Matrix...")

# สร้าง pivot table
user_item_matrix = df_ratings.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating',
    fill_value=0
)

print(f"Matrix shape: {user_item_matrix.shape}")
print(f"Sparsity: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.2%}")

# ===== ส่วนที่ 4: Item-Based Collaborative Filtering =====

print("\n🤝 Item-Based Collaborative Filtering")

# คำนวณ similarity ระหว่างสินค้า
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

def get_similar_items(product_id, n=5):
    """หาสินค้าที่คล้ายกัน"""
    if product_id not in item_similarity_df.columns:
        return []

    similarities = item_similarity_df[product_id].sort_values(ascending=False)
    similar_items = similarities.iloc[1:n+1]  # ข้าม item ตัวเอง

    return similar_items

# ทดสอบ
test_product = products[0]
similar_products = get_similar_items(test_product, n=5)

print(f"\n🔍 Similar products to {test_product}:")
for product, similarity in similar_products.items():
    print(f"  {product}: {similarity:.4f}")

def recommend_items_for_user(user_id, n=5):
    """แนะนำสินค้าให้ user"""
    if user_id not in user_item_matrix.index:
        return []

    # หาสินค้าที่ user เคย rate
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    # คำนวณ predicted ratings สำหรับสินค้าที่ยังไม่ได้ rate
    unrated_items = user_ratings[user_ratings == 0].index
    predictions = {}

    for item in unrated_items:
        # หา similar items ที่ user เคย rate
        similar_items = item_similarity_df[item]
        relevant_similarities = similar_items[rated_items]

        if len(relevant_similarities) > 0:
            # Weighted average
            numerator = (relevant_similarities * user_ratings[rated_items]).sum()
            denominator = relevant_similarities.sum()
            if denominator > 0:
                predictions[item] = numerator / denominator

    # Sort และเลือก top N
    recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]

    return recommendations

# ทดสอบ
test_user = users[0]
recommendations = recommend_items_for_user(test_user, n=5)

print(f"\n🎁 Recommendations for {test_user}:")
for product, score in recommendations:
    print(f"  {product}: {score:.4f}")

# ===== ส่วนที่ 5: Matrix Factorization (SVD) =====

print("\n📊 Matrix Factorization with SVD")

# SVD
n_components = 10
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_.T

print(f"User factors shape: {user_factors.shape}")
print(f"Item factors shape: {item_factors.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

# Reconstruct ratings
predicted_ratings = np.dot(user_factors, item_factors.T)
predicted_ratings_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)

def recommend_with_svd(user_id, n=5):
    """แนะนำด้วย SVD"""
    if user_id not in predicted_ratings_df.index:
        return []

    # หาสินค้าที่ยังไม่ได้ rate
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index

    # ดึง predicted ratings
    predictions = predicted_ratings_df.loc[user_id, unrated_items]
    top_recommendations = predictions.sort_values(ascending=False).head(n)

    return list(zip(top_recommendations.index, top_recommendations.values))

# ทดสอบ
svd_recommendations = recommend_with_svd(test_user, n=5)

print(f"\n🎁 SVD Recommendations for {test_user}:")
for product, score in svd_recommendations:
    print(f"  {product}: {score:.4f}")

# ===== ส่วนที่ 6: Evaluation =====

print("\n📊 Evaluation")

# Train/Test split
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df_ratings, test_size=0.2, random_state=42)

# สร้าง matrix จาก train data
train_matrix = train_data.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating',
    fill_value=0
)

# Train SVD
svd_eval = TruncatedSVD(n_components=10, random_state=42)
user_factors_eval = svd_eval.fit_transform(train_matrix)
item_factors_eval = svd_eval.components_.T
predicted_eval = np.dot(user_factors_eval, item_factors_eval.T)

# Calculate RMSE
def calculate_rmse(actual, predicted, test_data):
    """Calculate RMSE"""
    errors = []
    for _, row in test_data.iterrows():
        user = row['user_id']
        item = row['product_id']
        actual_rating = row['rating']

        if user in train_matrix.index and item in train_matrix.columns:
            user_idx = train_matrix.index.get_loc(user)
            item_idx = train_matrix.columns.get_loc(item)
            pred_rating = predicted[user_idx, item_idx]
            errors.append((actual_rating - pred_rating) ** 2)

    rmse = np.sqrt(np.mean(errors))
    return rmse

rmse = calculate_rmse(test_data, predicted_eval, test_data)
print(f"\n📊 RMSE: {rmse:.4f}")

# ===== ส่วนที่ 7: Visualization =====

# Heatmap ของ user-item interactions
plt.figure(figsize=(14, 8))
sample_users = user_item_matrix.index[:20]
sample_items = user_item_matrix.columns[:20]
sample_matrix = user_item_matrix.loc[sample_users, sample_items]

sns.heatmap(sample_matrix, cmap='YlOrRd', cbar_kws={'label': 'Rating'})
plt.title('User-Item Interaction Matrix (Sample)')
plt.xlabel('Products')
plt.ylabel('Users')
plt.tight_layout()
plt.savefig('user_item_matrix.png')
plt.show()

print("\n✅ Workshop 2 Completed!")
print("Saved: recommendation_eda.png")
print("Saved: user_item_matrix.png")
```

#### 🎯 แบบฝึกหัด

**Challenge 1: Hybrid Recommender**
- รวม Content-Based + Collaborative Filtering
- ใช้ product features (category, price, brand)
- เปรียบเทียบผลลัพธ์

**Challenge 2: Deep Learning**
- ใช้ Neural Collaborative Filtering (NCF)
- สร้าง autoencoder สำหรับ recommendations
- ประเมินผล performance

**Challenge 3: Cold Start Problem**
- จัดการกับ user/item ใหม่
- Implement popularity-based fallback
- A/B testing

---

### Workshop 3: ทำนายราคาหุ้นด้วย Real-time ML 📈

#### 🎯 เป้าหมาย
สร้างระบบทำนายราคาหุ้นแบบ real-time ด้วย Time Series Analysis และ LSTM

#### 📚 ความรู้ที่ใช้
- Time Series Analysis
- LSTM (Long Short-Term Memory)
- Technical Indicators
- Real-time Processing

#### 💻 โค้ดเต็ม

```python
# workshop3_stock_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("📈 Workshop 3: Stock Price Prediction")
print("=" * 60)

# ===== ส่วนที่ 1: สร้างข้อมูล Time Series =====

np.random.seed(42)

# สร้างข้อมูลราคาหุ้นจำลอง
dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
n_days = len(dates)

# Simulate stock price with trend + seasonality + noise
trend = np.linspace(100, 150, n_days)
seasonality = 10 * np.sin(np.linspace(0, 8*np.pi, n_days))
noise = np.random.randn(n_days) * 5
price = trend + seasonality + noise

# เพิ่ม random jumps (news events)
jump_indices = np.random.choice(n_days, size=20, replace=False)
price[jump_indices] += np.random.randn(20) * 15

df_stock = pd.DataFrame({
    'Date': dates,
    'Close': price
})

# คำนวณ technical indicators
df_stock['Returns'] = df_stock['Close'].pct_change()
df_stock['MA_7'] = df_stock['Close'].rolling(window=7).mean()
df_stock['MA_30'] = df_stock['Close'].rolling(window=30).mean()
df_stock['Volatility'] = df_stock['Returns'].rolling(window=30).std()
df_stock['RSI'] = 100 - (100 / (1 + df_stock['Returns'].rolling(14).mean() /
                               abs(df_stock['Returns'].rolling(14).mean())))

df_stock = df_stock.dropna()

print(f"\n📊 Stock Data:")
print(f"Period: {df_stock['Date'].min()} to {df_stock['Date'].max()}")
print(f"Days: {len(df_stock)}")
print(f"\n{df_stock.head()}")

# ===== ส่วนที่ 2: Exploratory Analysis =====

print("\n📈 Exploratory Analysis")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. Price และ Moving Averages
axes[0].plot(df_stock['Date'], df_stock['Close'], label='Close Price', linewidth=2)
axes[0].plot(df_stock['Date'], df_stock['MA_7'], label='MA 7', linestyle='--')
axes[0].plot(df_stock['Date'], df_stock['MA_30'], label='MA 30', linestyle='--')
axes[0].set_title('Stock Price and Moving Averages', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Returns
axes[1].plot(df_stock['Date'], df_stock['Returns'], color='orange', alpha=0.7)
axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1].set_title('Daily Returns', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Returns')
axes[1].grid(True, alpha=0.3)

# 3. Volatility
axes[2].plot(df_stock['Date'], df_stock['Volatility'], color='red', linewidth=2)
axes[2].set_title('30-Day Volatility', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Volatility')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stock_eda.png')
plt.show()

# Statistics
print(f"\n📊 Price Statistics:")
print(f"Mean: ${df_stock['Close'].mean():.2f}")
print(f"Std: ${df_stock['Close'].std():.2f}")
print(f"Min: ${df_stock['Close'].min():.2f}")
print(f"Max: ${df_stock['Close'].max():.2f}")
print(f"\n📊 Returns Statistics:")
print(f"Mean: {df_stock['Returns'].mean():.4f}")
print(f"Std: {df_stock['Returns'].std():.4f}")

# ===== ส่วนที่ 3: Prepare Data for LSTM =====

print("\n🔢 Preparing Data for LSTM...")

def create_sequences(data, seq_length):
    """สร้าง sequences สำหรับ LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_stock[['Close']].values)

# สร้าง sequences
seq_length = 30  # ใช้ 30 วันย้อนหลังเพื่อทำนาย
X, y = create_sequences(scaled_data, seq_length)

print(f"Sequences shape: X={X.shape}, y={y.shape}")

# แบ่ง train/test (80/20)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# แปลงเป็น PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

print(f"Train: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# ===== ส่วนที่ 4: LSTM Model =====

print("\n🧠 Building LSTM Model...")

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # เอา output จาก time step สุดท้าย
        last_output = lstm_out[:, -1, :]

        # Fully connected
        prediction = self.fc(last_output)

        return prediction

# สร้าง model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockLSTM().to(device)

print(model)

# Loss และ optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===== ส่วนที่ 5: Training =====

print("\n🔥 Training Model...")

num_epochs = 100
batch_size = 32
losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_tensor[i:i+batch_size].to(device)
        batch_y = y_train_tensor[i:i+batch_size].to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (len(X_train) // batch_size)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.title('Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.savefig('lstm_training_loss.png')
plt.show()

# ===== ส่วนที่ 6: Evaluation =====

print("\n📊 Evaluating Model...")

model.eval()
with torch.no_grad():
    # Predictions
    train_pred = model(X_train_tensor.to(device)).cpu().numpy()
    test_pred = model(X_test_tensor.to(device)).cpu().numpy()

# Inverse transform
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)
y_train_actual = scaler.inverse_transform(y_train)
y_test_actual = scaler.inverse_transform(y_test)

# Metrics
train_mae = mean_absolute_error(y_train_actual, train_pred)
test_mae = mean_absolute_error(y_test_actual, test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))

print(f"\n📈 Results:")
print(f"Train MAE: ${train_mae:.2f}")
print(f"Test MAE: ${test_mae:.2f}")
print(f"Train RMSE: ${train_rmse:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Train predictions
axes[0].plot(y_train_actual, label='Actual', linewidth=2)
axes[0].plot(train_pred, label='Predicted', linestyle='--', linewidth=2)
axes[0].set_title('Training Set Predictions', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test predictions
axes[1].plot(y_test_actual, label='Actual', linewidth=2)
axes[1].plot(test_pred, label='Predicted', linestyle='--', linewidth=2)
axes[1].set_title('Test Set Predictions', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('Price')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_predictions.png')
plt.show()

# ===== ส่วนที่ 7: Real-time Prediction Simulation =====

print("\n⚡ Real-time Prediction Simulation...")

def predict_next_price(model, last_sequence, scaler, device):
    """ทำนายราคาถัดไป"""
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        prediction = model(sequence_tensor).cpu().numpy()
        actual_price = scaler.inverse_transform(prediction)[0][0]
    return actual_price

# จำลอง real-time prediction
print("\n🔮 Next 10 Days Predictions:")
current_sequence = scaled_data[-seq_length:].copy()

for day in range(1, 11):
    next_price = predict_next_price(model, current_sequence, scaler, device)
    print(f"Day +{day}: ${next_price:.2f}")

    # Update sequence
    next_price_scaled = scaler.transform([[next_price]])
    current_sequence = np.vstack([current_sequence[1:], next_price_scaled])

print("\n✅ Workshop 3 Completed!")
print("Saved: stock_eda.png")
print("Saved: lstm_training_loss.png")
print("Saved: lstm_predictions.png")
```

#### 🎯 แบบฝึกหัด

**Challenge 1: Advanced Features**
- เพิ่ม technical indicators (MACD, Bollinger Bands)
- ใช้ news sentiment analysis
- รวม multiple stocks

**Challenge 2: Model Improvement**
- ลอง GRU, Transformer models
- Implement attention mechanism
- Ensemble methods

**Challenge 3: Production System**
- Real-time data streaming
- Auto-retraining
- Alert system
- Web dashboard

---

### 🎓 สรุป Workshops

#### สิ่งที่เราได้เรียนรู้:

✅ **Workshop 1: Sentiment Analysis**
- Text preprocessing
- TF-IDF vectorization
- Classification models
- Transformers

✅ **Workshop 2: Recommendation System**
- Collaborative Filtering
- Matrix Factorization (SVD)
- Item-based recommendations
- Evaluation metrics

✅ **Workshop 3: Stock Prediction**
- Time series analysis
- LSTM networks
- Real-time predictions
- Technical indicators

---

### 🚀 Next Steps

**ขั้นตอนต่อไป:**

1. **ทำโปรเจกต์ของคุณเอง**
   - เลือกปัญหาที่สนใจ
   - หาข้อมูล
   - สร้างและประเมินโมเดล
   - Deploy!

2. **เรียนรู้เพิ่มเติม**
   - Kaggle competitions
   - Research papers
   - Online courses
   - Community forums

3. **Build Portfolio**
   - GitHub repositories
   - Blog posts
   - YouTube tutorials
   - Open source contributions

---

### 📚 Resources

**Datasets:**
- Kaggle: https://www.kaggle.com/datasets
- UCI ML Repository: https://archive.ics.uci.edu/ml
- Google Dataset Search: https://datasetsearch.research.google.com

**Learning:**
- Fast.ai: https://www.fast.ai
- Coursera ML Courses
- YouTube (StatQuest, 3Blue1Brown)
- Papers with Code: https://paperswithcode.com

**Tools:**
- Hugging Face: https://huggingface.co
- PyTorch: https://pytorch.org
- TensorFlow: https://tensorflow.org
- Scikit-learn: https://scikit-learn.org

---

### 🎉 ขอให้สนุกกับการเรียนรู้ Data Science!

**Remember:**
- 💪 Practice makes perfect
- 🤝 Learn from community
- 🚀 Build real projects
- 🌱 Keep growing!

Happy Data Science Journey! 🎊🎓📊
