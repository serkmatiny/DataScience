# 📊 คู่มือ Data Science ระดับกลาง (Intermediate) - ปี 2025

---

## ส่วนที่ 2: ระดับกลาง (Intermediate)

### 6. Exploratory Data Analysis (EDA)

**EDA คืออะไร?**
การสำรวจและทำความเข้าใจข้อมูลก่อนที่จะสร้างโมเดล เปรียบเทียบได้กับการเป็น "นักสืบ" ที่หาเบาะแสจากข้อมูล!

#### 🔍 ขั้นตอน EDA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. โหลดข้อมูล
df = pd.read_csv('data.csv')

# 2. ดูภาพรวมข้อมูล
print("=" * 60)
print("📋 Dataset Overview")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# 3. ตรวจสอบข้อมูลหาย (Missing Data)
print("\n🔍 Missing Data:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
}).sort_values('Missing Count', ascending=False)
print(missing_df[missing_df['Missing Count'] > 0])

# 4. สถิติพื้นฐาน
print("\n📊 Statistical Summary:")
print(df.describe())

# 5. ดูประเภทข้อมูล
print("\n📝 Data Types:")
print(df.dtypes)
print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 6. Unique Values
print("\n🎲 Unique Values:")
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")
```

#### 📈 Visualizations สำหรับ EDA

```python
# ตั้งค่า style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Distribution Plots
def plot_distributions(df, numerical_cols):
    """แสดง distribution ของตัวแปร numerical"""
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col].dropna(), bins=30,
                      edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')

        # เพิ่ม mean line
        mean_val = df[col].mean()
        axes[idx].axvline(mean_val, color='red',
                         linestyle='--', label=f'Mean: {mean_val:.2f}')
        axes[idx].legend()

    # ซ่อน axes ที่ไม่ใช้
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

# 2. Box Plots สำหรับหา Outliers
def plot_boxplots(df, numerical_cols):
    """แสดง box plots เพื่อหา outliers"""
    n_cols = len(numerical_cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    for idx, col in enumerate(numerical_cols):
        sns.boxplot(data=df, y=col, ax=axes[idx])
        axes[idx].set_title(f'Box Plot: {col}')

    plt.tight_layout()
    plt.show()

# 3. Correlation Heatmap
def plot_correlation(df):
    """แสดง correlation matrix"""
    # เลือกเฉพาะ numerical columns
    numerical_df = df.select_dtypes(include=[np.number])

    correlation = numerical_df.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f',
               cmap='RdYlBu_r', center=0, square=True,
               linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # แสดง correlations ที่สูงสุด
    corr_pairs = correlation.unstack()
    corr_pairs = corr_pairs[corr_pairs != 1.0]
    top_corr = corr_pairs.abs().sort_values(ascending=False).head(10)
    print("\n🔥 Top 10 Correlations:")
    print(top_corr)

# 4. Categorical Analysis
def plot_categorical(df, categorical_cols):
    """วิเคราะห์ตัวแปร categorical"""
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5*n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts().head(10)

        axes[idx].bar(range(len(value_counts)), value_counts.values,
                     color='skyblue', edgecolor='navy')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[idx].set_title(f'Top values in {col}')
        axes[idx].set_ylabel('Count')

        # เพิ่มค่าบน bar
        for i, v in enumerate(value_counts.values):
            axes[idx].text(i, v, str(v), ha='center', va='bottom')

    # ซ่อน axes ที่ไม่ใช้
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
```

#### 🎯 ตัวอย่างที่ 5: EDA บ้านในกรุงเทพ

```python
# สร้างข้อมูลตัวอย่าง: ราคาบ้านในกรุงเทพ
np.random.seed(42)

n_samples = 500
data = {
    'District': np.random.choice(['สาทร', 'สุขุมวิท', 'รัชดา', 'บางนา', 'ลาดพร้าว'], n_samples),
    'Size_sqm': np.random.randint(30, 200, n_samples),
    'Bedrooms': np.random.randint(1, 5, n_samples),
    'Bathrooms': np.random.randint(1, 4, n_samples),
    'Floor': np.random.randint(1, 30, n_samples),
    'Age_years': np.random.randint(0, 25, n_samples),
    'Near_BTS': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
    'Has_Parking': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
}

# คำนวณราคาตามปัจจัยต่างๆ
base_price = data['Size_sqm'] * 80000
district_multiplier = {
    'สาทร': 1.5, 'สุขุมวิท': 1.4, 'รัชดา': 1.2,
    'บางนา': 1.0, 'ลาดพร้าว': 1.1
}
data['Price_THB'] = [
    base_price[i] * district_multiplier[data['District'][i]] *
    (1.1 if data['Near_BTS'][i] else 1.0) *
    (0.95 - data['Age_years'][i] * 0.01) *
    (1 + np.random.uniform(-0.1, 0.1))
    for i in range(n_samples)
]

df_house = pd.DataFrame(data)

print("🏠 Bangkok Housing Price Analysis")
print("=" * 60)

# 1. ภาพรวม
print(f"\nDataset Size: {df_house.shape[0]} properties")
print(f"\nPrice Statistics (Million THB):")
price_million = df_house['Price_THB'] / 1_000_000
print(price_million.describe())

# 2. ราคาเฉลี่ยตามย่าน
print("\n💰 Average Price by District:")
district_price = df_house.groupby('District')['Price_THB'].mean() / 1_000_000
district_price = district_price.sort_values(ascending=False)
print(district_price.round(2))

# 3. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 ราคาตามย่าน
axes[0, 0].bar(district_price.index, district_price.values, color='skyblue')
axes[0, 0].set_title('Average Price by District', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Price (Million THB)')
axes[0, 0].tick_params(axis='x', rotation=45)

# 3.2 ราคาตามขนาด
axes[0, 1].scatter(df_house['Size_sqm'], df_house['Price_THB']/1_000_000,
                  alpha=0.5, c=df_house['Bedrooms'], cmap='viridis')
axes[0, 1].set_title('Price vs Size', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Size (sqm)')
axes[0, 1].set_ylabel('Price (Million THB)')
cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
cbar.set_label('Bedrooms')

# 3.3 ราคาใกล้ BTS vs ไม่ใกล้
bts_comparison = df_house.groupby('Near_BTS')['Price_THB'].mean() / 1_000_000
axes[1, 0].bar(['Not Near BTS', 'Near BTS'], bts_comparison.values,
              color=['coral', 'lightgreen'])
axes[1, 0].set_title('Price: Near BTS vs Not', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Average Price (Million THB)')
for i, v in enumerate(bts_comparison.values):
    axes[1, 0].text(i, v, f'{v:.2f}M', ha='center', va='bottom')

# 3.4 Age distribution
axes[1, 1].hist(df_house['Age_years'], bins=20, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Building Age Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Age (years)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# 4. Key Insights
print("\n💡 Key Insights:")
print(f"1. Most expensive district: {district_price.index[0]} ({district_price.values[0]:.2f}M THB)")
print(f"2. Near BTS premium: {((bts_comparison.iloc[1]/bts_comparison.iloc[0] - 1) * 100):.1f}%")
print(f"3. Average property age: {df_house['Age_years'].mean():.1f} years")
print(f"4. Price per sqm range: {(df_house['Price_THB']/df_house['Size_sqm']).min()/1000:.0f}k - "
      f"{(df_house['Price_THB']/df_house['Size_sqm']).max()/1000:.0f}k THB")
```

---

### 7. Data Preprocessing และ Feature Engineering

#### 🧹 Data Cleaning

**1. จัดการข้อมูลหาย (Missing Data)**

```python
import pandas as pd
import numpy as np

# สร้างข้อมูลตัวอย่างที่มี missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, np.nan, 50],
    'C': ['x', 'y', np.nan, 'w', 'z'],
    'D': [100, 200, 300, 400, 500]
})

print("Original Data:")
print(df)
print(f"\nMissing values:\n{df.isnull().sum()}")

# วิธีจัดการ Missing Data

# 1. ลบแถวที่มี missing
df_drop_rows = df.dropna()
print(f"\nAfter dropping rows: {df_drop_rows.shape}")

# 2. ลบคอลัมน์ที่มี missing > threshold
df_drop_cols = df.dropna(axis=1, thresh=4)  # เก็บคอลัมน์ที่มีค่า >= 4
print(f"After dropping columns: {df_drop_cols.shape}")

# 3. เติมด้วยค่าเฉลี่ย (สำหรับ numerical)
df_filled = df.copy()
df_filled['A'] = df_filled['A'].fillna(df_filled['A'].mean())
df_filled['B'] = df_filled['B'].fillna(df_filled['B'].median())

# 4. เติมด้วย forward fill / backward fill
df_filled['B'] = df['B'].fillna(method='ffill')  # ใช้ค่าก่อนหน้า

# 5. เติมด้วยค่าที่พบบ่อยสุด (สำหรับ categorical)
most_common = df['C'].mode()[0]
df_filled['C'] = df['C'].fillna(most_common)

print(f"\nAfter filling:")
print(df_filled)
```

**2. จัดการ Outliers**

```python
def detect_outliers_iqr(data):
    """หา outliers ด้วย IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

# ตัวอย่าง
data = pd.Series([10, 12, 14, 15, 13, 100, 11, 14, 16, 200])
outliers, lower, upper = detect_outliers_iqr(data)

print(f"Outliers: {outliers.values}")
print(f"Lower bound: {lower:.2f}")
print(f"Upper bound: {upper:.2f}")

# วิธีจัดการ outliers
# 1. ลบทิ้ง
data_no_outliers = data[(data >= lower) & (data <= upper)]

# 2. Cap ที่ bounds
data_capped = data.clip(lower=lower, upper=upper)

# 3. แปลงด้วย log transform
data_log = np.log1p(data)  # log(1 + x) เพื่อหลีกเลี่ยง log(0)

print(f"\nOriginal: {data.values}")
print(f"Capped: {data_capped.values}")
```

**3. Encoding Categorical Variables**

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# ข้อมูลตัวอย่าง
df = pd.DataFrame({
    'City': ['Bangkok', 'Chiang Mai', 'Phuket', 'Bangkok', 'Pattaya'],
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small'],
    'Price': [100, 80, 120, 110, 90]
})

print("Original Data:")
print(df)

# 1. Label Encoding (สำหรับ ordinal data)
le = LabelEncoder()
df['Size_Encoded'] = le.fit_transform(df['Size'])
print(f"\nLabel Encoding:")
print(df[['Size', 'Size_Encoded']])

# 2. One-Hot Encoding (สำหรับ nominal data)
df_onehot = pd.get_dummies(df, columns=['City'], prefix='City')
print(f"\nOne-Hot Encoding:")
print(df_onehot)

# 3. Ordinal Encoding (กำหนดลำดับเอง)
size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
df['Size_Ordinal'] = df['Size'].map(size_mapping)
print(f"\nOrdinal Encoding:")
print(df[['Size', 'Size_Ordinal']])
```

**4. Scaling และ Normalization**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ข้อมูลตัวอย่าง
data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45],
    'Salary': [30000, 50000, 60000, 80000, 100000],
    'Experience': [2, 5, 7, 10, 12]
})

print("Original Data:")
print(data)
print(f"\nStatistics:\n{data.describe()}")

# 1. Standardization (Z-score normalization)
# x_scaled = (x - mean) / std
scaler = StandardScaler()
data_standardized = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)
print(f"\nStandardized (mean=0, std=1):")
print(data_standardized)
print(f"Mean: {data_standardized.mean().values}")
print(f"Std: {data_standardized.std().values}")

# 2. Min-Max Scaling (0-1 range)
# x_scaled = (x - min) / (max - min)
minmax_scaler = MinMaxScaler()
data_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(data),
    columns=data.columns
)
print(f"\nMin-Max Scaled (0-1):")
print(data_minmax)

# 3. Robust Scaling (ใช้ median และ IQR, ดีกับ outliers)
robust_scaler = RobustScaler()
data_robust = pd.DataFrame(
    robust_scaler.fit_transform(data),
    columns=data.columns
)
print(f"\nRobust Scaled:")
print(data_robust)
```

#### 🛠️ Feature Engineering

**1. สร้าง Features ใหม่จาก Features เดิม**

```python
import pandas as pd
import numpy as np

# ข้อมูล E-commerce
df = pd.DataFrame({
    'Order_Date': pd.date_range('2025-01-01', periods=100, freq='D'),
    'Product_Price': np.random.randint(100, 1000, 100),
    'Quantity': np.random.randint(1, 10, 100),
    'Customer_Age': np.random.randint(18, 70, 100),
    'Delivery_Time_Hours': np.random.randint(24, 168, 100)
})

# Feature Engineering

# 1. จาก Date: แยกเป็น components
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Day'] = df['Order_Date'].dt.day
df['DayOfWeek'] = df['Order_Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Quarter'] = df['Order_Date'].dt.quarter

# 2. Interaction Features
df['Total_Amount'] = df['Product_Price'] * df['Quantity']
df['Price_Per_Day'] = df['Product_Price'] / df['Delivery_Time_Hours'] * 24

# 3. Binning (แบ่งกลุ่ม)
df['Age_Group'] = pd.cut(df['Customer_Age'],
                        bins=[0, 25, 35, 50, 100],
                        labels=['Young', 'Adult', 'Middle', 'Senior'])

df['Price_Category'] = pd.qcut(df['Product_Price'],
                               q=3,
                               labels=['Low', 'Medium', 'High'])

# 4. Polynomial Features
df['Age_Squared'] = df['Customer_Age'] ** 2
df['Price_Log'] = np.log1p(df['Product_Price'])

# 5. Aggregation Features (สำหรับ time series)
df['Rolling_Avg_7d'] = df['Total_Amount'].rolling(window=7).mean()
df['Cumulative_Sales'] = df['Total_Amount'].cumsum()

print("Feature Engineered Dataset:")
print(df.head(10))
print(f"\nNew features created: {len(df.columns) - 5}")
```

**2. Feature Selection**

```python
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# สร้างข้อมูลตัวอย่าง
X, y = make_regression(n_samples=200, n_features=20,
                      n_informative=5, random_state=42)

feature_names = [f'Feature_{i}' for i in range(20)]
X_df = pd.DataFrame(X, columns=feature_names)

# 1. Univariate Feature Selection
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X_df.columns[selector.get_support()].tolist()

print("Top 5 Features (Univariate):")
scores = pd.DataFrame({
    'Feature': feature_names,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)
print(scores.head(5))

# 2. Recursive Feature Elimination (RFE)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=5)
rfe.fit(X, y)

rfe_features = X_df.columns[rfe.support_].tolist()
print(f"\nTop 5 Features (RFE): {rfe_features}")

# 3. Feature Importance จาก Random Forest
rf.fit(X, y)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance (Random Forest):")
print(importance_df.head(10))

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'].head(10),
        importance_df['Importance'].head(10))
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

---

### 8. Machine Learning พื้นฐาน

#### 🤖 Machine Learning คืออะไร?

**คำตอบง่ายๆ:** การสอนคอมพิวเตอร์ให้เรียนรู้จากข้อมูล โดยไม่ต้องเขียนโปรแกรมแบบละเอียด!

#### ประเภทของ Machine Learning

```
1. Supervised Learning (มีคำตอบ)
   - Classification: ทำนายหมวดหมู่ (A, B, C)
   - Regression: ทำนายตัวเลข (1.5, 2.3, 4.8)

2. Unsupervised Learning (ไม่มีคำตอบ)
   - Clustering: จัดกลุ่มข้อมูล
   - Dimensionality Reduction: ลดมิติข้อมูล

3. Reinforcement Learning (เรียนรู้จากการลองผิดลองถูก)
   - Agent เรียนรู้จาก reward/penalty
```

#### 🎯 Linear Regression (ทำนายตัวเลข)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูล: ชั่วโมงศึกษา vs คะแนนสอบ
np.random.seed(42)
study_hours = np.random.randint(1, 10, 100).reshape(-1, 1)
exam_scores = (study_hours.flatten() * 9 +
              np.random.randint(-10, 10, 100) + 10)

# แบ่งข้อมูล train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    study_hours, exam_scores, test_size=0.2, random_state=42
)

# สร้างและ train model
model = LinearRegression()
model.fit(X_train, y_train)

# ทำนาย
y_pred = model.predict(X_test)

# ประเมินผล
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📊 Linear Regression Results")
print("=" * 50)
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")

# Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Scatter + Regression Line
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, alpha=0.5, color='red', label='Test data')
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Prediction')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Study Hours vs Exam Scores')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ทำนายคะแนนจากชั่วโมงที่ศึกษา
hours_to_predict = np.array([[3], [5], [8]])
predicted_scores = model.predict(hours_to_predict)
print(f"\n🎓 Score Predictions:")
for hours, score in zip(hours_to_predict.flatten(), predicted_scores):
    print(f"  {hours} hours → {score:.1f} points")
```

#### 🎲 Logistic Regression (Classification)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# สร้างข้อมูล: ทำนายว่าจะผ่านสอบหรือไม่
np.random.seed(42)
n_samples = 200

study_hours = np.random.randint(1, 12, n_samples)
attendance = np.random.randint(50, 100, n_samples)

# คำนวณโอกาสผ่าน (study_hours และ attendance มีผล)
pass_probability = (study_hours * 5 + attendance * 0.3 - 200) / 100
passed = (pass_probability + np.random.randn(n_samples) * 0.3 > 0).astype(int)

X = np.column_stack([study_hours, attendance])
y = passed

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# ทำนาย
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ประเมินผล
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("🎯 Classification Results")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                          target_names=['Failed', 'Passed']))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Failed', 'Passed'],
           yticklabels=['Failed', 'Passed'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ทำนายโอกาสผ่านสอบ
test_students = np.array([
    [3, 60],   # น้อย study, attendance กลาง
    [8, 90],   # เยอะ study, attendance สูง
    [5, 75]    # กลาง study, attendance กลาง
])

predictions = model.predict(test_students)
probabilities = model.predict_proba(test_students)

print(f"\n👨‍🎓 Student Predictions:")
for i, (hours, attend) in enumerate(test_students):
    result = "PASS ✓" if predictions[i] == 1 else "FAIL ✗"
    prob = probabilities[i][1] * 100
    print(f"  Student {i+1}: {hours}h study, {attend}% attendance")
    print(f"    → {result} (Confidence: {prob:.1f}%)")
```

#### 🌳 Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# ใช้ข้อมูลเดิม
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# ทำนาย
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# Visualize Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model,
         feature_names=['Study Hours', 'Attendance'],
         class_names=['Failed', 'Passed'],
         filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Visualization', fontsize=16)
plt.show()

# Feature Importance
importance = dt_model.feature_importances_
features = ['Study Hours', 'Attendance']

plt.figure(figsize=(8, 5))
plt.bar(features, importance, color=['skyblue', 'lightcoral'])
plt.title('Feature Importance')
plt.ylabel('Importance')
for i, v in enumerate(importance):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
plt.show()
```

---

### 9. Machine Learning ขั้นสูง

#### 🚀 Ensemble Methods

**Ensemble คืออะไร?**
การรวมหลายๆ โมเดลเข้าด้วยกัน เพื่อให้ได้ผลลัพธ์ที่ดีกว่าโมเดลเดี่ยว!

**1. Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# สร้างข้อมูล
X, y = make_classification(n_samples=1000, n_features=20,
                          n_informative=15, n_redundant=5,
                          random_state=42)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,      # จำนวน trees
    max_depth=10,          # ความลึกสูงสุดของ tree
    min_samples_split=5,   # ตัวอย่างขั้นต่ำในการ split
    random_state=42,
    n_jobs=-1              # ใช้ CPU ทุก core
)

rf_model.fit(X_train, y_train)

# ประเมินผลด้วย Cross-Validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

print("🌳 Random Forest Results")
print("=" * 50)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ทดสอบกับ test set
test_score = rf_model.score(X_test, y_test)
print(f"Test Set Accuracy: {test_score:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': [f'Feature_{i}' for i in range(20)],
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 5 Important Features:")
print(feature_importance.head())
```

**2. Gradient Boosting (XGBoost, LightGBM, CatBoost)**

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import time

# เตรียมข้อมูล
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}

# 1. XGBoost
print("🚀 Training XGBoost...")
start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_time = time.time() - start
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, xgb_pred),
    'time': xgb_time
}

# 2. LightGBM
print("⚡ Training LightGBM...")
start = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_time = time.time() - start
results['LightGBM'] = {
    'accuracy': accuracy_score(y_test, lgb_pred),
    'time': lgb_time
}

# 3. CatBoost
print("🐱 Training CatBoost...")
start = time.time()
cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=False
)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
cat_time = time.time() - start
results['CatBoost'] = {
    'accuracy': accuracy_score(y_test, cat_pred),
    'time': cat_time
}

# เปรียบเทียบผลลัพธ์
print("\n📊 Model Comparison")
print("=" * 60)
results_df = pd.DataFrame(results).T
print(results_df)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
axes[0].bar(results_df.index, results_df['accuracy'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0.8, 1.0])
for i, v in enumerate(results_df['accuracy']):
    axes[0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

# Training time comparison
axes[1].bar(results_df.index, results_df['time'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1].set_title('Training Time Comparison')
axes[1].set_ylabel('Time (seconds)')
for i, v in enumerate(results_df['time']):
    axes[1].text(i, v, f'{v:.3f}s', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"\n🏆 Best Model: {results_df['accuracy'].idxmax()}")
print(f"⚡ Fastest Model: {results_df['time'].idxmin()}")
```

---

### 10. AutoML ในปี 2025

**AutoML คืออะไร?**
เครื่องมือที่ทำ Machine Learning อัตโนมัติ - จาก data preprocessing ไปจนถึง model selection และ hyperparameter tuning!

#### 🎯 PyCaret: AutoML แบบง่าย

```python
from pycaret.classification import *
import pandas as pd
from sklearn.datasets import load_breast_cancer

# โหลดข้อมูล
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("🤖 PyCaret AutoML Demo")
print("=" * 60)

# 1. Setup (เตรียมข้อมูล)
print("\n1. Setting up...")
clf_setup = setup(
    data=df,
    target='target',
    session_id=42,
    verbose=False,
    silent=True
)

# 2. เปรียบเทียบโมเดลทั้งหมด
print("\n2. Comparing all models...")
best_models = compare_models(n_select=5, verbose=False)

# 3. เลือกโมเดลที่ดีที่สุด
print("\n3. Selecting best model...")
best_model = best_models[0]
print(f"Best Model: {type(best_model).__name__}")

# 4. Tune hyperparameters
print("\n4. Tuning hyperparameters...")
tuned_model = tune_model(best_model, verbose=False)

# 5. ประเมินผล
print("\n5. Evaluating model...")
evaluate_model(tuned_model)

# 6. ทำนาย
print("\n6. Making predictions...")
predictions = predict_model(tuned_model)
print(predictions.head())

# 7. บันทึกโมเดล
print("\n7. Saving model...")
save_model(tuned_model, 'best_cancer_model')
print("Model saved as 'best_cancer_model.pkl'")

# โหลดโมเดลกลับมาใช้
# loaded_model = load_model('best_cancer_model')
```

#### ⚡ Optuna: Hyperparameter Optimization

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# สร้างข้อมูล
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

def objective(trial):
    """Objective function สำหรับ Optuna"""

    # กำหนด hyperparameters ที่จะ optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42
    }

    # สร้างและทดสอบโมเดล
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()

    return score

print("🔍 Optuna Hyperparameter Optimization")
print("=" * 60)

# สร้าง study และ optimize
study = optuna.create_study(direction='maximize', study_name='RF_optimization')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ผลลัพธ์
print(f"\n✨ Best Trial:")
print(f"  Accuracy: {study.best_value:.4f}")
print(f"\n📋 Best Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Visualization
import plotly.graph_objects as go
from optuna.visualization import plot_optimization_history, plot_param_importances

# Optimization history
fig1 = plot_optimization_history(study)
fig1.update_layout(title="Optimization History")
fig1.show()

# Parameter importances
fig2 = plot_param_importances(study)
fig2.update_layout(title="Hyperparameter Importances")
fig2.show()

# Train final model with best parameters
final_model = RandomForestClassifier(**study.best_params)
final_model.fit(X, y)
print(f"\n🎉 Final model trained with best parameters!")
```

---

## 🎓 สรุปส่วนที่ 2: ระดับกลาง

### สิ่งที่เราได้เรียนรู้:

✅ **Exploratory Data Analysis (EDA)**
- การสำรวจและทำความเข้าใจข้อมูล
- Visualizations สำหรับ EDA
- หา patterns และ insights

✅ **Data Preprocessing**
- จัดการ Missing Data
- จัดการ Outliers
- Encoding Categorical Variables
- Scaling และ Normalization

✅ **Feature Engineering**
- สร้าง features ใหม่
- Feature Selection
- Binning และ Transformations

✅ **Machine Learning Basics**
- Linear Regression
- Logistic Regression
- Decision Tree

✅ **Advanced ML**
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Ensemble Methods

✅ **AutoML**
- PyCaret
- Optuna

### 🎯 แบบฝึกหัดท้ายบท:

**Challenge 1: Customer Churn Prediction**
ทำนายว่าลูกค้าจะยกเลิกบริการหรือไม่:
- Load และ clean data
- EDA และ feature engineering
- เปรียบเทียบหลาย models
- เลือก model ที่ดีที่สุด

**Challenge 2: House Price Prediction**
ทำนายราคาบ้าน:
- จัดการ missing values
- Feature engineering จาก features ที่มี
- ใช้ ensemble methods
- Tune hyperparameters

**Challenge 3: AutoML Competition**
ใช้ PyCaret หรือ Optuna:
- เลือก dataset ที่สนใจ
- ใช้ AutoML หา best model
- เปรียบเทียบกับ manual tuning

---

### 🚀 พร้อมไปต่อไหม?

ในส่วนถัดไป เราจะเรียนรู้:
- Deep Learning และ Neural Networks
- Generative AI และ LLMs
- Agentic AI
- Real-time Data Science
- Edge AI และ IoT
- Green AI และ Ethical AI

Let's level up! 🔥
