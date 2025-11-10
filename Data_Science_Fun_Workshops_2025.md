# 🎮 Data Science Workshops สนุกๆ ปี 2025
**เรียนรู้ผ่านโปรเจกต์จริง (และมีมุกเพิ่มความสนุก!) 🎉**

---

## 📑 Workshops ทั้งหมด

### 🎯 ระดับกลาง
1. [Workshop: สร้างบอทพยากรณ์อากาศด้วย AI](#workshop-1-บอทพยากรณ์อากาศด้วย-ai)
2. [Workshop: ระบบตรวจจับหน้ากากอนามัย](#workshop-2-ระบบตรวจจับหน้ากากอนามัย)
3. [Workshop: แชทบอทตอบคำถามภาษาไทย](#workshop-3-แชทบอทตอบคำถามภาษาไทย)

### 🚀 ระดับสูง
4. [Workshop: สร้างเกมด้วย Reinforcement Learning](#workshop-4-สร้างเกมด้วย-rl)
5. [Workshop: Deepfake Detector](#workshop-5-deepfake-detector)
6. [Workshop: AI ช่วยแต่งเพลง](#workshop-6-ai-ช่วยแต่งเพลง)

---

## Workshop 1: บอทพยากรณ์อากาศด้วย AI 🌤️

### 🎯 เป้าหมาย
สร้างบอทที่:
- ทำนายอุณหภูมิ 7 วันข้างหน้า
- บอกว่าควรเอาร่มไปหรือเปล่า
- แนะนำการแต่งกาย

**มุก:** หมอดูอากาศยุคใหม่ - แต่แม่นกว่าหมอดูทั่วไป! 🔮

### 💻 Code เต็ม

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from datetime import datetime, timedelta

print("🌤️ Weather Forecast Bot with AI")
print("=" * 60)

# ===== ส่วนที่ 1: สร้างข้อมูลอุณหภูมิ =====

def generate_weather_data(days=365):
    """
    สร้างข้อมูลอุณหภูมิจำลอง

    มุก: จำลองสภาพอากาศไทย - ร้อนแน่นอน แต่บางทีฝนตก! ☀️🌧️
    """
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

    # สร้าง pattern อุณหภูมิ
    # - Trend: ร้อนขึ้นช่วงเมษายน-พฤษภาคม
    # - Seasonality: รูปแบบตามฤดูกาล
    # - Noise: สุ่มนิดหน่อย
    trend = 28 + 4 * np.sin(2 * np.pi * np.arange(days) / 365)
    seasonality = 2 * np.sin(4 * np.pi * np.arange(days) / 365)
    noise = np.random.randn(days) * 1.5
    temperature = trend + seasonality + noise

    # เพิ่มความชื้น (humidity)
    humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.randn(days) * 10

    # ฝนตก (rain)
    rain_probability = 0.3 + 0.2 * np.sin(2 * np.pi * np.arange(days) / 365)
    rain = (np.random.rand(days) < rain_probability).astype(int)

    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'rain': rain
    })

    return df

# สร้างข้อมูล
df_weather = generate_weather_data(365 * 2)  # 2 years
print(f"\n📊 Weather Data: {len(df_weather)} days")
print(df_weather.head())

# ===== ส่วนที่ 2: LSTM Model สำหรับพยากรณ์ =====

class WeatherLSTM(nn.Module):
    """
    LSTM สำหรับพยากรณ์อากาศ

    มุก: LSTM มี "ความจำ" - จำว่าเมื่อวานอากาศเป็นยังไง
    (ไม่เหมือนคน บางทีลืม!)
    """

    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 3)  # temperature, humidity, rain

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # เอา output ของ timestep สุดท้าย
        last_output = lstm_out[:, -1, :]

        # Prediction
        output = self.fc(last_output)

        return output

# สร้างโมเดล
model = WeatherLSTM()
print(f"\n🧠 Model Architecture:")
print(model)

# ===== ส่วนที่ 3: เตรียมข้อมูลสำหรับ training =====

def prepare_sequences(data, seq_length=7):
    """สร้าง sequences สำหรับ LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Normalize data
scaler = MinMaxScaler()
features = ['temperature', 'humidity', 'rain']
scaled_data = scaler.fit_transform(df_weather[features].values)

# สร้าง sequences
seq_length = 7  # ใช้ 7 วันย้อนหลัง
X, y = prepare_sequences(scaled_data, seq_length)

print(f"\n📦 Training Data:")
print(f"  X shape: {X.shape} (samples, seq_length, features)")
print(f"  y shape: {y.shape}")

# แปลงเป็น PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# ===== ส่วนที่ 4: Training =====

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\n🔥 Training Model...")
num_epochs = 50
losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses, linewidth=2)
plt.title('📉 Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===== ส่วนที่ 5: พยากรณ์อากาศ! =====

def forecast_weather(model, last_sequence, days=7, scaler=None):
    """
    พยากรณ์อากาศ X วันข้างหน้า!

    มุก: คล้ายการทำนายอนาคต แต่ใช้ AI แทนลูกแก้ว! 🔮
    """
    model.eval()
    forecasts = []

    current_seq = torch.FloatTensor(last_sequence).unsqueeze(0)

    with torch.no_grad():
        for _ in range(days):
            # ทำนายวันถัดไป
            next_pred = model(current_seq)
            forecasts.append(next_pred.squeeze().numpy())

            # อัพเดท sequence
            current_seq = torch.cat([
                current_seq[:, 1:, :],
                next_pred.unsqueeze(1)
            ], dim=1)

    # Inverse transform
    forecasts = np.array(forecasts)
    if scaler:
        forecasts = scaler.inverse_transform(forecasts)

    return forecasts

# ทำนาย 7 วันข้างหน้า
last_sequence = scaled_data[-seq_length:]
forecast = forecast_weather(model, last_sequence, days=7, scaler=scaler)

print(f"\n🌤️ พยากรณ์อากาศ 7 วันข้างหน้า:")
print("=" * 60)

today = datetime.now()
for i, pred in enumerate(forecast):
    date = today + timedelta(days=i+1)
    temp = pred[0]
    humidity = pred[1]
    rain_prob = pred[2]

    # คำแนะนำ
    if temp > 35:
        temp_advice = "ร้อนมาก! อยู่ในร่ม 🥵"
    elif temp > 30:
        temp_advice = "ร้อน ควรดื่มน้ำเยอะๆ 💧"
    elif temp > 25:
        temp_advice = "อุณหภูมิพอดี ☺️"
    else:
        temp_advice = "เย็นสบาย 😌"

    umbrella = "🌂 เอาร่ม!" if rain_prob > 0.5 else "☀️ ไม่ต้องร่ม"

    print(f"\n📅 {date.strftime('%d/%m/%Y')} ({date.strftime('%A')})")
    print(f"  🌡️ อุณหภูมิ: {temp:.1f}°C - {temp_advice}")
    print(f"  💧 ความชื้น: {humidity:.1f}%")
    print(f"  {umbrella} (โอกาสฝน: {rain_prob*100:.0f}%)")

# ===== ส่วนที่ 6: Visualization =====

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Historical data
hist_days = 30
historical = df_weather.iloc[-hist_days:][features].values

# Create future dates
future_dates = [today + timedelta(days=i+1) for i in range(7)]
all_dates = list(df_weather['date'].iloc[-hist_days:]) + future_dates

# 1. Temperature
axes[0].plot(range(hist_days), historical[:, 0],
            'b-', linewidth=2, label='Historical')
axes[0].plot(range(hist_days, hist_days+7), forecast[:, 0],
            'r--', linewidth=2, marker='o', label='Forecast')
axes[0].axvline(hist_days, color='gray', linestyle=':', alpha=0.5)
axes[0].set_title('🌡️ Temperature Forecast', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Temperature (°C)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Humidity
axes[1].plot(range(hist_days), historical[:, 1],
            'b-', linewidth=2, label='Historical')
axes[1].plot(range(hist_days, hist_days+7), forecast[:, 1],
            'r--', linewidth=2, marker='o', label='Forecast')
axes[1].axvline(hist_days, color='gray', linestyle=':', alpha=0.5)
axes[1].set_title('💧 Humidity Forecast', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Humidity (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Rain Probability
axes[2].plot(range(hist_days), historical[:, 2],
            'b-', linewidth=2, label='Historical')
axes[2].plot(range(hist_days, hist_days+7), forecast[:, 2],
            'r--', linewidth=2, marker='o', label='Forecast')
axes[2].axvline(hist_days, color='gray', linestyle=':', alpha=0.5)
axes[2].fill_between(range(hist_days, hist_days+7), 0, forecast[:, 2],
                     alpha=0.3, color='blue')
axes[2].set_title('🌧️ Rain Probability', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Days')
axes[2].set_ylabel('Probability')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Weather Forecast Bot Complete!")
```

### 🎯 Challenge: ทำให้ดีขึ้น!

1. **เพิ่ม features:**
   - ความเร็วลม
   - ทิศทางลม
   - ดัชนี UV

2. **ปรับปรุงโมเดล:**
   - ใช้ Transformer แทน LSTM
   - Ensemble หลายโมเดล
   - Attention mechanism

3. **สร้าง API:**
   - FastAPI endpoint
   - รับ location
   - ส่ง forecast กลับ

**มุก:** AI พยากรณ์อากาศ - แม่นขนาดไหนก็อย่าลืมเอาร่มไว้ในกระเป๋านะ! 🌂

---

## Workshop 2: ระบบตรวจจับหน้ากากอนามัย 😷

### 🎯 เป้าหมาย
สร้าง AI ที่:
- ตรวจจับใบหน้า
- บอกว่าใส่หน้ากากหรือไม่
- Real-time detection

**มุก:** AI เป็นยามรักษาความปลอดภัย - ไม่เคยเหนื่อย ไม่เคยนอน! 💪

### 💻 Code เต็ม

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("😷 Mask Detection System with AI")
print("=" * 60)

# ===== ส่วนที่ 1: CNN Model สำหรับ Classification =====

class MaskDetectorCNN(nn.Module):
    """
    CNN สำหรับตรวจจับหน้ากาก

    Architecture:
    - 3 Convolutional blocks
    - MaxPooling
    - Fully connected layers
    - Binary classification (Mask / No Mask)

    มุก: CNN เห็นรูปแบบ (patterns) เหมือนเรารู้จักเพื่อนจากทรงผม! 👀
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Mask / No Mask
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# สร้างโมเดล
model = MaskDetectorCNN()
print(f"\n🧠 Model Summary:")
print(model)
print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===== ส่วนที่ 2: สร้างข้อมูล Training (จำลอง) =====

def create_synthetic_face_data(num_samples=1000):
    """
    สร้างข้อมูลใบหน้าจำลอง

    มุก: ในโลกจริง ต้องใช้รูปจริง
    แต่เพื่อการเรียนรู้ เราจำลองขึ้นมา! 🎭
    """
    X = []
    y = []

    for i in range(num_samples):
        # สร้างภาพสุ่ม (128x128x3)
        img = np.random.rand(128, 128, 3)

        # จำลอง: ครึ่งหนึ่งใส่หน้ากาก (เพิ่ม pattern ด้านล่าง)
        has_mask = i < num_samples // 2

        if has_mask:
            # เพิ่ม "หน้ากาก" (สี่เหลี่ยมด้านล่าง)
            img[80:120, 30:98, :] = [0.2, 0.5, 0.8]  # สีฟ้า
            label = 1
        else:
            # ไม่มีหน้ากาก (ปล่อยเป็น random)
            label = 0

        X.append(img)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)

# สร้างข้อมูล
print(f"\n📊 Creating Training Data...")
X_train, y_train = create_synthetic_face_data(1000)
X_test, y_test = create_synthetic_face_data(200)

print(f"Train: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test: {X_test.shape}, Labels: {y_test.shape}")
print(f"Mask samples: {(y_train == 1).sum()}")
print(f"No mask samples: {(y_train == 0).sum()}")

# แปลงเป็น PyTorch format
X_train_tensor = torch.from_numpy(X_train).permute(0, 3, 1, 2)
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).permute(0, 3, 1, 2)
y_test_tensor = torch.from_numpy(y_test).long()

# ===== ส่วนที่ 3: Training =====

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\n🔥 Training Model...")
num_epochs = 20
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]

        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / (len(X_train_tensor) // batch_size)

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# ===== ส่วนที่ 4: Evaluation =====

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = 100 * (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"\n📊 Test Accuracy: {accuracy:.2f}%")

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

print(f"\n📋 Classification Report:")
print(classification_report(y_test_tensor.numpy(), predicted.numpy(),
                          target_names=['No Mask', 'Mask']))

# ===== ส่วนที่ 5: Real-time Detection (Simulation) =====

def detect_mask(image, model):
    """
    ตรวจจับหน้ากากในภาพ

    Returns:
        - has_mask: True/False
        - confidence: 0-1

    มุก: AI มองภาพเร็วกว่าคน - ในเวลาที่เรากระพริบตา
    AI วิเคราะห์เสร็จแล้ว! ⚡
    """
    model.eval()
    with torch.no_grad():
        # Preprocess
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        # Predict
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        has_mask = predicted_class == 1

    return has_mask, confidence

# ทดสอบกับรูปในชุด test
print(f"\n🔍 Testing Detection...")
print("=" * 60)

# สุ่มเลือกรูปมาทดสอบ
test_indices = np.random.choice(len(X_test), 6, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, test_idx in enumerate(test_indices):
    img = X_test[test_idx]
    true_label = y_test[test_idx]

    # Detect
    has_mask, confidence = detect_mask(img, model)

    # Plot
    axes[idx].imshow(img)
    axes[idx].axis('off')

    # Title with result
    predicted_label = "😷 Mask" if has_mask else "😊 No Mask"
    true_label_text = "😷 Mask" if true_label == 1 else "😊 No Mask"
    color = 'green' if has_mask == (true_label == 1) else 'red'

    title = f"True: {true_label_text}\n"
    title += f"Pred: {predicted_label}\n"
    title += f"Conf: {confidence:.2%}"

    axes[idx].set_title(title, fontsize=10, color=color, fontweight='bold')

plt.suptitle('🎯 Mask Detection Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ===== ส่วนที่ 6: Real-time Video (Concept) =====

print(f"\n📹 Real-time Video Detection (Concept):")
print("=" * 60)
print("""
def process_video_stream():
    '''
    Process video in real-time

    Steps:
    1. Capture frame from camera
    2. Detect faces (using OpenCV)
    3. Classify each face (Mask / No Mask)
    4. Draw bounding boxes + labels
    5. Display result

    มุก: เหมือนมี "ผู้เชี่ยวชาญ" คอยดูกล้อง 24/7
    (แต่ไม่เหนื่อย ไม่ง่วงนอน! 🤖)
    '''

    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = face_detector.detect(frame)

        for (x, y, w, h) in faces:
            # Crop face
            face_img = frame[y:y+h, x:x+w]

            # Resize to model input size
            face_resized = cv2.resize(face_img, (128, 128))

            # Detect mask
            has_mask, confidence = detect_mask(face_resized, model)

            # Draw box and label
            color = (0, 255, 0) if has_mask else (0, 0, 255)
            label = f"Mask: {confidence:.2%}" if has_mask else f"No Mask: {confidence:.2%}"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display
        cv2.imshow('Mask Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
""")

print("\n✅ Mask Detection System Complete!")
print("\n💡 Applications:")
print("  - ทางเข้าออฟฟิศ/โรงพยาบาล")
print("  - ห้างสรรพสินค้า")
print("  - สนามบิน/สถานีรถไฟ")
print("  - โรงเรียน/มหาวิทยาลัย")
```

### 🎯 Challenge: ปรับปรุงระบบ!

1. **Improve Model:**
   - ใช้ pre-trained model (MobileNet, ResNet)
   - Transfer learning
   - Data augmentation

2. **Add Features:**
   - ตรวจจับว่าใส่ถูกวิธีหรือไม่ (mask on chin?)
   - นับจำนวนคนที่ไม่ใส่หน้ากาก
   - Alert system

3. **Deploy:**
   - Raspberry Pi + Camera
   - Edge deployment
   - Mobile app

**มุก:** AI ตรวจหน้ากาก - ไม่ใช่เพื่อจับผิด แต่เพื่อปกป้องทุกคน! 💙

---

## 🎉 สรุป Fun Workshops

### สิ่งที่เราได้เรียนรู้:

✅ **Weather Forecast Bot** 🌤️
- LSTM สำหรับ time series
- Multi-step prediction
- การแนะนำเชิงปฏิบัติ

✅ **Mask Detection** 😷
- CNN สำหรับภาพ
- Real-time processing
- Practical AI application

### 🚀 Next Level

พร้อมสำหรับ workshops ขั้นสูงหรือยัง?
- Reinforcement Learning Game
- Deepfake Detector
- AI Music Composer

**มุก สุดท้าย:**
```python
while learning:
    build_projects()
    have_fun()
    become_expert()

print("You're awesome! 🌟")
```

**Happy Workshop-ing! 🎊**
