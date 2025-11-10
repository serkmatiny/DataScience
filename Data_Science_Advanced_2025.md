# 🚀 คู่มือ Data Science ระดับสูง (Advanced) - ปี 2025

---

## ส่วนที่ 3: ระดับสูง (Advanced)

### 11. Deep Learning และ Neural Networks

#### 🧠 Neural Networks คืออะไร?

**คำตอบง่ายๆ:** โมเดลที่เลียนแบบสมองมนุษย์ มีหลายชั้น (layers) ที่เรียนรู้ patterns ที่ซับซ้อน!

```
Input Layer → Hidden Layers → Output Layer
    ↓            ↓                ↓
  [X1]       [Neurons]         [Predictions]
  [X2]    [Neurons] [Neurons]
  [X3]       [Neurons]
```

#### 🔥 PyTorch: Deep Learning Framework

**ทำไมเลือก PyTorch?**
- ยืดหยุ่น เหมาะกับการ research
- Community ใหญ่
- Dynamic computation graph
- นิยมมากในปี 2025!

**สร้าง Neural Network ง่ายๆ:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# เช็คว่ามี GPU หรือไม่
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# สร้างข้อมูลตัวอย่าง
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)

# แปลงเป็น PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# สร้าง DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# กำหนด Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# สร้างโมเดล
model = SimpleNN(input_size=10, hidden_size=64, output_size=1).to(device)
print(model)

# Loss function และ optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\n🔥 Training Neural Network...")
losses = []
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.show()

# ทดสอบโมเดล
model.eval()
with torch.no_grad():
    test_X = torch.randn(10, 10).to(device)
    predictions = model(test_X)
    print("\n📊 Sample Predictions:")
    print(predictions.cpu().numpy().flatten())
```

#### 🖼️ Convolutional Neural Networks (CNN) - สำหรับภาพ

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# โหลด MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# สร้างและ train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🖼️ Training CNN on MNIST...")
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {running_loss/200:.4f}")
            running_loss = 0.0

# ทดสอบโมเดล
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n✨ Test Accuracy: {accuracy:.2f}%")
```

---

### 12. Generative AI และ LLMs

#### 🤖 Large Language Models (LLMs) ในปี 2025

**เทรนด์สำคัญ:**
- Small Language Models (SLMs) < 10B parameters
- Efficient และประหยัดพลังงาน
- Local deployment ได้ง่าย

#### 🔧 ใช้งาน Hugging Face Transformers

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

print("🤗 Hugging Face Transformers Demo")
print("=" * 60)

# 1. Sentiment Analysis
print("\n1️⃣ Sentiment Analysis")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I love this product! It's amazing!",
    "This is terrible, worst purchase ever.",
    "It's okay, nothing special."
]

for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}")
    print(f"  → {result['label']} (confidence: {result['score']:.4f})")

# 2. Text Generation
print("\n2️⃣ Text Generation")
generator = pipeline("text-generation", model="gpt2")

prompt = "In 2025, Data Science has evolved to"
generated = generator(
    prompt,
    max_length=100,
    num_return_sequences=2,
    temperature=0.7
)

print(f"Prompt: {prompt}")
for i, gen in enumerate(generated, 1):
    print(f"\nGeneration {i}:")
    print(gen['generated_text'])

# 3. Named Entity Recognition (NER)
print("\n3️⃣ Named Entity Recognition")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Elon Musk founded SpaceX in California and Tesla in 2003."
entities = ner(text)

print(f"Text: {text}")
print("Entities found:")
for entity in entities:
    print(f"  - {entity['word']}: {entity['entity']} "
          f"(score: {entity['score']:.4f})")

# 4. Question Answering
print("\n4️⃣ Question Answering")
qa = pipeline("question-answering")

context = """
Data Science in 2025 focuses on AI-augmented workflows, real-time analytics,
and green AI practices. The field emphasizes ethical AI, with tools like
AutoML making machine learning accessible to everyone. Edge AI enables
processing at the source, crucial for IoT applications.
"""

questions = [
    "What does Data Science focus on in 2025?",
    "Why is Edge AI important?",
    "What makes machine learning accessible?"
]

print(f"Context: {context[:100]}...")
for question in questions:
    answer = qa(question=question, context=context)
    print(f"\nQ: {question}")
    print(f"A: {answer['answer']} (score: {answer['score']:.4f})")

# 5. Text Summarization
print("\n5️⃣ Text Summarization")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

long_text = """
Artificial Intelligence and Machine Learning have revolutionized the way we
process and analyze data. In 2025, the focus has shifted towards more
sustainable and ethical AI practices. Companies are now prioritizing Green AI,
which aims to reduce the environmental impact of training large models.
Additionally, Small Language Models are gaining popularity as they offer
comparable performance to larger models while consuming significantly less
energy. The democratization of AI through AutoML tools has made it possible
for non-experts to build sophisticated machine learning models. Real-time
data processing has become crucial for applications ranging from financial
trading to IoT devices, enabling instant decision-making capabilities.
"""

summary = summarizer(
    long_text,
    max_length=60,
    min_length=30,
    do_sample=False
)[0]

print(f"Original length: {len(long_text)} chars")
print(f"Summary length: {len(summary['summary_text'])} chars")
print(f"\nSummary: {summary['summary_text']}")
```

#### 💬 สร้าง Chatbot ด้วย LangChain

```python
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline

print("🤖 Building Chatbot with LangChain")
print("=" * 60)

# สร้าง text generation pipeline
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_length=200
)

# Wrap ด้วย LangChain
llm = HuggingFacePipeline(pipeline=generator)

# สร้าง prompt template
template = """
You are a helpful Data Science assistant. Answer the question clearly and concisely.

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# สร้าง chain
chain = LLMChain(llm=llm, prompt=prompt)

# ทดสอบ chatbot
questions = [
    "What is machine learning?",
    "How does AutoML work?",
    "What are the benefits of Green AI?"
]

for question in questions:
    print(f"\n❓ {question}")
    response = chain.run(question=question)
    print(f"🤖 {response}")
```

---

### 13. Agentic AI

#### 🎯 Agentic AI คืออะไร?

**คำนิยาม:** AI ที่สามารถทำงานอัตโนมัติ ตัดสินใจ และดำเนินการเองได้ โดยไม่ต้องมีคนคอยสั่ง!

**ตัวอย่างการใช้งาน:**
- Virtual assistants ที่ช่วยจัดการงานอัตโนมัติ
- Automated data analysis และ reporting
- Self-optimizing systems

#### 🤖 สร้าง Simple Agentic System

```python
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: int
    name: str
    description: str
    status: TaskStatus
    priority: int  # 1-5 (5 = highest)
    dependencies: List[int]  # Task IDs

class DataScienceAgent:
    """
    Autonomous Agent ที่จัดการ Data Science tasks
    """

    def __init__(self, name: str):
        self.name = name
        self.tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.knowledge_base: Dict[str, Any] = {}

    def add_task(self, task: Task):
        """เพิ่ม task ใหม่"""
        self.tasks.append(task)
        print(f"✅ Task added: {task.name} (Priority: {task.priority})")

    def can_execute_task(self, task: Task) -> bool:
        """เช็คว่า task พร้อมทำหรือยัง (dependencies สำเร็จหรือยัง)"""
        if not task.dependencies:
            return True

        completed_ids = [t.id for t in self.completed_tasks]
        return all(dep_id in completed_ids for dep_id in task.dependencies)

    def prioritize_tasks(self) -> List[Task]:
        """จัดลำดับ tasks ตาม priority และ dependencies"""
        pending_tasks = [t for t in self.tasks
                        if t.status == TaskStatus.PENDING and self.can_execute_task(t)]
        return sorted(pending_tasks, key=lambda t: t.priority, reverse=True)

    def execute_task(self, task: Task):
        """ทำงาน task"""
        print(f"\n🔄 Executing: {task.name}")
        print(f"   Description: {task.description}")

        task.status = TaskStatus.IN_PROGRESS

        # Simulate work
        time.sleep(1)

        # Random success/failure (90% success rate)
        if random.random() < 0.9:
            task.status = TaskStatus.COMPLETED
            self.completed_tasks.append(task)
            self.tasks.remove(task)

            # Store knowledge
            self.knowledge_base[task.name] = {
                'completed_at': time.time(),
                'result': 'success'
            }

            print(f"   ✅ Completed: {task.name}")
        else:
            task.status = TaskStatus.FAILED
            print(f"   ❌ Failed: {task.name}")

    def run(self):
        """Agent ทำงานอัตโนมัติ"""
        print(f"\n🤖 Agent '{self.name}' starting...")
        print("=" * 60)

        iteration = 1
        while self.tasks:
            print(f"\n📋 Iteration {iteration}")
            print(f"Pending tasks: {len(self.tasks)}")
            print(f"Completed tasks: {len(self.completed_tasks)}")

            # Get next task to execute
            prioritized = self.prioritize_tasks()

            if not prioritized:
                print("⏸️  No tasks ready to execute (waiting for dependencies)")
                time.sleep(1)
                iteration += 1
                continue

            # Execute highest priority task
            next_task = prioritized[0]
            self.execute_task(next_task)

            iteration += 1

        print(f"\n🎉 All tasks completed!")
        print(f"Total tasks: {len(self.completed_tasks)}")
        print(f"\n📚 Knowledge Base:")
        for task_name, info in self.knowledge_base.items():
            print(f"  - {task_name}: {info['result']}")

# สร้าง Agent และ Tasks
agent = DataScienceAgent("DataBot-2025")

# เพิ่ม tasks
tasks = [
    Task(1, "Load Data", "Load dataset from CSV", TaskStatus.PENDING, 5, []),
    Task(2, "Clean Data", "Handle missing values and outliers", TaskStatus.PENDING, 4, [1]),
    Task(3, "EDA", "Perform exploratory data analysis", TaskStatus.PENDING, 4, [2]),
    Task(4, "Feature Engineering", "Create new features", TaskStatus.PENDING, 3, [3]),
    Task(5, "Train Model", "Train ML model", TaskStatus.PENDING, 5, [4]),
    Task(6, "Evaluate Model", "Evaluate model performance", TaskStatus.PENDING, 4, [5]),
    Task(7, "Deploy Model", "Deploy to production", TaskStatus.PENDING, 5, [6]),
]

for task in tasks:
    agent.add_task(task)

# รัน Agent
agent.run()
```

#### 🔄 Multi-Agent System

```python
from typing import List
import threading

class AgentCoordinator:
    """
    ประสานงาน multiple agents ให้ทำงานร่วมกัน
    """

    def __init__(self):
        self.agents: List[DataScienceAgent] = []
        self.shared_knowledge: Dict[str, Any] = {}

    def register_agent(self, agent: DataScienceAgent):
        """ลงทะเบียน agent"""
        self.agents.append(agent)
        print(f"🤖 Registered: {agent.name}")

    def distribute_tasks(self, tasks: List[Task]):
        """แบ่ง tasks ให้ agents"""
        for i, task in enumerate(tasks):
            agent = self.agents[i % len(self.agents)]
            agent.add_task(task)

    def run_parallel(self):
        """รัน agents แบบ parallel"""
        print("\n🚀 Running agents in parallel...")
        threads = []

        for agent in self.agents:
            thread = threading.Thread(target=agent.run)
            threads.append(thread)
            thread.start()

        # รอให้ทุก thread เสร็จ
        for thread in threads:
            thread.join()

        print("\n✅ All agents completed!")

        # รวม knowledge จากทุก agents
        for agent in self.agents:
            self.shared_knowledge.update(agent.knowledge_base)

        print(f"\n📊 Shared Knowledge Base ({len(self.shared_knowledge)} items):")
        for key in self.shared_knowledge:
            print(f"  - {key}")

# ตัวอย่างการใช้งาน Multi-Agent
coordinator = AgentCoordinator()

# สร้าง agents
agent1 = DataScienceAgent("DataBot-Alpha")
agent2 = DataScienceAgent("DataBot-Beta")
agent3 = DataScienceAgent("DataBot-Gamma")

coordinator.register_agent(agent1)
coordinator.register_agent(agent2)
coordinator.register_agent(agent3)

# สร้าง tasks
tasks = [
    Task(i, f"Task-{i}", f"Process dataset {i}", TaskStatus.PENDING, 3, [])
    for i in range(1, 10)
]

# แบ่ง tasks
coordinator.distribute_tasks(tasks)

# รัน parallel
coordinator.run_parallel()
```

---

### 14. Real-time Data Science

#### ⚡ Real-time Processing คืออะไร?

**ตอบง่ายๆ:** ประมวลผลและวิเคราะห์ข้อมูลทันที (milliseconds ถึง seconds) ไม่ต้องรอ batch processing!

**Use Cases:**
- Fraud detection (ธนาคาร)
- Stock trading (การเงิน)
- IoT monitoring (อุตสาหกรรม)
- Gaming analytics

#### 📊 Real-time Analytics ด้วย Dask

```python
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client

print("⚡ Real-time Data Processing with Dask")
print("=" * 60)

# สร้าง Dask client (parallel processing)
client = Client(n_workers=4, threads_per_worker=2)
print(f"\nDask Dashboard: {client.dashboard_link}")

# สร้างข้อมูลขนาดใหญ่
print("\n1️⃣ Creating large dataset...")
df = dd.from_pandas(
    pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=10_000_000, freq='1s'),
        'user_id': np.random.randint(1, 100000, 10_000_000),
        'transaction_amount': np.random.exponential(100, 10_000_000),
        'category': np.random.choice(['food', 'transport', 'shopping', 'entertainment'],
                                    10_000_000)
    }),
    npartitions=10
)

print(f"Dataset size: {len(df)} rows")

# Real-time aggregations
print("\n2️⃣ Real-time aggregations...")

# Group by และคำนวณ
result = df.groupby('category').agg({
    'transaction_amount': ['mean', 'sum', 'count']
}).compute()

print("\nCategory Statistics:")
print(result)

# Real-time filtering
print("\n3️⃣ Real-time filtering (high-value transactions)...")
high_value = df[df['transaction_amount'] > 500]
high_value_count = len(high_value)
print(f"High-value transactions (>500): {high_value_count.compute():,}")

# Rolling window analysis
print("\n4️⃣ Rolling window analysis...")
df['hour'] = df['timestamp'].dt.hour
hourly_avg = df.groupby('hour')['transaction_amount'].mean().compute()

print("\nAverage transaction by hour:")
print(hourly_avg)

# Performance comparison: Pandas vs Dask
print("\n5️⃣ Performance Comparison")

# Pandas (single-threaded)
import time
pdf = df.compute()  # แปลงเป็น pandas

start = time.time()
pandas_result = pdf.groupby('category')['transaction_amount'].mean()
pandas_time = time.time() - start

# Dask (parallel)
start = time.time()
dask_result = df.groupby('category')['transaction_amount'].mean().compute()
dask_time = time.time() - start

print(f"Pandas time: {pandas_time:.4f}s")
print(f"Dask time: {dask_time:.4f}s")
print(f"Speedup: {pandas_time/dask_time:.2f}x")

client.close()
```

#### 🌊 Stream Processing Simulation

```python
import time
import random
from datetime import datetime
from collections import deque
from typing import Dict, List

class RealTimeMonitor:
    """
    Real-time monitoring system สำหรับ streaming data
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.alerts: List[Dict] = []
        self.metrics: Dict[str, float] = {}

    def process_datapoint(self, value: float) -> Dict:
        """ประมวลผล data point ใหม่แบบ real-time"""
        self.data_buffer.append({
            'timestamp': datetime.now(),
            'value': value
        })

        # คำนวณ metrics แบบ real-time
        values = [d['value'] for d in self.data_buffer]
        self.metrics = {
            'current': value,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }

        # Anomaly detection (3-sigma rule)
        if len(values) >= 30:  # ต้องมีข้อมูลพอ
            z_score = (value - self.metrics['mean']) / (self.metrics['std'] + 1e-10)
            if abs(z_score) > 3:
                alert = {
                    'timestamp': datetime.now(),
                    'value': value,
                    'z_score': z_score,
                    'severity': 'HIGH' if abs(z_score) > 4 else 'MEDIUM'
                }
                self.alerts.append(alert)
                return alert

        return None

    def get_current_metrics(self) -> Dict:
        """ดึง metrics ปัจจุบัน"""
        return self.metrics.copy()

    def get_recent_alerts(self, n: int = 5) -> List[Dict]:
        """ดึง alerts ล่าสุด"""
        return self.alerts[-n:]

# Simulation: Real-time monitoring
print("🔴 Real-time Monitoring System")
print("=" * 60)

monitor = RealTimeMonitor(window_size=50)

print("\n📊 Streaming data (Ctrl+C to stop)...")
print("\nProcessing", end="", flush=True)

try:
    for i in range(200):
        # สร้างข้อมูล (บางทีมี anomaly)
        if random.random() < 0.05:  # 5% chance of anomaly
            value = random.uniform(80, 120)  # Anomaly
        else:
            value = random.gauss(50, 10)  # Normal

        # ประมวลผลแบบ real-time
        alert = monitor.process_datapoint(value)

        # แสดง alert ถ้ามี
        if alert:
            print(f"\n🚨 ALERT at {alert['timestamp'].strftime('%H:%M:%S')}")
            print(f"   Value: {alert['value']:.2f}")
            print(f"   Z-score: {alert['z_score']:.2f}")
            print(f"   Severity: {alert['severity']}")
            print("\nProcessing", end="", flush=True)

        # แสดง progress
        if (i + 1) % 10 == 0:
            metrics = monitor.get_current_metrics()
            print(f"\n\n📈 Current Metrics (n={metrics['count']}):")
            print(f"   Current: {metrics['current']:.2f}")
            print(f"   Mean: {metrics['mean']:.2f}")
            print(f"   Std: {metrics['std']:.2f}")
            print(f"   Range: [{metrics['min']:.2f}, {metrics['max']:.2f}]")
            print(f"   Total Alerts: {len(monitor.alerts)}")
            print("\nProcessing", end="", flush=True)

        time.sleep(0.1)  # Simulate streaming delay

except KeyboardInterrupt:
    print("\n\n⏹️  Stopped by user")

print(f"\n\n📋 Final Summary:")
print(f"   Total data points: {monitor.metrics['count']}")
print(f"   Total alerts: {len(monitor.alerts)}")
print(f"\n🚨 Recent Alerts:")
for alert in monitor.get_recent_alerts():
    print(f"   - {alert['timestamp'].strftime('%H:%M:%S')}: "
          f"Value={alert['value']:.2f}, Z-score={alert['z_score']:.2f}")
```

---

### 15. Edge AI และ IoT

#### 📱 Edge AI คืออะไร?

**คำอธิบาย:** AI ที่รันบน device เอง (phone, IoT sensors) ไม่ต้องส่งข้อมูลไป cloud!

**ข้อดี:**
- ⚡ Latency ต่ำมาก (real-time)
- 🔒 Privacy ดีกว่า (data ไม่ออกจาก device)
- 💰 ลดค่าใช้จ่าย bandwidth
- 🌱 Energy efficient

#### 🔧 Model Optimization สำหรับ Edge Devices

```python
import torch
import torch.nn as nn
import torch.quantization

# สร้าง model ตัวอย่าง
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

print("📱 Edge AI Model Optimization")
print("=" * 60)

# สร้าง model
model = SimpleNet()
model.eval()

# 1. Model Pruning (ลดขนาด model)
print("\n1️⃣ Model Pruning")

import torch.nn.utils.prune as prune

# Prune 30% ของ weights
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, 'weight')  # ทำให้ permanent

print("✅ Pruned 30% of weights")

# 2. Quantization (ลด precision จาก float32 → int8)
print("\n2️⃣ Quantization")

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

print("✅ Quantized to int8")

# เปรียบเทียบขนาด
def get_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    size = os.path.getsize("temp_model.pth") / 1024  # KB
    os.remove("temp_model.pth")
    return size

import os
original_size = get_model_size(model)
quantized_size = get_model_size(quantized_model)

print(f"\n📊 Size Comparison:")
print(f"   Original: {original_size:.2f} KB")
print(f"   Quantized: {quantized_size:.2f} KB")
print(f"   Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

# 3. TorchScript (export สำหรับ production)
print("\n3️⃣ TorchScript Export")

# Trace model
example_input = torch.randn(1, 100)
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save("edge_model.pt")
print("✅ Exported to edge_model.pt")

# 4. ONNX Export (cross-platform)
print("\n4️⃣ ONNX Export")

torch.onnx.export(
    model,
    example_input,
    "edge_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✅ Exported to edge_model.onnx")
```

#### 🌡️ IoT Sensor Data Processing

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class IoTSensorSimulator:
    """
    จำลอง IoT sensors และ edge processing
    """

    def __init__(self, sensor_id: str, normal_range: tuple):
        self.sensor_id = sensor_id
        self.normal_min, self.normal_max = normal_range
        self.data_buffer = []

    def read_sensor(self) -> dict:
        """อ่านค่าจาก sensor"""
        # จำลองการอ่านค่า
        base_value = (self.normal_min + self.normal_max) / 2
        noise = np.random.normal(0, (self.normal_max - self.normal_min) * 0.1)
        value = base_value + noise

        # บางครั้งมี anomaly
        if np.random.random() < 0.05:
            value += np.random.choice([-1, 1]) * (self.normal_max - self.normal_min) * 0.5

        return {
            'sensor_id': self.sensor_id,
            'timestamp': datetime.now(),
            'value': value,
            'is_anomaly': value < self.normal_min or value > self.normal_max
        }

    def edge_processing(self, reading: dict) -> dict:
        """ประมวลผลบน edge device (ไม่ส่งไป cloud)"""
        self.data_buffer.append(reading)

        # เก็บแค่ 50 readings ล่าสุด
        if len(self.data_buffer) > 50:
            self.data_buffer.pop(0)

        # คำนวณ statistics แบบ local
        recent_values = [r['value'] for r in self.data_buffer]
        stats = {
            'mean': np.mean(recent_values),
            'std': np.std(recent_values),
            'min': np.min(recent_values),
            'max': np.max(recent_values)
        }

        # ตัดสินใจว่าต้องส่ง alert ไหม
        should_alert = reading['is_anomaly']

        return {
            **reading,
            'local_stats': stats,
            'should_alert': should_alert,
            'processed_on_edge': True
        }

# สร้าง IoT sensors
print("🌡️  IoT Edge AI System")
print("=" * 60)

sensors = [
    IoTSensorSimulator("TEMP-001", (18, 26)),  # Temperature
    IoTSensorSimulator("HUMID-001", (40, 70)),  # Humidity
    IoTSensorSimulator("PRESSURE-001", (980, 1020))  # Pressure
]

print(f"\n✅ Initialized {len(sensors)} sensors")
print("\n📊 Collecting data...")

# Collect data
all_readings = []
alerts = []

for _ in range(100):
    for sensor in sensors:
        # Read sensor
        reading = sensor.read_sensor()

        # Process on edge
        processed = sensor.edge_processing(reading)
        all_readings.append(processed)

        # Check for alerts
        if processed['should_alert']:
            alerts.append(processed)

    time.sleep(0.05)  # Simulate interval

print(f"\n✅ Collected {len(all_readings)} readings")
print(f"🚨 Detected {len(alerts)} anomalies")

# Analyze results
df = pd.DataFrame(all_readings)

print(f"\n📈 Statistics by Sensor:")
for sensor_id in df['sensor_id'].unique():
    sensor_data = df[df['sensor_id'] == sensor_id]
    print(f"\n{sensor_id}:")
    print(f"  Mean: {sensor_data['value'].mean():.2f}")
    print(f"  Std: {sensor_data['value'].std():.2f}")
    print(f"  Anomalies: {sensor_data['is_anomaly'].sum()}")

# Visualization
fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 8))

for idx, sensor in enumerate(sensors):
    sensor_data = df[df['sensor_id'] == sensor.sensor_id]

    axes[idx].plot(range(len(sensor_data)), sensor_data['value'],
                  label='Sensor Reading', linewidth=2)

    # Mark anomalies
    anomalies = sensor_data[sensor_data['is_anomaly']]
    axes[idx].scatter(anomalies.index, anomalies['value'],
                     color='red', s=100, label='Anomaly', zorder=5)

    # Normal range
    axes[idx].axhline(sensor.normal_min, color='green',
                     linestyle='--', alpha=0.5, label='Normal Range')
    axes[idx].axhline(sensor.normal_max, color='green',
                     linestyle='--', alpha=0.5)

    axes[idx].set_title(f'{sensor.sensor_id} Readings')
    axes[idx].set_ylabel('Value')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (samples)')
plt.tight_layout()
plt.show()

print("\n💡 Edge Processing Benefits:")
print(f"  - All processing done locally (no cloud needed)")
print(f"  - Real-time anomaly detection")
print(f"  - Reduced bandwidth (only alerts sent to cloud)")
print(f"  - Privacy preserved (raw data stays on device)")
```

---

### 16. Green AI และ Ethical AI

#### 🌱 Green AI คืออะไร?

**คำอธิบาย:** AI ที่คำนึงถึงสิ่งแวดล้อม ลดการใช้พลังงาน และ carbon footprint!

**เป้าหมายในปี 2025:**
- ลดพลังงานในการ train model 50%
- ใช้ efficient architectures
- Reuse pretrained models
- Optimize inference

#### ⚡ Energy-Efficient Training

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time

print("🌱 Green AI: Energy-Efficient Training")
print("=" * 60)

# สร้าง model และ data
model = nn.Sequential(
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

X = torch.randn(10000, 1000)
y = torch.randint(0, 10, (10000,))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X, y = X.to(device), y.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 1. Traditional Training (FP32)
print("\n1️⃣ Traditional Training (FP32)")
start_time = time.time()
start_energy = time.time()  # ใน production ใช้ power meter จริง

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

fp32_time = time.time() - start_time
print(f"Time: {fp32_time:.2f}s")

# 2. Mixed Precision Training (FP16 + FP32)
print("\n2️⃣ Mixed Precision Training (FP16)")
model = model.to(device)  # Reset model
scaler = GradScaler()

start_time = time.time()

model.train()
for epoch in range(10):
    optimizer.zero_grad()

    # Mixed precision context
    with autocast():
        outputs = model(X)
        loss = criterion(outputs, y)

    # Scaled backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

fp16_time = time.time() - start_time
print(f"Time: {fp16_time:.2f}s")

print(f"\n⚡ Speedup: {fp32_time/fp16_time:.2f}x")
print(f"💰 Energy Saved: ~{(1 - fp16_time/fp32_time)*100:.1f}%")

# 3. Carbon Footprint Estimation
def estimate_carbon_footprint(training_time_hours, power_watts=300):
    """
    ประมาณ CO2 emissions จากการ train model

    Args:
        training_time_hours: เวลาในการ train (ชั่วโมง)
        power_watts: พลังงานที่ใช้ (watts)

    Returns:
        CO2 emissions (kg)
    """
    # สมมติ: 0.5 kg CO2 per kWh (average grid)
    energy_kwh = (power_watts * training_time_hours) / 1000
    co2_kg = energy_kwh * 0.5
    return co2_kg

print("\n🌍 Carbon Footprint Estimation")
print("=" * 40)

# ตัวอย่าง: train โมเดลใหญ่
training_scenarios = [
    ("Small Model (1 GPU, 1 hour)", 1, 300),
    ("Medium Model (4 GPUs, 24 hours)", 24, 1200),
    ("Large Model (64 GPUs, 1 week)", 168, 19200),
]

for name, hours, watts in training_scenarios:
    co2 = estimate_carbon_footprint(hours, watts)
    print(f"\n{name}:")
    print(f"  Energy: {(watts * hours) / 1000:.1f} kWh")
    print(f"  CO2: {co2:.2f} kg")
    print(f"  Equivalent to driving: {co2 / 0.2:.0f} km")
```

#### 🎯 Ethical AI Practices

```python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("⚖️  Ethical AI: Fairness Analysis")
print("=" * 60)

# สร้างข้อมูลจำลอง: Loan Approval
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'income': np.random.exponential(50000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'ethnicity': np.random.choice(['Group_A', 'Group_B', 'Group_C'], n_samples)
})

# สร้าง labels (approved/rejected) - มี bias!
def approval_with_bias(row):
    score = (row['income'] / 1000 +
            row['credit_score'] +
            row['age'] * 2)

    # Bias: Group_A ได้เปรียบ
    if row['ethnicity'] == 'Group_A':
        score *= 1.2
    elif row['ethnicity'] == 'Group_C':
        score *= 0.9

    return 1 if score > 800 else 0

data['approved'] = data.apply(approval_with_bias, axis=1)

print(f"\n📊 Dataset: {len(data)} loan applications")
print(f"Overall approval rate: {data['approved'].mean()*100:.1f}%")

# 1. Fairness Analysis by Group
print("\n1️⃣ Approval Rate by Ethnicity")
print("=" * 40)

for group in data['ethnicity'].unique():
    group_data = data[data['ethnicity'] == group]
    approval_rate = group_data['approved'].mean()
    print(f"{group}: {approval_rate*100:.1f}% "
          f"({group_data['approved'].sum()}/{len(group_data)})")

# Statistical Parity Difference
approval_rates = data.groupby('ethnicity')['approved'].mean()
spd = approval_rates.max() - approval_rates.min()
print(f"\nStatistical Parity Difference: {spd:.3f}")
print(f"⚠️  Threshold: >0.1 indicates potential bias")

# 2. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart: Approval rate by ethnicity
approval_by_group = data.groupby('ethnicity')['approved'].mean() * 100
axes[0].bar(approval_by_group.index, approval_by_group.values,
           color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0].set_title('Approval Rate by Ethnicity', fontweight='bold')
axes[0].set_ylabel('Approval Rate (%)')
axes[0].axhline(data['approved'].mean() * 100, color='red',
               linestyle='--', label='Overall Average')
axes[0].legend()

# Income distribution by ethnicity
for group in data['ethnicity'].unique():
    group_data = data[data['ethnicity'] == group]
    axes[1].hist(group_data['income'], alpha=0.5, bins=30, label=group)

axes[1].set_title('Income Distribution by Ethnicity', fontweight='bold')
axes[1].set_xlabel('Income')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.show()

# 3. Bias Mitigation
print("\n2️⃣ Bias Mitigation")
print("=" * 40)

def approval_fair(row):
    """Approval function without bias"""
    score = (row['income'] / 1000 +
            row['credit_score'] +
            row['age'] * 2)
    return 1 if score > 800 else 0

data['approved_fair'] = data.apply(approval_fair, axis=1)

print("\nAfter Bias Mitigation:")
for group in data['ethnicity'].unique():
    group_data = data[data['ethnicity'] == group]
    approval_rate = group_data['approved_fair'].mean()
    print(f"{group}: {approval_rate*100:.1f}%")

approval_rates_fair = data.groupby('ethnicity')['approved_fair'].mean()
spd_fair = approval_rates_fair.max() - approval_rates_fair.min()
print(f"\nStatistical Parity Difference: {spd_fair:.3f}")
print(f"✅ Improved from {spd:.3f} to {spd_fair:.3f}")

# 4. Explainability
print("\n3️⃣ Model Explainability")
print("=" * 40)

# Feature importance (simplified)
features = ['income', 'credit_score', 'age']
correlations = [data[feat].corr(data['approved_fair']) for feat in features]

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': [abs(c) for c in correlations]
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

print("\n💡 Ethical AI Checklist:")
print("  ✅ Measured fairness across groups")
print("  ✅ Identified and mitigated bias")
print("  ✅ Model decisions are explainable")
print("  ✅ Regular auditing process")
print("  ✅ Transparent about limitations")
```

---

### 17. Deployment และ Production AI

#### 🚀 Deploying ML Models

**Production Checklist:**
- ✅ Model versioning
- ✅ API endpoints
- ✅ Monitoring และ logging
- ✅ A/B testing
- ✅ Scalability

#### 🔧 FastAPI: สร้าง ML API

```python
# file: ml_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List
import uvicorn

# สร้าง FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production-ready ML API with FastAPI",
    version="1.0.0"
)

# Define request/response models
class PredictionRequest(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.5, 3.2, 4.1, 5.0]
            }
        }

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    confidence: float

# Load model (ในการใช้งานจริง)
# model = joblib.load('model.pkl')

# Mock model สำหรับ demo
class MockModel:
    def predict(self, X):
        return np.sum(X, axis=1)

    def predict_proba(self, X):
        pred = self.predict(X)
        return np.column_stack([1 - pred, pred])

model = MockModel()
MODEL_VERSION = "1.0.0"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction with ML model

    - **features**: List of numerical features
    """
    try:
        # Validate input
        if len(request.features) != 5:
            raise HTTPException(
                status_code=400,
                detail="Expected 5 features"
            )

        # Prepare input
        X = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))

        return PredictionResponse(
            prediction=float(prediction),
            model_version=MODEL_VERSION,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(features: List[List[float]]):
    """
    Batch prediction for multiple inputs
    """
    try:
        X = np.array(features)
        predictions = model.predict(X)

        return {
            "predictions": predictions.tolist(),
            "count": len(predictions),
            "model_version": MODEL_VERSION
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoint
@app.get("/metrics")
async def metrics():
    """
    Get API metrics
    """
    return {
        "total_requests": 1000,  # ในการใช้งานจริง track ด้วย prometheus
        "avg_response_time_ms": 50,
        "error_rate": 0.01,
        "model_version": MODEL_VERSION
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**วิธีรัน API:**
```bash
# ติดตั้ง dependencies
pip install fastapi uvicorn

# รัน server
python ml_api.py

# ทดสอบ API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.5, 3.2, 4.1, 5.0]}'
```

#### 🐳 Docker Deployment

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "ml_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
fastapi==0.109.0
uvicorn==0.27.0
numpy==1.26.3
scikit-learn==1.4.0
joblib==1.3.2
pydantic==2.5.3
```

**Build และ Run Docker:**
```bash
# Build image
docker build -t ml-api:1.0 .

# Run container
docker run -d -p 8000:8000 ml-api:1.0

# Check logs
docker logs <container_id>
```

#### 📊 Model Monitoring

```python
import time
import logging
from datetime import datetime
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Monitor ML model performance in production
    """

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.start_time = time.time()

    def log_prediction(self, input_data, prediction, latency_ms, error=None):
        """Log prediction for monitoring"""

        self.predictions.append({
            'timestamp': datetime.now(),
            'input': input_data,
            'prediction': prediction,
            'latency_ms': latency_ms,
            'error': error
        })

        self.latencies.append(latency_ms)

        if error:
            self.errors.append(error)
            logger.error(f"Prediction error: {error}")

    def get_metrics(self):
        """Get current metrics"""

        if not self.predictions:
            return {}

        predictions_array = np.array([p['prediction'] for p in self.predictions])

        return {
            'total_predictions': len(self.predictions),
            'error_count': len(self.errors),
            'error_rate': len(self.errors) / len(self.predictions),
            'avg_latency_ms': np.mean(self.latencies),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'p99_latency_ms': np.percentile(self.latencies, 99),
            'prediction_mean': np.mean(predictions_array),
            'prediction_std': np.std(predictions_array),
            'uptime_hours': (time.time() - self.start_time) / 3600
        }

    def detect_drift(self, reference_mean, reference_std, threshold=2.0):
        """
        Detect data drift

        Returns True if drift detected
        """

        if len(self.predictions) < 100:
            return False

        predictions_array = np.array([p['prediction'] for p in self.predictions])
        current_mean = np.mean(predictions_array)

        # Z-score
        z_score = abs(current_mean - reference_mean) / reference_std

        if z_score > threshold:
            logger.warning(f"Data drift detected! Z-score: {z_score:.2f}")
            return True

        return False

    def check_health(self):
        """Check system health"""

        metrics = self.get_metrics()

        issues = []

        # Check error rate
        if metrics.get('error_rate', 0) > 0.05:
            issues.append(f"High error rate: {metrics['error_rate']*100:.1f}%")

        # Check latency
        if metrics.get('avg_latency_ms', 0) > 100:
            issues.append(f"High latency: {metrics['avg_latency_ms']:.1f}ms")

        if issues:
            logger.warning(f"Health check failed: {', '.join(issues)}")
            return False, issues

        return True, []

# ตัวอย่างการใช้งาน
monitor = ModelMonitor()

print("📊 Model Monitoring System")
print("=" * 60)

# จำลองการใช้งาน
reference_mean = 5.0
reference_std = 1.0

for i in range(200):
    # จำลอง prediction
    input_data = np.random.randn(5)
    prediction = np.random.randn() * reference_std + reference_mean

    # จำลอง latency
    latency = np.random.exponential(50)  # Average 50ms

    # จำลอง error (5% chance)
    error = "Timeout" if np.random.random() < 0.05 else None

    # Log
    monitor.log_prediction(input_data, prediction, latency, error)

    # Check every 50 predictions
    if (i + 1) % 50 == 0:
        print(f"\n📈 Metrics at {i+1} predictions:")
        metrics = monitor.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Health check
        healthy, issues = monitor.check_health()
        if healthy:
            print("  ✅ System healthy")
        else:
            print(f"  ⚠️  Issues: {', '.join(issues)}")

        # Drift detection
        if monitor.detect_drift(reference_mean, reference_std):
            print("  🚨 Data drift detected!")

print("\n✅ Monitoring completed!")
```

---

## 🎓 สรุปส่วนที่ 3: ระดับสูง

### สิ่งที่เราได้เรียนรู้:

✅ **Deep Learning**
- Neural Networks พื้นฐาน
- PyTorch framework
- CNN สำหรับภาพ

✅ **Generative AI & LLMs**
- Hugging Face Transformers
- Text generation, sentiment analysis, NER
- LangChain chatbots

✅ **Agentic AI**
- Autonomous agents
- Multi-agent systems
- Task automation

✅ **Real-time Data Science**
- Dask parallel processing
- Stream processing
- Real-time monitoring

✅ **Edge AI & IoT**
- Model optimization (pruning, quantization)
- Edge deployment
- IoT sensor processing

✅ **Green AI & Ethical AI**
- Energy-efficient training
- Carbon footprint estimation
- Fairness analysis
- Bias mitigation

✅ **Production Deployment**
- FastAPI ML APIs
- Docker containers
- Model monitoring
- Health checks

---

### 🎯 แบบฝึกหัดท้ายบท:

**Challenge 1: Build Production ML System**
สร้างระบบ ML แบบครบวงจร:
- Train model
- Create FastAPI endpoint
- Dockerize application
- Add monitoring

**Challenge 2: Green AI Project**
เปรียบเทียบ energy consumption:
- Train model แบบ standard vs optimized
- Measure training time และ energy
- Implement efficient inference

**Challenge 3: Edge AI Application**
สร้าง edge AI app:
- Optimize model สำหรับ mobile
- Implement on-device inference
- Real-time processing

**Challenge 4: Ethical AI Audit**
วิเคราะห์ fairness:
- เลือก dataset
- Measure bias metrics
- Implement mitigation strategies
- Create audit report

---

### 🚀 ต่อไป: Workshops!

ในส่วนถัดไป เราจะมี hands-on workshops:
- Sentiment Analysis Project
- Recommendation System
- Real-time Stock Prediction
- AI Chatbot
- Edge AI for IoT

Let's build something amazing! 💪🔥
