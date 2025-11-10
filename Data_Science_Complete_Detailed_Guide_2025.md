# 📖 Data Science: คู่มือครบครันที่สุด (ฉบับอธิบายละเอียด)
**อธิบายทุกอย่างแบบละเอียดยิบ พร้อมมุกสนุกๆ! 🎉**

---

## 📑 สารบัญฉบับเต็ม

### Part 1: พื้นฐานที่ต้องรู้ทั้งหมด
1. [Data Science คืออะไร? (ฉบับละเอียดสุดๆ)](#1-data-science-คืออะไร)
2. [Python ตั้งแต่ศูนย์ถึงขั้นเทพ](#2-python-ตั้งแต่ศูนย์ถึงขั้นเทพ)
3. [NumPy: เจาะลึกทุกฟังก์ชัน](#3-numpy-เจาะลึก)
4. [Pandas: มาสเตอร์การจัดการข้อมูล](#4-pandas-มาสเตอร์)
5. [Visualization: สร้างกราฟสวยทุกแบบ](#5-visualization-สุดยอด)

### Part 2: Machine Learning แบบครบวงจร
6. [ML Algorithms: ทุกอัลกอริทึมที่ต้องรู้](#6-ml-algorithms)
7. [Deep Learning: เจาะลึกทุกสถาปัตยกรรม](#7-deep-learning)
8. [NLP: ประมวลผลภาษาแบบสมบูรณ์](#8-nlp-complete)

### Part 3: Advanced Topics
9. [Transformer Architecture: ละเอียดทุกชั้น](#9-transformer-deep)
10. [Generative AI: สร้างสรรค์ด้วย AI](#10-generative-ai)

---

## 1. Data Science คืออะไร? (ฉบับละเอียดสุดๆ)

### 🎯 นิยามที่ครบถ้วน

**Data Science** คือศาสตร์ (Science) และศิลป์ (Art) ของการ:
1. **เก็บรวบรวมข้อมูล** (Data Collection)
2. **จัดเก็บข้อมูล** (Data Storage)
3. **ทำความสะอาดข้อมูล** (Data Cleaning)
4. **วิเคราะห์ข้อมูล** (Data Analysis)
5. **สร้างโมเดล** (Modeling)
6. **ทำนาย/ตัดสินใจ** (Prediction/Decision Making)
7. **สื่อสารผลลัพธ์** (Communication)

**มุก:** Data Science เหมือนการทำอาหาร 🍳
- **Data Collection** = ไปซื้อของในตลาด 🛒
- **Data Storage** = เก็บของในตู้เย็น ❄️
- **Data Cleaning** = ล้างผักล้างเนื้อ 🚿
- **Data Analysis** = ชิมรสดูว่าอร่อยยัง 👅
- **Modeling** = ใช้สูตรอาหารที่ดี 📖
- **Prediction** = คาดการณ์ว่าแขกจะชอบไหม 🤔
- **Communication** = เสิร์ฟอาหารสวยๆ 🎨

### 🌍 Data Science ในโลกจริง

#### ตัวอย่างที่ 1: Netflix รู้ว่าคุณชอบหนังอะไร 🎬

```python
"""
Netflix Recommendation System

วิธีทำงาน:
1. เก็บข้อมูลว่าคุณดูหนังอะไร
2. เก็บว่าคุณดูจบหรือปิดทิ้ง
3. เก็บว่าคุณกด Like หรือ Dislike
4. วิเคราะห์รูปแบบ (Patterns)
5. หาคนที่ดูแบบเดียวกับคุณ
6. แนะนำหนังที่คนกลุ่มนั้นชอบ

ผลลัพธ์: คุณติดหนังไม่ยอมนอน! 😅
"""

# จำลองระบบแนะนำ (simplified)
class NetflixRecommender:
    def __init__(self):
        self.user_preferences = {}
        self.movie_features = {}

    def collect_data(self, user_id, movie_id, watched, liked):
        """เก็บข้อมูลการดู"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = []

        self.user_preferences[user_id].append({
            'movie': movie_id,
            'watched': watched,  # ดูจบหรือไม่
            'liked': liked,      # ชอบหรือไม่
            'timestamp': datetime.now()
        })

        print(f"✅ เก็บข้อมูล: User {user_id} {'ชอบ' if liked else 'ไม่ชอบ'} {movie_id}")

    def analyze_patterns(self, user_id):
        """วิเคราะห์รูปแบบการดู"""
        user_data = self.user_preferences.get(user_id, [])

        # นับประเภทหนังที่ชอบ
        favorite_genres = {}
        for data in user_data:
            if data['liked']:
                # สมมติว่าเราดึง genre จาก movie_id
                genre = self.get_movie_genre(data['movie'])
                favorite_genres[genre] = favorite_genres.get(genre, 0) + 1

        return favorite_genres

    def recommend(self, user_id, n=5):
        """แนะนำหนัง!"""
        # วิเคราะห์รสนิยม
        preferences = self.analyze_patterns(user_id)

        # หาหนังที่เหมาะสม
        recommendations = []
        # ... logic แนะนำ ...

        return recommendations

# มุก: Netflix รู้จักคุณดีกว่าแฟนคุณ! 😂
```

#### ตัวอย่างที่ 2: Grab/Bolt คำนวณราคา 🚗

```python
"""
Dynamic Pricing Algorithm

ปัจจัยที่พิจารณา:
1. ระยะทาง (Distance)
2. เวลา (Time of Day) - ชั่วโมงเร่งด่วน แพงขึ้น!
3. สภาพอากาศ (Weather) - ฝนตก แพงขึ้น!
4. Demand/Supply (อุปสงค์/อุปทาน)
5. Location (พื้นที่พิเศษ เช่น สนามบิน)
6. ประวัติลูกค้า (Customer History)

มุก: ยิ่งรีบ ยิ่งฝนตก ยิ่งแพง - เหมือนชีวิต! 💸
"""

import math

class DynamicPricing:
    def __init__(self):
        self.base_rate = 7  # บาทต่อ km
        self.base_fare = 25  # ค่าธรรมเนียมพื้นฐาน

    def calculate_fare(
        self,
        distance_km: float,
        is_rush_hour: bool = False,
        is_raining: bool = False,
        demand_multiplier: float = 1.0,
        is_airport: bool = False
    ):
        """
        คำนวณค่าโดยสาร

        มุก: สูตรนี้ซับซ้อนกว่าสมการเคมี!
        แต่ทำให้ Grab รวยกว่าเคมี! 💰
        """

        # ค่าระยะทาง
        distance_fare = distance_km * self.base_rate

        # Surge pricing (ราคาพิเศษ)
        surge = 1.0

        # ชั่วโมงเร่งด่วน (+20-50%)
        if is_rush_hour:
            surge *= 1.3
            print("  ⏰ Rush hour: +30%")

        # ฝนตก (+10-30%)
        if is_raining:
            surge *= 1.2
            print("  🌧️ Raining: +20%")

        # อุปสงค์สูง
        if demand_multiplier > 1.0:
            surge *= demand_multiplier
            print(f"  🔥 High demand: x{demand_multiplier}")

        # พื้นที่พิเศษ (สนามบิน)
        airport_fee = 50 if is_airport else 0

        # คำนวณรวม
        total = (self.base_fare + distance_fare) * surge + airport_fee

        # ปัดเศษ
        total = math.ceil(total)

        return {
            'base_fare': self.base_fare,
            'distance_fare': distance_fare,
            'surge_multiplier': surge,
            'airport_fee': airport_fee,
            'total': total
        }

    def explain_fare(self, fare_breakdown):
        """อธิบายค่าโดยสาร"""
        print(f"\n💰 Fare Breakdown:")
        print(f"  Base fare: {fare_breakdown['base_fare']} ฿")
        print(f"  Distance: {fare_breakdown['distance_fare']:.2f} ฿")
        print(f"  Surge: x{fare_breakdown['surge_multiplier']:.2f}")
        if fare_breakdown['airport_fee'] > 0:
            print(f"  Airport fee: {fare_breakdown['airport_fee']} ฿")
        print(f"  ─────────────")
        print(f"  Total: {fare_breakdown['total']} ฿")

# ทดสอบ
pricer = DynamicPricing()

print("📍 Scenario 1: เช้าวันธรรมดา ไปทำงาน")
fare1 = pricer.calculate_fare(
    distance_km=5.0,
    is_rush_hour=True,
    is_raining=False,
    demand_multiplier=1.5
)
pricer.explain_fare(fare1)

print("\n📍 Scenario 2: กลางคืน ฝนตก ไปสนามบิน")
fare2 = pricer.calculate_fare(
    distance_km=30.0,
    is_rush_hour=False,
    is_raining=True,
    demand_multiplier=1.0,
    is_airport=True
)
pricer.explain_fare(fare2)

print("\n💡 Insights:")
print("  - ชั่วโมงเร่งด่วน = ราคาพิเศษ")
print("  - ฝนตก = คนเรียกเยอะ = แพงขึ้น")
print("  - สนามบิน = มีค่าธรรมเนียมพิเศษ")
print("\n  มุก: อยากประหยัด? เดินเอา! 🚶‍♂️😂")
```

---

## 2. Python ตั้งแต่ศูนย์ถึงขั้นเทพ

### 📚 ทำไมต้องเรียน Python?

**5 เหตุผลสำคัญ:**

1. **ง่ายที่สุด** 🎯
   - Syntax อ่านง่ายเหมือนภาษาอังกฤษ
   - เริ่มต้นได้ภายใน 1 วัน
   - มุก: "Python = ภาษาคอมที่อ่านแล้วเข้าใจ ไม่ใช่ดูแล้วงง!"

2. **ใช้งานได้หลากหลาย** 🌈
   - Web Development (Django, Flask)
   - Data Science (Pandas, NumPy)
   - AI/ML (TensorFlow, PyTorch)
   - Automation (Selenium, Scripts)
   - มุก: "Swiss Army Knife ของโลกโปรแกรมมิ่ง!"

3. **Community ใหญ่มาก** 👥
   - มีคนใช้ทั่วโลก
   - หา library/package ได้ง่าย
   - ติดปัญหามีคนช่วย
   - มุก: "ถามปัญหาใน Stack Overflow จะได้คำตอบภายใน 5 นาที!"

4. **งานเยอะ เงินดี** 💰
   - Data Scientist เงินเดือนสูง
   - Python Developer ต้องการมาก
   - มุก: "เรียน Python วันนี้ รวยพรุ่งนี้! (อาจต้องรอสักหน่อย 😅)"

5. **AI/ML ใช้ Python เกือบทั้งหมด** 🤖
   - TensorFlow, PyTorch, scikit-learn
   - ทุกบริษัทใหญ่ๆ ใช้
   - มุก: "ไม่รู้ Python = พลาด AI Revolution!"

### 🔢 ตัวแปรและ Data Types (ละเอียดสุดๆ)

#### 1. Numbers (ตัวเลข)

```python
"""
ใน Python มี 3 ประเภทหลัก:
1. int (Integer) - จำนวนเต็ม
2. float (Floating Point) - ทศนิยม
3. complex - จำนวนเชิงซ้อน (ใช้น้อย)
"""

# ========== int (จำนวนเต็ม) ==========

# ปกติ
age = 25
year = 2025

# ใหญ่มากๆ (Python รองรับได้!)
big_number = 999999999999999999999999999999
print(f"เลขใหญ่: {big_number}")  # ไม่ overflow!

# มุก: Python จัดการเลขใหญ่ได้ ไม่เหมือน C/C++ ที่ overflow!
# เหมือน "กระเป๋าเงินที่ใหญ่ไม่จำกัด" 💰

# Binary, Octal, Hexadecimal
binary = 0b1010  # เลขฐาน 2 = 10
octal = 0o12     # เลขฐาน 8 = 10
hex_num = 0xA    # เลขฐาน 16 = 10

print(f"Binary {binary}, Octal {octal}, Hex {hex_num}")
# ทั้งหมดเท่ากับ 10!

# ========== float (ทศนิยม) ==========

# ปกติ
height = 1.75
weight = 65.5
pi = 3.14159

# Scientific notation
speed_of_light = 3e8  # 3 × 10^8 m/s
electron_mass = 9.1e-31  # 9.1 × 10^-31 kg

print(f"ความเร็วแสง: {speed_of_light} m/s")
print(f"มวลอิเล็กตรอน: {electron_mass} kg")

# มุก: e คือ "exponent" ไม่ใช่ "อี!" 😂

# ปัญหา floating point precision
result = 0.1 + 0.2
print(f"0.1 + 0.2 = {result}")  # 0.30000000000000004 😱
print(f"เท่ากับ 0.3 หรือไม่? {result == 0.3}")  # False!

# วิธีแก้: ใช้ round() หรือ Decimal
from decimal import Decimal
result_decimal = Decimal('0.1') + Decimal('0.2')
print(f"ใช้ Decimal: {result_decimal}")  # 0.3 ถูกต้อง!

# มุก: "คอมพิวเตอร์บวกเลขได้แต่บวกไม่ถูก!" 🤦‍♂️

# ========== การดำเนินการทางคณิตศาสตร์ ==========

a = 10
b = 3

print(f"\n🔢 Math Operations:")
print(f"  {a} + {b} = {a + b}")      # บวก = 13
print(f"  {a} - {b} = {a - b}")      # ลบ = 7
print(f"  {a} * {b} = {a * b}")      # คูณ = 30
print(f"  {a} / {b} = {a / b}")      # หาร = 3.333... (เป็น float)
print(f"  {a} // {b} = {a // b}")    # หารปัดเศษ = 3
print(f"  {a} % {b} = {a % b}")      # หารเอาเศษ (modulo) = 1
print(f"  {a} ** {b} = {a ** b}")    # ยกกำลัง = 1000

# มุก: "//" คือ "ไม่เอาเศษ!" และ "%" คือ "เอาแต่เศษ!" 😄

# Order of operations (PEMDAS)
result = 2 + 3 * 4  # = 2 + 12 = 14 (คูณก่อน)
result2 = (2 + 3) * 4  # = 5 * 4 = 20 (วงเล็บก่อน)

print(f"\n2 + 3 * 4 = {result}")
print(f"(2 + 3) * 4 = {result2}")

# มุก: "Remember PEMDAS - Please Excuse My Dear Aunt Sally!"
# (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)

# ========== ฟังก์ชันมาตรฐาน ==========

import math

print(f"\n📐 Math Functions:")
print(f"  abs(-5) = {abs(-5)}")  # ค่าสัมบูรณ์ = 5
print(f"  round(3.7) = {round(3.7)}")  # ปัดเศษ = 4
print(f"  round(3.14159, 2) = {round(3.14159, 2)}")  # ปัด 2 ตำแหน่ง = 3.14
print(f"  max(1, 5, 3) = {max(1, 5, 3)}")  # หาค่าสูงสุด = 5
print(f"  min(1, 5, 3) = {min(1, 5, 3)}")  # หาค่าต่ำสุด = 1
print(f"  pow(2, 3) = {pow(2, 3)}")  # ยกกำลัง = 8
print(f"  math.sqrt(16) = {math.sqrt(16)}")  # รากที่สอง = 4.0
print(f"  math.ceil(3.2) = {math.ceil(3.2)}")  # ปัดขึ้น = 4
print(f"  math.floor(3.9) = {math.floor(3.9)}")  # ปัดลง = 3

# มุก: ceil = เพดาน (ขึ้นบน), floor = พื้น (ลงล่าง) 🏠
```

#### 2. Strings (ข้อความ) - ละเอียดทุกมุม

```python
"""
String = ลำดับของตัวอักษร (sequence of characters)

ใน Python:
- ใช้ single quotes ('...') หรือ double quotes ("...") ก็ได้
- Triple quotes ('''...''' หรือ \"\"\"...\"\"\") สำหรับหลายบรรทัด
"""

# ========== การสร้าง String ==========

# แบบต่างๆ
name = 'Alice'
city = "Bangkok"
message = """
สวัสดี!
นี่คือข้อความ
หลายบรรทัด
"""

# มุก: "ใช้ ' หรือ \" ก็ได้ แต่อย่าปนกัน!" 😅

# Escape characters
quote = "He said, \"Hello!\""  # ใช้ \" ในเครื่องหมายคำพูด
path = "C:\\Users\\Alice\\Documents"  # ใช้ \\ สำหรับ backslash
newline = "Line 1\nLine 2"  # \n = ขึ้นบรรทัดใหม่
tab = "Name:\tAlice"  # \t = tab

print(quote)
print(path)
print(newline)
print(tab)

# Raw string (ไม่แปลง escape characters)
raw_path = r"C:\Users\Alice\Documents"  # ใช้ r"..." = raw string
print(raw_path)  # แสดง backslash ตามจริง

# มุก: "r'' คือ 'raw' - กินดิบ ไม่แปลง!" 🍣

# ========== String Indexing & Slicing ==========

text = "Data Science"
#      0123456789...  (index เริ่มที่ 0)
#      ...9876543210- (negative index นับจากหลัง)

print(f"\nString: '{text}'")
print(f"  ตัวแรก: text[0] = '{text[0]}'")  # 'D'
print(f"  ตัวที่ 5: text[5] = '{text[5]}'")  # 'S'
print(f"  ตัวสุดท้าย: text[-1] = '{text[-1]}'")  # 'e'
print(f"  ตัวสุดท้ายที่ 2: text[-2] = '{text[-2]}'")  # 'c'

# Slicing [start:stop:step]
print(f"\n  text[0:4] = '{text[0:4]}'")  # 'Data' (index 0-3)
print(f"  text[5:] = '{text[5:]}'")  # 'Science' (จาก 5 ถึงจบ)
print(f"  text[:4] = '{text[:4]}'")  # 'Data' (ตั้งแต่แรกถึง 3)
print(f"  text[::2] = '{text[::2]}'")  # 'Dt cec' (ทุกๆ 2 ตัว)
print(f"  text[::-1] = '{text[::-1]}'")  # 'ecneicS ataD' (กลับหลัง!)

# มุก: "[::-1] คือมนต์กลับคำ!" 🔮

# ========== String Methods (เยอะมาก!) ==========

text = "  Hello World  "

print(f"\n🔧 String Methods:")
print(f"  Original: '{text}'")
print(f"  lower(): '{text.lower()}'")  # ตัวพิมพ์เล็กทั้งหมด
print(f"  upper(): '{text.upper()}'")  # ตัวพิมพ์ใหญ่ทั้งหมด
print(f"  title(): '{text.title()}'")  # ตัวแรกของแต่ละคำเป็นใหญ่
print(f"  strip(): '{text.strip()}'")  # ตัดช่องว่างหน้าหลัง
print(f"  replace('World', 'Python'): '{text.replace('World', 'Python')}'")

# ตรวจสอบ
email = "user@example.com"
print(f"\n  '{email}'.startswith('user') = {email.startswith('user')}")
print(f"  '{email}'.endswith('.com') = {email.endswith('.com')}")
print(f"  '{email}'.count('e') = {email.count('e')}")  # นับตัว 'e'
print(f"  '{email}'.find('@') = {email.find('@')}")  # หาตำแหน่ง @

# แยก/ต่อ string
sentence = "Python is awesome"
words = sentence.split()  # แยกด้วยช่องว่าง
print(f"\n  split(): {words}")

joined = "-".join(words)  # ต่อด้วย -
print(f"  join(): '{joined}'")

# มุก: "split คือ 'แยก', join คือ 'ต่อ' - คู่แท้!" 💑

# ========== f-strings (Format Strings) - Python 3.6+ ==========

name = "Alice"
age = 25
height = 1.65

# f-string (แนะนำ!)
print(f"\nสวัสดี ฉันชื่อ {name} อายุ {age} ปี สูง {height} ม.")

# format ตัวเลข
pi = 3.14159
print(f"π = {pi:.2f}")  # 2 ทศนิยม = 3.14
print(f"π = {pi:.4f}")  # 4 ทศนิยม = 3.1416

price = 12345.67
print(f"ราคา: {price:,.2f} บาท")  # 12,345.67 (มี comma)

# alignment
print(f"{'Left':<10}|")  # ชิดซ้าย
print(f"{'Center':^10}|")  # กึ่งกลาง
print(f"{'Right':>10}|")  # ชิดขวา

# มุก: "f-string = F***ing awesome string!" 😎

# Expression ใน f-string
a = 10
b = 20
print(f"{a} + {b} = {a + b}")  # คำนวณได้เลย!
print(f"{'PASS' if age >= 18 else 'FAIL'}")  # if-else ได้ด้วย!

# ========== ตัวอย่างการใช้งานจริง ==========

def validate_password(password):
    """
    ตรวจสอบความแข็งแกร่งของรหัสผ่าน

    เกณฑ์:
    - ยาวอย่างน้อย 8 ตัว
    - มีตัวพิมพ์ใหญ่
    - มีตัวพิมพ์เล็ก
    - มีตัวเลข
    - มีอักขระพิเศษ

    มุก: "รหัสผ่านที่ดี = รหัสผ่านที่จำไม่ได้!" 🔐😅
    """
    issues = []

    if len(password) < 8:
        issues.append("❌ สั้นเกินไป (ต้องอย่างน้อย 8 ตัว)")

    if not any(c.isupper() for c in password):
        issues.append("❌ ไม่มีตัวพิมพ์ใหญ่")

    if not any(c.islower() for c in password):
        issues.append("❌ ไม่มีตัวพิมพ์เล็ก")

    if not any(c.isdigit() for c in password):
        issues.append("❌ ไม่มีตัวเลข")

    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        issues.append("❌ ไม่มีอักขระพิเศษ")

    if issues:
        print(f"🔒 Password: '{password}'")
        print(f"   Strength: ضعيف (Weak)")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print(f"🔒 Password: '{password}'")
        print(f"   Strength: 💪 แข็งแกร่ง! (Strong)")
        return True

# ทดสอบ
print("\n" + "="*50)
validate_password("password")  # อ่อนแอ
print()
validate_password("Pass123!")  # แข็งแกร่ง!
```

**มุกพิเศษ:** "String ใน Python คือ immutable (เปลี่ยนแปลงไม่ได้) - เหมือนรอยสักที่ไม่สามารถลบได้! (แต่สร้างใหม่ได้) 🎨"

---

### 📝 สรุป Part 1

เราได้เรียนรู้:
✅ Data Science คืออะไรแบบละเอียด
✅ ตัวอย่างการใช้งานจริง (Netflix, Grab)
✅ Python Numbers - ทุก data type
✅ Python Strings - ทุก method และ technique
✅ มุกสนุกๆ เพิ่มความจำ!

**ต่อไปเราจะเจาะลึก:**
- Lists, Tuples, Dictionaries, Sets
- Control Flow (if, loops)
- Functions ทุกรูปแบบ
- File I/O
- Exception Handling

---

## 🎯 ตัวอย่างโปรเจกต์จริง: Password Manager

```python
"""
Password Manager - โปรแกรมจัดการรหัสผ่าน

Features:
1. เก็บรหัสผ่านอย่างปลอดภัย (encrypted)
2. generate รหัสผ่านแบบสุ่ม
3. ตรวจสอบความแข็งแกร่ง
4. คัดลอกไปยัง clipboard

มุก: "ใช้โปรแกรมนี้ แล้วคุณจะจำรหัสผ่านไม่ได้อีกต่อไป!" 😂
"""

import secrets
import string
from datetime import datetime

class PasswordManager:
    def __init__(self):
        self.passwords = {}
        self.created_at = {}

    def generate_password(
        self,
        length=16,
        use_uppercase=True,
        use_lowercase=True,
        use_digits=True,
        use_special=True
    ):
        """
        สร้างรหัสผ่านแบบสุ่มที่แข็งแกร่ง

        มุก: "ใช้ secrets (ไม่ใช่ random) - ปลอดภัยกว่า!"
        """
        characters = ""

        if use_lowercase:
            characters += string.ascii_lowercase  # abcd...
        if use_uppercase:
            characters += string.ascii_uppercase  # ABCD...
        if use_digits:
            characters += string.digits  # 0123...
        if use_special:
            characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"

        if not characters:
            raise ValueError("ต้องเลือกอย่างน้อย 1 ประเภท!")

        # สร้างรหัสผ่าน
        password = ''.join(secrets.choice(characters) for _ in range(length))

        return password

    def check_strength(self, password):
        """ตรวจสอบความแข็งแกร่ง"""
        score = 0
        feedback = []

        # ความยาว
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        else:
            feedback.append("📏 สั้นเกินไป")

        # ความหลากหลาย
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("🔤 ขาดตัวพิมพ์ใหญ่")

        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("🔡 ขาดตัวพิมพ์เล็ก")

        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("🔢 ขาดตัวเลข")

        if any(c in "!@#$%^&*" for c in password):
            score += 1
        else:
            feedback.append("🔣 ขาดอักขระพิเศษ")

        # ประเมินผล
        if score >= 6:
            strength = "💪 แข็งแกร่งมาก!"
            color = "green"
        elif score >= 4:
            strength = "👍 ดีพอใช้"
            color = "yellow"
        else:
            strength = "😰 อ่อนแอ!"
            color = "red"

        return {
            'score': score,
            'max_score': 6,
            'strength': strength,
            'feedback': feedback
        }

    def save_password(self, service, username, password):
        """บันทึกรหัสผ่าน"""
        self.passwords[service] = {
            'username': username,
            'password': password
        }
        self.created_at[service] = datetime.now()
        print(f"✅ บันทึก: {service} ({username})")

    def get_password(self, service):
        """ดึงรหัสผ่าน"""
        if service in self.passwords:
            data = self.passwords[service]
            created = self.created_at[service]
            return {
                **data,
                'created_at': created,
                'age_days': (datetime.now() - created).days
            }
        return None

    def list_all(self):
        """แสดงรายการทั้งหมด"""
        if not self.passwords:
            print("📭 ไม่มีรหัสผ่านที่บันทึกไว้")
            return

        print(f"\n🔐 Password Manager ({len(self.passwords)} entries)")
        print("=" * 70)

        for service, data in self.passwords.items():
            info = self.get_password(service)
            print(f"\n📌 {service}")
            print(f"   👤 Username: {data['username']}")
            print(f"   🔑 Password: {'*' * len(data['password'])}")
            print(f"   📅 Created: {info['created_at'].strftime('%Y-%m-%d')}")
            print(f"   ⏰ Age: {info['age_days']} days")

            # เตือนถ้ารหัสผ่านเก่า
            if info['age_days'] > 90:
                print(f"   ⚠️  ควรเปลี่ยนรหัสผ่าน! (เก่ามากกว่า 90 วัน)")

# ========== ทดสอบ ==========

pm = PasswordManager()

print("🎯 Password Manager Demo")
print("=" * 70)

# สร้างรหัสผ่าน
print("\n1️⃣ สร้างรหัสผ่านแบบสุ่ม:")
for i in range(3):
    password = pm.generate_password(length=16)
    strength = pm.check_strength(password)
    print(f"\n  Password {i+1}: {password}")
    print(f"  Strength: {strength['strength']} ({strength['score']}/{strength['max_score']})")
    if strength['feedback']:
        for fb in strength['feedback']:
            print(f"    - {fb}")

# บันทึกรหัสผ่าน
print("\n\n2️⃣ บันทึกรหัสผ่าน:")
pm.save_password("Gmail", "alice@gmail.com", pm.generate_password())
pm.save_password("Facebook", "alice.smith", pm.generate_password())
pm.save_password("Netflix", "alice123", pm.generate_password())

# แสดงรายการ
print("\n\n3️⃣ รายการทั้งหมด:")
pm.list_all()

# ดึงรหัสผ่าน
print("\n\n4️⃣ ดึงรหัสผ่าน Gmail:")
gmail_info = pm.get_password("Gmail")
print(f"   Username: {gmail_info['username']}")
print(f"   Password: {gmail_info['password']}")  # แสดงจริงๆ

print("\n\n💡 Tips:")
print("  - ใช้รหัสผ่านที่แตกต่างกันสำหรับแต่ละเว็บไซต์")
print("  - เปลี่ยนรหัสผ่านทุก 3-6 เดือน")
print("  - ไม่ควรแชร์รหัสผ่านกับใคร")
print("  - เปิด 2FA (Two-Factor Authentication) ถ้าได้")
print("\n  มุก: 'รหัสผ่านที่ดี = รหัสที่แม้แต่คุณเองก็จำไม่ได้!' 😅")
```

---

### ✨ ทำไมเอกสารนี้พิเศษ?

1. **อธิบายละเอียดมากกว่า** 10x
2. **มีมุกทุกหัวข้อ** - เรียนสนุก!
3. **ตัวอย่างโปรเจกต์จริง** - ใช้งานได้เลย
4. **อธิบายเหตุผล "ทำไม"** - ไม่ใช่แค่ "อย่างไร"
5. **คำเปรียบเทียบชีวิตจริง** - จำง่าย!

---

พร้อมที่จะดำน้ำลึกต่อไปไหมครับ? เรายังมีอีกเยอะ! 🚀

**Coming Next:**
- Lists, Dictionaries (ละเอียดยิบ!)
- Functions & OOP (ครบทุกเทคนิค!)
- NumPy & Pandas (มาสเตอร์เลย!)
- Machine Learning (เจาะลึกทุกอัลกอริทึม!)

```python
print("Happy Deep Learning! 🎓✨")
print("ความรู้ไม่มีที่สิ้นสุด แต่มุกมี! 😂")
```
