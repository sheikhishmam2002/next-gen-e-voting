# 📘 Next-Gen Smart & Secure e-Voting Machine
*A Modern, Automated, and AI-Powered Voting System*

## 📌 Overview
The **Next-Gen Smart & Secure e-Voting Machine** is an advanced voting system designed to bring **security, automation, and transparency** to the election process. The project integrates **AI-based face recognition**, **Raspberry Pi hardware**, **secure database management**, and a **Flask-based admin dashboard** to ensure a streamlined and reliable voting experience.

This system reduces dependency on manual verification, minimizes human error, and establishes a trustworthy, technology-driven voting environment.

---

## 🧠 Key Features

### ✅ AI-Based Voter Authentication
- Uses **InsightFace (MobileFaceNet)** for real-time facial recognition.  
- Matches voters with pre-registered datasets.  
- Prevents multiple voting using `vote_status`.

### ✅ Automated Voting Workflow
- Once the voter is authenticated:
  - Gate opens using servo motor  
  - LCD instructs voter  
  - Buttons allow party selection  
  - Gate closes after voting  
- Vote gets instantly updated in the local SQLite database.

### ✅ Secure Local Database (SQLite)
Stores:
- `voter_id`  
- `name`  
- `vote_status` (True/False)  
- `vote_party`  

Ensures each voter can vote **only once**.

### ✅ Smart Admin Dashboard (Flask)
- Admin login (`admin` / `1234`)  
- Dashboard shows:
  - Voter list  
  - Who voted  
  - Party vote counts  
  - Ranked parties  
  - Winning party  
- Clean UI using template inheritance.

### ✅ Raspberry Pi Hardware Integration
- **16x2 LCD Display** for user instructions  
- **Physical buttons** for party selection  
- **Servo motor** for gate mechanism  
- **LED indicators** for match & mismatch  
- **Buzzer** for system alerts  
- Works in real-time with the face recognition system.

---

## 🔧 Technology Stack

### Software
- Python  
- Flask  
- SQLite  
- InsightFace  
- OpenCV  
- NumPy  

### Hardware
- Raspberry Pi 4  
- Servo Motor  
- Push Buttons  
- LED Indicators  
- Buzzer  
- LCD Display  