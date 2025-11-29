📘 Next-Gen Smart & Secure e-Voting Machine

A Modern, Automated, and AI-Powered Voting System

📌 Overview

The Next-Gen Smart & Secure e-Voting Machine is an advanced voting system designed to bring security, automation, and transparency to the election process. The project integrates AI-based face recognition, Raspberry Pi hardware, secure data management, and a Flask-based admin dashboard to ensure a streamlined and reliable voting experience.

This system reduces dependency on manual verification, minimizes errors, and establishes a trustworthy, technology-driven voting environment.

🧠 Key Features
✅ AI-Based Voter Authentication

Utilizes InsightFace MobileFaceNet for real-time facial recognition.

Matches voters with pre-registered face datasets.

Prevents double voting using database-controlled vote_status.

✅ Automated Voting Workflow

Once a voter is authenticated:

Gate opens automatically

Voting panel activates

Voter selects a party using physical hardware buttons

Gate closes after vote completion

✅ Secure Local Database (SQLite)

Stores voter details:

voter_id

name

vote_status

vote_party

Ensures each voter can vote only once.

✅ Smart Admin Dashboard (Flask Web App)

Admin login (ID: admin, Password: 1234)

Real-time dashboard showing:

Voter list

Voting status

Party-wise vote counts

Real-time party ranking

Winning party prediction

Elegant UI with navigation bar and template inheritance.

✅ Raspberry Pi Hardware Integration

16×2 LCD display for guidance

Button panel for party selection

Servo motor for gate mechanism

LED indicators for access validation

Buzzer for feedback

Real-time communication between hardware, AI model, and database.

🔧 Technology Stack
Software

Python

Flask

SQLite

InsightFace

OpenCV

NumPy

Hardware

Raspberry Pi 4

LCD Display

Push Buttons

Servo Motor

LEDs + Buzzer

🗂 Project Structure
NextGenEVM/
│
├── app.py                     # Flask admin dashboard
├── face_recognition.py        # Laptop simulation mode
├── face_recognition_pi.py     # Raspberry Pi hardware + face recognition
├── db_init.py                 # Initializes SQLite database and voters
├── evm.db                     # SQLite database file
│
├── images/                    # Face datasets (V001, V002, ...)
│
├── templates/
│   ├── base.html
│   ├── login.html
│   └── dashboard.html
│
└── static/
    ├── css/style.css
    └── js/main.js

🚀 How It Works

Voter stands in front of the camera.

System detects and recognizes the face.

If verified and not yet voted:

Gate opens

LCD gives instructions

Voter selects party via button press

System updates the database with the vote.

Admin dashboard displays results instantly.

🛡️ Why This Project Matters

Traditional voting systems rely heavily on manual verification, which can lead to delays, errors, and inconsistencies.

This project brings:

AI-driven identity verification

Automated vote capture

Database-backed vote control

Real-time result monitoring

By combining AI and IoT, the system makes voting secure, transparent, and efficient.

🧪 Usage Instructions
Admin Dashboard
Username: admin  
Password: 1234

Run Face Recognition (Laptop Simulation)
python3 face_recognition.py

Run Full Hardware Version (Raspberry Pi)
python3 face_recognition_pi.py

📚 Future Enhancements

Cloud-based vote syncing

Encrypted communication channel

Fingerprint / RFID integration

Multi-camera support

Remote monitoring dashboard