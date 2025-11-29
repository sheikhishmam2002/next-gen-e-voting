# face_recognition_pi.py
# Combined: Face Recognition (MobileFaceNet) + SQLite DB + Raspberry Pi GPIO Hardware

import os
import time
import sqlite3

import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np

import RPi.GPIO as GPIO  # Only works on Raspberry Pi


# =========================
# DB CONFIG (SQLite)
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "evm.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn


def get_voter_by_id(voter_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, voter_id, name, vote_status, vote_party FROM voters WHERE voter_id = ?",
        (voter_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row  # None if not found


def update_vote(voter_id, party_name):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE voters
        SET vote_status = 1, vote_party = ?
        WHERE voter_id = ?
        """,
        (party_name, voter_id),
    )
    conn.commit()
    conn.close()


# =========================
# HARDWARE CONFIG (GPIO + LCD + SERVO + BUTTONS)
# =========================
# Adjust these pins if your wiring is different
LCD_RS = 7
LCD_E = 11
LCD_D4 = 12
LCD_D5 = 13
LCD_D6 = 15
LCD_D7 = 16

SW1 = 29  # Party A button
SW2 = 31  # Party B button
SW3 = 32  # Party C button

BUZZER = 33
LED_MATCH = 35
LED_NOMATCH = 37
SERVO_PIN = 22

# Optional in-RAM counters, DB remains the main source of truth
vote_counts = {"A": 0, "B": 0, "C": 0}

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

GPIO.setup(LCD_RS, GPIO.OUT)
GPIO.setup(LCD_E, GPIO.OUT)
GPIO.setup(LCD_D4, GPIO.OUT)
GPIO.setup(LCD_D5, GPIO.OUT)
GPIO.setup(LCD_D6, GPIO.OUT)
GPIO.setup(LCD_D7, GPIO.OUT)

GPIO.setup(BUZZER, GPIO.OUT)
GPIO.setup(LED_MATCH, GPIO.OUT)
GPIO.setup(LED_NOMATCH, GPIO.OUT)

GPIO.setup(SW1, GPIO.IN)
GPIO.setup(SW2, GPIO.IN)
GPIO.setup(SW3, GPIO.IN)

GPIO.setup(SERVO_PIN, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)


# ----- LCD helpers -----
def pulse():
    GPIO.output(LCD_E, 1)
    time.sleep(0.0005)
    GPIO.output(LCD_E, 0)
    time.sleep(0.0005)


def lcd_nibble(data):
    GPIO.output(LCD_D4, (data >> 0) & 1)
    GPIO.output(LCD_D5, (data >> 1) & 1)
    GPIO.output(LCD_D6, (data >> 2) & 1)
    GPIO.output(LCD_D7, (data >> 3) & 1)
    pulse()


def lcd_byte(value, mode):
    GPIO.output(LCD_RS, mode)
    lcd_nibble(value >> 4)
    lcd_nibble(value & 0x0F)


def lcd_init():
    lcd_byte(0x33, 0)
    lcd_byte(0x32, 0)
    lcd_byte(0x28, 0)
    lcd_byte(0x0C, 0)
    lcd_byte(0x06, 0)
    lcd_byte(0x01, 0)
    time.sleep(0.003)


def lcd_print(msg, line=1):
    # line = 1 or 2
    lcd_byte(0x80 if line == 1 else 0xC0, 0)
    for ch in msg.ljust(16):
        lcd_byte(ord(ch), 1)


def lcd_clear():
    lcd_byte(0x01, 0)
    time.sleep(0.003)


# ----- Servo helpers -----
def gate_open():
    # Adjust duty cycle values as needed for your servo
    servo.ChangeDutyCycle(7.5)  # ~90 degrees
    time.sleep(1)
    servo.ChangeDutyCycle(0)


def gate_close():
    servo.ChangeDutyCycle(2.5)  # ~0 degrees
    time.sleep(1)
    servo.ChangeDutyCycle(0)


# ----- Voting buttons -----
def detect_vote():
    """
    Wait for voter to press a button,
    increment in-memory count, and return vote code ("A", "B", or "C").
    """
    lcd_print("Vote Now       ", 1)
    lcd_print("Press A/B/C    ", 2)
    print("Waiting for vote button (A/B/C)...")

    while True:
        if GPIO.input(SW1) == 1:
            vote_counts["A"] += 1
            print("Button A pressed")
            return "A"
        elif GPIO.input(SW2) == 1:
            vote_counts["B"] += 1
            print("Button B pressed")
            return "B"
        elif GPIO.input(SW3) == 1:
            vote_counts["C"] += 1
            print("Button C pressed")
            return "C"
        time.sleep(0.1)


def buzzer_beep(duration=0.3):
    GPIO.output(BUZZER, 1)
    time.sleep(duration)
    GPIO.output(BUZZER, 0)


# Map hardware choice to real party names stored in DB
PARTY_MAP = {
    "A": "Party Alpha",
    "B": "Party Beta",
    "C": "Party Gamma",
}


# =========================
# FACE RECOGNITION SETUP (MobileFaceNet)
# =========================
print("\n" + "=" * 60)
print("LOADING MOBILEFACENET MODEL")
print("=" * 60)

app_mobile = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app_mobile.prepare(ctx_id=0, det_size=(320, 320))
print("MobileFaceNet model loaded")


def get_embedding_mobilefacenet(image):
    """
    Get 128-d embedding using MobileFaceNet
    Input: BGR or RGB image
    Output: 128-dimensional embedding or None
    """
    faces = app_mobile.get(image)
    if len(faces) == 0:
        return None, None

    face = faces[0]
    embedding = face.embedding
    bbox = face.bbox
    return embedding, bbox


def compute_dis(em1, em2):
    """Calculate L2 distance"""
    em1 = em1 / np.linalg.norm(em1)
    em2 = em2 / np.linalg.norm(em2)
    return np.linalg.norm(em1 - em2)


def get_dataset_mobilefacenet(folder_name):
    """
    Create dataset from folder structure:

    folder_name/
        V001/
            img1.jpg
            img2.jpg
        V002/
            ...

    Each folder name (V001, V002, ...) is the voter_id in DB.
    """
    dataset = {}

    for file in os.listdir(folder_name):
        path = os.path.join(folder_name, file)
        if not os.path.isdir(path):
            continue

        embeddings = []
        print(f"\n Processing: {file}")

        for filename in os.listdir(path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"  Could not read: {filename}")
                continue

            em, bbox = get_embedding_mobilefacenet(image)

            if em is None:
                print(f"  No face: {filename}")
                continue

            embeddings.append(em)
            print(f"   {filename}")

        if embeddings:
            dataset[file] = np.mean(embeddings, axis=0)
            print(f"  Averaged {len(embeddings)} embeddings")

    print(f"\n Dataset created for: {list(dataset.keys())}")
    return dataset


def save_emb(dataset, file_name="face_embeddings_mobilefacenet.npz"):
    """Save embeddings"""
    names = list(dataset.keys())
    embad = [dataset[name] for name in names]
    np.savez(file_name, names=names, embadding=embad)
    print(f"Saved to {file_name}")


def load_emb(file_name="face_embeddings_mobilefacenet.npz"):
    """Load embeddings"""
    if not os.path.exists(file_name):
        print("No embeddings file found, returning None")
        return None
    data = np.load(file_name, allow_pickle=True)
    names = data["names"]
    embadding = data["embadding"]
    dataset = {name: emb for name, emb in zip(names, embadding)}
    return dataset


def recognize_mobilefacenet(dataset, image, threshold=1.1):
    """
    Recognize face
    threshold: 1.1 is good for MobileFaceNet
    Returns: identity (str), distance (float), bbox
    """
    emb, bbox = get_embedding_mobilefacenet(image)

    if emb is None:
        return "unknown", 999, None

    min_dist = 100
    identity = "unknown"

    for nam, emba in dataset.items():
        dis = compute_dis(emba, emb)
        print(f"  Distance to {nam}: {dis:.4f}")
        if dis < min_dist:
            min_dist = dis
            identity = nam

    if min_dist > threshold:
        return "unknown", min_dist, bbox
    return identity, min_dist, bbox


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    lcd_init()
    lcd_print("Scan your face", 1)
    lcd_print("Please wait...", 2)

    embeddings_file = "face_embeddings_mobilefacenet.npz"
    images_folder = r"images"

    # If embeddings already exist, load them.
    # Otherwise, create from images and save.
    if os.path.exists(embeddings_file):
        print("Loading existing embeddings...")
        dataset_mobile = load_emb(embeddings_file)
    else:
        print("No embeddings found. Creating new dataset...")
        dataset_mobile = get_dataset_mobilefacenet(images_folder)
        if dataset_mobile is not None:
            save_emb(dataset_mobile, embeddings_file)

    if not dataset_mobile:
        print("No dataset available. Exiting.")
        GPIO.cleanup()
        exit(1)

    print("\nStarting webcam (MobileFaceNet)...")
    cap = cv2.VideoCapture(0)

    frame_count = 0
    process_every = 2  # Process every 2nd frame for speed

    last_identity = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            name = "unknown"
            dist = 999
            bbox = None

            # Process every Nth frame
            if frame_count % process_every == 0:
                name, dist, bbox = recognize_mobilefacenet(
                    dataset_mobile, frame, threshold=1.1
                )

                # Optional live preview
                if bbox is not None:
                    x1, y1, x2, y2 = bbox.astype(int)
                    color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{name} ({dist:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                # ------------------------
                # DB Logic: Check voter & vote
                # ------------------------
                if name != "unknown" and name != last_identity:
                    print(f"\nRecognized voter_id: {name}")
                    voter = get_voter_by_id(name)

                    if voter is None:
                        print("⚠️  Voter not found in database. Access denied.")
                        GPIO.output(LED_MATCH, 0)
                        GPIO.output(LED_NOMATCH, 1)
                        lcd_print("Unknown Voter   ", 1)
                        lcd_print("Access Denied   ", 2)
                        buzzer_beep(0.5)
                        time.sleep(2)
                        lcd_print("Scan your face  ", 1)
                        lcd_print("Please wait...  ", 2)
                    else:
                        print(
                            f"Voter in DB: {voter['name']} (ID={voter['voter_id']}), "
                            f"vote_status={voter['vote_status']}, vote_party={voter['vote_party']}"
                        )

                        if voter["vote_status"] == 1:
                            print("❌ This voter has already voted. Do not open gate.")
                            GPIO.output(LED_MATCH, 0)
                            GPIO.output(LED_NOMATCH, 1)
                            lcd_print("Already Voted   ", 1)
                            lcd_print("Access Denied   ", 2)
                            buzzer_beep(0.5)
                            time.sleep(2)
                            lcd_print("Scan your face  ", 1)
                            lcd_print("Please wait...  ", 2)
                        else:
                            print("✅ Voter is allowed to vote. Open gate & enable circuit.")
                            GPIO.output(LED_NOMATCH, 0)
                            GPIO.output(LED_MATCH, 1)

                            lcd_print("Face Matched    ", 1)
                            lcd_print("Gate Opening... ", 2)
                            gate_open()
                            time.sleep(1)

                            # Wait for hardware to register chosen party
                            vote_code = detect_vote()
                            party_name = PARTY_MAP.get(vote_code)

                            if party_name:
                                update_vote(name, party_name)
                                print(
                                    f"✅ Vote stored: voter_id={name}, party={party_name}"
                                )
                                buzzer_beep(0.3)
                                lcd_print("Exit the booth  ", 1)
                                lcd_print("Thank you       ", 2)
                                time.sleep(3)

                                lcd_print("Closing Gate    ", 1)
                                lcd_print("                ", 2)
                                gate_close()

                                lcd_clear()
                                lcd_print("Vote Saved      ", 1)
                                lcd_print(f"Vote: {vote_code}", 2)
                                time.sleep(3)

                                GPIO.output(LED_MATCH, 0)
                                lcd_print("Scan your face  ", 1)
                                lcd_print("Please wait...  ", 2)
                            else:
                                print("No valid party selected. Vote not stored.")
                                lcd_print("No vote stored  ", 1)
                                lcd_print("Try again       ", 2)
                                time.sleep(3)

                    last_identity = name

            # Show preview window (optional)
            cv2.imshow("Face Recognition (MobileFaceNet)", frame)

            # Press 'q' to quit
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("🔴 Webcam closed, GPIO cleaned up.")
        print("Vote counts (session):", vote_counts)
