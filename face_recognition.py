# face_recognition.py

import os
import sqlite3
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np

# =========================
# DB CONFIG (SQLite)
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "evm.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # so we can access columns by name
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
    Create dataset: folder structure
    folder_name/
        V001/
            img1.jpg
            img2.jpg
        V002/
            ...
    Each folder name (V001, V002, ...) is the voter_id in DB
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
# HARDWARE / VOTE SELECTION (PLACEHOLDER)
# =========================
def wait_for_party_selection():
    """
    TODO: Replace this with your circuit / GPIO logic.

    For now, this is a simple CLI input to simulate the hardware switch
    that selects the voting party.
    """
    print("\n[SIMULATION] Ask voter to choose party using hardware switch.")
    print("[SIMULATION] For testing, type party name here (e.g., 'Party Alpha'):")
    party_name = input("Party selected: ").strip()
    if not party_name:
        return None
    return party_name


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CREATING / LOADING DATASET (MobileFaceNet)")
    print("=" * 60)

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
        exit(1)

    print("\nStarting webcam (MobileFaceNet)...")
    cap = cv2.VideoCapture(0)

    frame_count = 0
    process_every = 2  # Process every 2nd frame for speed

    last_identity = None

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
            if name != "unknown":
                # Only act if new identity (avoid spamming)
                if name != last_identity:
                    print(f"\nRecognized voter_id: {name}")
                    voter = get_voter_by_id(name)

                    if voter is None:
                        print("⚠️  Voter not found in database. Access denied.")
                    else:
                        print(
                            f"Voter in DB: {voter['name']} (ID={voter['voter_id']}), "
                            f"vote_status={voter['vote_status']}, vote_party={voter['vote_party']}"
                        )

                        if voter["vote_status"] == 1:
                            print("❌ This voter has already voted. Do not open gate.")
                        else:
                            print("✅ Voter is allowed to vote. Open gate & enable circuit.")
                            # TODO: gate open / circuit ON (using GPIO on Raspberry Pi)
                            # e.g., GPIO.output(GATE_PIN, GPIO.HIGH)

                            # Wait for hardware to register chosen party
                            selected_party = wait_for_party_selection()
                            if selected_party:
                                update_vote(name, selected_party)
                                print(
                                    f"✅ Vote stored: voter_id={name}, party={selected_party}"
                                )
                            else:
                                print("No party selected. Vote not stored.")

                    last_identity = name

        cv2.imshow("Face Recognition (MobileFaceNet)", frame)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🔴 Webcam closed")
