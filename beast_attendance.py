import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import insightface

# ---------------- CONFIG ----------------
ATTENDANCE_DIR = "attendance"
DB_PATH = "face_db"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ✅ Load YOLOv8n (auto-download)
face_detector = YOLO("yolov8n.pt")

# ✅ Load ArcFace via InsightFace (no TensorFlow!)
arcface = insightface.app.FaceAnalysis(name="arcface_r100_v1")
arcface.prepare(ctx_id=0, det_size=(128,128))

# ---------------- Attendance ----------------
def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=["Name", "Time"])

    if name not in df["Name"].values:
        now = datetime.now().strftime("%H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(file, index=False)
        print(f"[MARKED] {name} at {now}")
    else:
        print(f"[SKIPPED] {name} already marked!")

# ---------------- Build Face DB ----------------
def build_face_db():
    known = {}
    print("[INFO] Building face database...")
    for person in os.listdir(DB_PATH):
        person_dir = os.path.join(DB_PATH, person)
        if not os.path.isdir(person_dir):
            continue
        reps = []
        for img_file in os.listdir(person_dir):
            path = os.path.join(person_dir, img_file)
            img = cv2.imread(path)
            result = arcface.get(img)
            if len(result) > 0:
                reps.append(result[0]['embedding'])
        if reps:
            known[person] = np.mean(reps, axis=0)
            print(f" ✔ {person}: {len(reps)} samples")
    print(f"[INFO] Loaded {len(known)} persons.")
    return known

# ---------------- Cosine Similarity ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- MAIN ----------------
def run_attendance():
    known_faces = build_face_db()
    if not known_faces:
        print("[ERROR] No faces in face_db/")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detector(frame, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # keep only "person"
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                faces = arcface.get(face_roi)
                if len(faces) == 0:
                    continue

                emb = faces[0]['embedding']
                best_match, best_score = "Unknown", 0.0
                for name, ref in known_faces.items():
                    sim = cosine_similarity(emb, ref)
                    if sim > best_score:
                        best_match, best_score = name, sim

                if best_score > 0.55:
                    mark_attendance(best_match)
                    label, color = f"{best_match} ({best_score:.2f})", (0, 255, 0)
                else:
                    label, color = "Unknown", (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("🚀 YOLOv8n + InsightFace Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    run_attendance()