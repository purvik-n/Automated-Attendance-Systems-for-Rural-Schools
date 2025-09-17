#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recognize faces from webcam using data/embeddings/embeddings.pkl
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

YOLO_FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
YOLO_FACE_LOCAL = Path.cwd() / "yolov8n-face.pt"
EMBEDDINGS_PATH = Path("data/embeddings/embeddings.pkl")
TARGET_SIZE = 224
RECOG_SIM_THRESHOLD = 0.5  # declare a match if cosine similarity >= threshold

def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_weights(path: Path, url: str) -> str:
    if path.exists():
        return str(path)
    print(f"[INFO] Downloading YOLO face weights to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(url, str(path))
    print("[INFO] Download complete.")
    return str(path)

def align_simple(img_bgr, box, margin=0.2):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    s = int(max(w, h) * (1 + margin))
    x1 = int(max(0, cx - s / 2))
    y1 = int(max(0, cy - s / 2))
    x2 = int(min(img_bgr.shape[1], x1 + s))
    y2 = int(min(img_bgr.shape[0], y1 + s))
    face = img_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

def preprocess(rgb):
    img = cv2.resize(rgb, (160, 160))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(device_str())

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def main():
    if not EMBEDDINGS_PATH.exists():
        print(f"[ERROR] {EMBEDDINGS_PATH} not found. Run build_embeddings first.")
        return
    with open(EMBEDDINGS_PATH, "rb") as f:
        records = pickle.load(f)
    if not records:
        print("[ERROR] No embeddings present.")
        return

    weights_path = ensure_weights(YOLO_FACE_LOCAL, YOLO_FACE_URL)
    yolo = YOLO(weights_path)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device_str())

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Recognizing... ESC to exit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = yolo(frame, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            cv2.imshow("Recognize", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            rgb = align_simple(frame, [x1, y1, x2, y2])
            if rgb is None:
                continue
            with torch.no_grad():
                vec = model(preprocess(rgb)).cpu().numpy()[0]

            # find best match
            best_sim, best = -1.0, None
            for rec in records:
                sim = cosine_similarity(vec, rec["embedding"])
                if sim > best_sim:
                    best_sim, best = sim, rec

            label = "Unknown"
            if best is not None and best_sim >= RECOG_SIM_THRESHOLD:
                label = f"{best['student_id']} - {best['name']} ({best_sim:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

        cv2.imshow("Recognize", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import cv2
    main()