#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Register faces into data/enrolled/<student_id>_<name> as aligned 224x224 JPGs.
During capture, reject faces that match other students' faces using embeddings.

Install once in your venv:
    pip install ultralytics face-alignment facenet-pytorch torch torchvision pillow opencv-python numpy tqdm scikit-learn

Then build embeddings for already enrolled students before (or after) registering new ones:
    python src/build_embeddings.py
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import sys
import uuid
import urllib.request
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from ultralytics import YOLO
import face_alignment
from facenet_pytorch import InceptionResnetV1

# Config
TARGET_SIZE = 224
YOLO_FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
YOLO_FACE_LOCAL = Path.cwd() / "yolov8n-face.pt"  # download to project root
EMBEDDINGS_PATH = Path("data/embeddings/embeddings.pkl")
DUPLICATE_COSINE_THRESHOLD = 0.5  # if similarity >= threshold with other students, reject

def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_weights(path: Path, url: str) -> str:
    if path.exists():
        return str(path)
    print(f"[INFO] Downloading YOLO face weights to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(path))
    print("[INFO] Download complete.")
    return str(path)

def load_models():
    weights_path = ensure_weights(YOLO_FACE_LOCAL, YOLO_FACE_URL)
    face_model = YOLO(weights_path)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=device_str(),
        flip_input=False
    )
    # Facenet model for embeddings
    embed_model = InceptionResnetV1(pretrained="vggface2").eval().to(device_str())
    return face_model, fa, embed_model

def detect_faces_yolo(model: YOLO, img_bgr):
    results = model(img_bgr, verbose=False)[0]
    boxes = []
    if results.boxes is not None and len(results.boxes) > 0:
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

def align_from_landmarks(img_bgr, landmarks):
    if landmarks is None or landmarks.shape[0] < 68:
        return None
    pts = landmarks
    left_eye = pts[[36, 37, 38, 39, 40, 41]].mean(axis=0)
    right_eye = pts[[42, 43, 44, 45, 46, 47]].mean(axis=0)
    nose_tip = pts[30]
    left_mouth = pts[48]
    right_mouth = pts[54]
    src = np.array([left_eye, right_eye, nose_tip, left_mouth, right_mouth], dtype=np.float32)
    dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None
    aligned = cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0)
    aligned = cv2.resize(aligned, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

def crop_from_box(img_bgr, box, margin=0.2):
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

def save_image(rgb_array, path_jpg, quality=95):
    Path(path_jpg).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_array).save(path_jpg, format="JPEG", quality=quality)

def safe_dir_name(student_id, name):
    n = "_".join(str(name).strip().split())
    return f"{student_id}_{n}"

def write_meta(student_dir, student_id, name, images_count):
    meta = {
        "student_id": student_id,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "images_count": int(images_count),
        "uuid": str(uuid.uuid4()),
    }
    with open(student_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta

# ---------- Embeddings helpers ----------
def preprocess_for_embedding(rgb_img_224: np.ndarray) -> torch.Tensor:
    # facenet expects 160x160 usually; it also works with resizing
    img = cv2.resize(rgb_img_224, (160, 160), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    # normalize to [-1, 1]
    img = (img - 0.5) / 0.5
    # HWC -> CHW
    img = np.transpose(img, (0, 1, 2))[..., :3]
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device_str())
    return tensor

def embed_face(rgb_img_224: np.ndarray, embed_model: InceptionResnetV1) -> np.ndarray:
    with torch.no_grad():
        tensor = preprocess_for_embedding(rgb_img_224)
        vec = embed_model(tensor).cpu().numpy()[0]  # 512-d
    return vec

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def load_existing_embeddings(exclude_student_id: str) -> List[Dict]:
    """
    Load all embeddings except those belonging to exclude_student_id.
    Returns a list of dicts with keys: student_id, name, embedding (np.ndarray)
    """
    if not EMBEDDINGS_PATH.exists():
        return []
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    filtered = []
    for rec in data:
        if rec.get("student_id") != exclude_student_id:
            emb = np.array(rec["embedding"], dtype=np.float32)
            filtered.append({
                "student_id": rec["student_id"],
                "name": rec["name"],
                "embedding": emb
            })
    return filtered

# ---------- Pipeline ----------
def process_frame_and_save(img_bgr, face_model, fa, embed_model, out_path, other_embs, duplicate_threshold, stats) -> Tuple[bool, str]:
    boxes = detect_faces_yolo(face_model, img_bgr)
    if not boxes:
        return False, "No face"
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    box = boxes[int(np.argmax(areas))]

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(rgb)
    aligned = None
    if preds and len(preds) > 0:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best_i, best_d = 0, 1e18
        for i, lm in enumerate(preds):
            p = lm.mean(axis=0)
            d = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
            if d < best_d:
                best_d, best_i = d, i
        aligned = align_from_landmarks(img_bgr, np.array(preds[best_i]))

    if aligned is None:
        aligned = crop_from_box(img_bgr, box)
        if aligned is None:
            return False, "Align/crop failed"

    # Duplicate-face guard: compute embedding and compare to others
    emb = embed_face(aligned, embed_model)
    # Check against all other students' embeddings
    for rec in other_embs:
        sim = cosine_similarity(emb, rec["embedding"])
        if sim >= duplicate_threshold:
            stats["duplicate_rejected"] += 1
            return False, f"Matched other student ({rec['student_id']}, sim={sim:.2f})"

    save_image(aligned, out_path)
    return True, "OK"

def capture_from_camera(student_dir, student_id, name, count, camera_index=0):
    face_model, fa, embed_model = load_models()
    # Load other students’ embeddings for duplicate checks
    other_embs = load_existing_embeddings(exclude_student_id=student_id)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        return 0

    saved = 0
    stats = {"duplicate_rejected": 0}
    print("[INFO] Press SPACE to capture, ESC to exit")
    while saved < count:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame")
            break

        # live draw
        disp = frame.copy()
        for b in detect_faces_yolo(face_model, frame):
            cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        status_text = f"{name} ({student_id})  Saved: {saved}/{count}  RejectedDup: {stats['duplicate_rejected']}"
        cv2.putText(disp, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, "SPACE: capture | ESC: exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Register", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k == 32:  # SPACE
            out = student_dir / f"img_{saved + 1:03d}.jpg"
            ok, msg = process_frame_and_save(
                frame, face_model, fa, embed_model, out, other_embs, DUPLICATE_COSINE_THRESHOLD, stats
            )
            if ok:
                saved += 1
                print(f"[OK] {out}")
            else:
                print(f"[REJECTED] {msg}. Try again with a clearer frame.")
    cap.release()
    cv2.destroyAllWindows()
    return saved

def save_from_files(student_dir, student_id, files):
    face_model, fa, embed_model = load_models()
    other_embs = load_existing_embeddings(exclude_student_id=student_id)
    saved = 0
    stats = {"duplicate_rejected": 0}
    for p in tqdm(files, desc="Files"):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Cannot read {p}")
            continue
        out = student_dir / f"img_{saved + 1:03d}.jpg"
        ok, msg = process_frame_and_save(
            img, face_model, fa, embed_model, out, other_embs, DUPLICATE_COSINE_THRESHOLD, stats
        )
        if ok:
            saved += 1
            print(f"[OK] {out}")
        else:
            print(f"[REJECTED] {p} -> {msg}")
    print(f"[STATS] Rejected as duplicate: {stats['duplicate_rejected']}")
    return saved

def main():
    parser = argparse.ArgumentParser(description="Register face images into data/enrolled (with duplicate guard).")
    parser.add_argument("--student_id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--count", type=int, default=10, help="Images to capture from camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--files", nargs="*", help="Optional: image files instead of camera")
    args = parser.parse_args()

    base = Path("data/enrolled")
    student_dir = base / safe_dir_name(args.student_id, args.name)
    student_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        files = [Path(p) for p in args.files]
        saved = save_from_files(student_dir, args.student_id, files)
    else:
        saved = capture_from_camera(student_dir, args.student_id, args.name, args.count, args.camera)

    write_meta(student_dir, args.student_id, args.name, saved)
    print(f"[DONE] Registered {saved} image(s).")
    print(f"[PATH] {student_dir}")
    print(f"[META] {student_dir / 'meta.json'}")
    if saved == 0:
        sys.exit(2)

if __name__ == "__main__":
    main()