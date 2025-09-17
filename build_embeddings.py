#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build embeddings from data/enrolled into data/embeddings/embeddings.pkl
Each record: {student_id, name, path, embedding (list)}
"""

import os
import re
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import InceptionResnetV1

ENROLLED_DIR = Path("data/enrolled")
OUT_PATH = Path("data/embeddings/embeddings.pkl")

def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"

def parse_student_dirname(dirname: str):
    # Expect "<student_id>_<Name with underscores>"
    # student_id is everything before first "_"
    if "_" not in dirname:
        return None, None
    sid, name = dirname.split("_", 1)
    name = name.replace("_", " ").strip()
    return sid.strip(), name

def preprocess_for_embedding(rgb_img: np.ndarray) -> torch.Tensor:
    # resize to 160 and normalize to [-1, 1]
    img = np.asarray(Image.fromarray(rgb_img).resize((160, 160)))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(img).unsqueeze(0).to(device_str())
    return tensor

def embed_image(img_path: Path, embed_model: InceptionResnetV1) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    rgb = np.array(img)
    with torch.no_grad():
        t = preprocess_for_embedding(rgb)
        vec = embed_model(t).cpu().numpy()[0]
    return vec

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device_str())

    records: List[Dict] = []
    if not ENROLLED_DIR.exists():
        print(f"No enrolled dir at {ENROLLED_DIR}. Nothing to do.")
        return

    dirs = [d for d in ENROLLED_DIR.iterdir() if d.is_dir()]
    for d in tqdm(dirs, desc="Students"):
        sid, name = parse_student_dirname(d.name)
        if not sid or not name:
            print(f"[SKIP] Bad folder name: {d.name}")
            continue
        images = sorted([p for p in d.glob("*.jpg")])
        if not images:
            print(f"[WARN] No images in {d}")
            continue
        for p in images:
            try:
                emb = embed_image(p, model)
                records.append({
                    "student_id": sid,
                    "name": name,
                    "path": str(p),
                    "embedding": emb.astype(np.float32)
                })
            except Exception as e:
                print(f"[WARN] Failed {p}: {e}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(records, f)
    print(f"[DONE] Wrote {len(records)} embeddings -> {OUT_PATH}")

if __name__ == "__main__":
    main()