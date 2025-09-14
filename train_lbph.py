# train_lbph.py
import os, json, argparse, cv2
import numpy as np

def load_dataset(root):
    X, y = [], []
    label_map, next_label = {}, 0
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir): 
            continue
        if person not in label_map:
            label_map[person] = next_label
            next_label += 1
        lbl = label_map[person]
        for fn in os.listdir(pdir):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(pdir, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Standardize to 200x200 in case sizes differ
            if img.shape != (200, 200):
                img = cv2.resize(img, (200, 200), cv2.INTER_AREA)
            X.append(img)
            y.append(lbl)
    return X, np.array(y, dtype=np.int32), label_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/faces", help="Root folder with subfolders per ID")
    ap.add_argument("--model", default="models/lbph.yml")
    ap.add_argument("--labels", default="models/labels.json")
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--neighbors", type=int, default=8)
    ap.add_argument("--grid_x", type=int, default=8)
    ap.add_argument("--grid_y", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=70.0, help="Lower = stricter, Higher = more permissive")
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        print(f"[ERROR] Dataset folder not found: {args.data}")
        return

    X, y, label_map = load_dataset(args.data)
    if len(X) == 0:
        print("[ERROR] No images found. Run capture_samples.py first.")
        return

    # LBPH recognizer is in opencv-contrib-python
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=args.radius,
        neighbors=args.neighbors,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        threshold=args.threshold
    )

    recognizer.train(X, y)
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    recognizer.save(args.model)
    os.makedirs(os.path.dirname(args.labels), exist_ok=True)
    with open(args.labels, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[OK] Trained on {len(X)} images across {len(label_map)} IDs")
    print(f"[SAVED] Model: {args.model}")
    print(f"[SAVED] Labels: {args.labels}")
    print(f"[INFO] Threshold set to {args.threshold:.1f} (lower=more strict)")

if __name__ == "__main__":
    main()
