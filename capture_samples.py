# capture_samples.py  (backend-fallback version)
import cv2, os, time, uuid, argparse
import numpy as np
import mediapipe as mp

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def preprocess_gray(img):
    # equalize lighting (CLAHE) + normalize
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(img)
    return cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX)

def try_open_cam(index: int):
    """Try multiple backends and return (cap, backend_name) or (None, None)."""
    tried = []

    # Build a prioritized list of backends, skipping those not present
    candidates = []
    for name in ("CAP_DSHOW", "CAP_MSMF"):
        v = getattr(cv2, name, None)
        if v is not None:
            candidates.append((v, name))
    candidates.append((None, "DEFAULT"))  # default constructor (no backend hint)

    for backend, name in candidates:
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
        ok = cap.isOpened()
        tried.append(name + ("(OK)" if ok else "(FAIL)"))
        if ok:
            # Try setting a reasonable resolution (optional)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"[INFO] Opened camera index {index} via {name}")
            return cap, name
        else:
            cap.release()
    print("[ERROR] Could not open camera. Tried: " + " -> ".join(tried))
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Student ID (folder name)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default="data/faces")
    ap.add_argument("--every", type=int, default=5, help="save 1 of every N frames")
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--far", action="store_true", help="Far-range detector")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    person_dir = os.path.join(args.out, args.id)
    ensure_dir(person_dir)

    # --- Open camera with backend fallback ---
    cap, backend_used = try_open_cam(args.cam)
    if cap is None:
        print("[HINT] Try a different index: --cam 1 or --cam 2; "
              "close other apps using the camera; or check Windows Camera privacy settings.")
        return

    # --- Mediapipe face detector ---
    mp_fd = mp.solutions.face_detection
    detector = mp_fd.FaceDetection(1 if args.far else 0, args.conf)

    capturing = False
    saved, frames = 0, 0
    tip = "SPACE: toggle capture | q/ESC: quit"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Empty frame. Camera busy or disconnected?")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        h, w = frame.shape[:2]
        boxes = []
        if result.detections:
            for det in result.detections:
                bb = det.location_data.relative_bounding_box
                x, y, bw, bh = bb.xmin, bb.ymin, bb.width, bb.height
                X = max(0, int((x - 0.08)*w)); Y = max(0, int((y - 0.08)*h))
                W = min(w - X, int((bw + 0.16)*w)); H = min(h - Y, int((bh + 0.16)*h))
                boxes.append((X,Y,W,H))
                cv2.rectangle(frame,(X,Y),(X+W,Y+H),(0,255,0),2)

        # Save samples when toggled ON
        if capturing and boxes and saved < args.max and frames % max(1,args.every)==0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (X,Y,W,H) in boxes:
                roi = gray[Y:Y+H, X:X+W]
                if roi.size == 0: continue
                hh, ww = roi.shape
                s = min(hh, ww); cy, cx = hh//2, ww//2
                roi = roi[cy - s//2: cy + s//2, cx - s//2: cx + s//2]
                if roi.size == 0: continue
                roi = cv2.resize(roi, (200,200), cv2.INTER_AREA)
                roi = preprocess_gray(roi)
                name = f"{args.id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
                cv2.imwrite(os.path.join(person_dir, name), roi)
                saved += 1

        status = f"ID:{args.id}  Saved:{saved}/{args.max}  {'CAPTURE:ON' if capturing else 'CAPTURE:OFF'}  Cam:{args.cam} [{backend_used}]"
        for i, text in enumerate([status, tip], start=1):
            y = 24*i
            cv2.putText(frame, text, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Capture Samples (Auto-backend)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        if key == 32: capturing = not capturing  # SPACE toggles
        frames += 1

    cap.release(); cv2.destroyAllWindows()
    print(f"[DONE] Saved {saved} samples to {person_dir}")

if __name__ == "__main__":
    main()
