# recognize_attendance.py (robust: draw only for real matches)
import cv2, json, csv, os, time, argparse, datetime
import numpy as np
import mediapipe as mp

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def preprocess_gray(img):
    clahe = cv2.createCLAHE(2.0, (8,8))
    eq = clahe.apply(img)
    return cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX)

def load_labels(path):
    with open(path, "r") as f:
        label_map = json.load(f)
    return {v: k for k, v in label_map.items()}  # int -> str

def try_open_cam(index: int):
    tried, candidates = [], []
    for name in ("CAP_MSMF", "CAP_DSHOW"):
        v = getattr(cv2, name, None)
        if v is not None:
            candidates.append((v, name))
    candidates.append((None, "DEFAULT"))
    for backend, name in candidates:
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        ok = cap.isOpened()
        tried.append(f"{name}({'OK' if ok else 'FAIL'})")
        if ok:
            # 1080p is helpful for small faces; you can lower to 1280x720 if slow
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
    ap.add_argument("--model", default="models/lbph.yml")
    ap.add_argument("--labels", default="models/labels.json")
    ap.add_argument("--cam", type=int, default=0)
    # NOTE: keep --far OFF if you’re within ~2m; use --far for >2–5m
    ap.add_argument("--far", action="store_true", help="Use far-range detector (2–5m). Leave OFF if seated near.")
    ap.add_argument("--conf", type=float, default=0.7, help="Min detection confidence (raise to avoid background hits)")
    ap.add_argument("--session", default=None, help="Session label")
    ap.add_argument("--outdir", default="attendance")
    ap.add_argument("--stable", type=int, default=6, help="Consecutive frames to confirm an ID")
    ap.add_argument("--show_conf", action="store_true", help="Show raw LBPH confidence")
    ap.add_argument("--thresh", type=float, default=None, help="Override LBPH threshold (lower=stricter)")
    # New gates:
    ap.add_argument("--min_face", type=int, default=140, help="Skip faces smaller than this height in px")
    ap.add_argument("--min_eye_frac", type=float, default=0.25, help="Min eye distance as fraction of box width")
    ap.add_argument("--min_sharp", type=float, default=20.0, help="Min Laplacian variance (focus/texture)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    today = datetime.date.today().strftime("%Y-%m-%d")
    csv_path = os.path.join(args.outdir, f"attendance_{today}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "session", "student_id", "confidence"])

    if not (os.path.exists(args.model) and os.path.exists(args.labels)):
        print("[ERROR] Model/labels not found. Run train_lbph.py first.")
        return
    inv_labels = load_labels(args.labels)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.model)
    try:
        model_thresh = recognizer.getThreshold()
        if model_thresh <= 0: model_thresh = 70.0
    except:
        model_thresh = 70.0
    THRESH = args.thresh if args.thresh is not None else model_thresh
    print(f"[INFO] Using LBPH threshold = {THRESH:.1f}")

    cap, backend_used = try_open_cam(args.cam)
    if cap is None: return

    mp_fd = mp.solutions.face_detection
    detector = mp_fd.FaceDetection(1 if args.far else 0, args.conf)

    session_label = args.session or "Session"
    marked = set()
    stable_counts = {}

    def log_attendance(student_id, conf):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ts, session_label, student_id, f"{conf:.2f}"])
        print(f"[MARKED] {student_id} @ {ts}  conf={conf:.2f}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Empty frame. Camera busy or disconnected?")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        h, w = frame.shape[:2]

        if result.detections:
            for det in result.detections:
                # Bounding box
                bb = det.location_data.relative_bounding_box
                x, y, bw, bh = bb.xmin, bb.ymin, bb.width, bb.height
                X = max(0, int((x - 0.08) * w))
                Y = max(0, int((y - 0.08) * h))
                W = min(w - X, int((bw + 0.16) * w))
                H = min(h - Y, int((bh + 0.16) * h))
                if W <= 0 or H <= 0: 
                    continue

                # ---- GATE 1: minimum face size ----
                if H < args.min_face:
                    continue

                # ---- GATE 2: eyes must be inside box & far enough apart ----
                kps = det.location_data.relative_keypoints
                if not kps or len(kps) < 2:
                    continue
                # MediaPipe order: RIGHT_EYE=0, LEFT_EYE=1, ...
                re = (int(kps[0].x * w), int(kps[0].y * h))
                le = (int(kps[1].x * w), int(kps[1].y * h))
                # inside?
                def inside(pt):
                    return X <= pt[0] <= X+W and Y <= pt[1] <= Y+H
                if not (inside(re) and inside(le)):
                    continue
                eye_dist = abs(le[0] - re[0])
                if eye_dist < args.min_eye_frac * W:
                    continue

                # ---- GATE 3: sharpness (reject flat/mirror patches) ----
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = gray_full[Y:Y+H, X:X+W]
                if roi.size == 0:
                    continue
                focus = cv2.Laplacian(roi, cv2.CV_64F).var()
                if focus < args.min_sharp:
                    continue

                # Center-square -> 200x200 -> preprocess -> predict
                hh, ww = roi.shape
                s = min(hh, ww); cy, cx = hh//2, ww//2
                roi_sq = roi[cy - s//2: cy + s//2, cx - s//2: cx + s//2]
                if roi_sq.size == 0:
                    continue
                roi_200 = cv2.resize(roi_sq, (200, 200), cv2.INTER_AREA)
                roi_200 = preprocess_gray(roi_200)

                pred_label, conf = recognizer.predict(roi_200)
                conf_val = float(conf)
                label_text = inv_labels.get(pred_label, "Unknown")
                good = (label_text != "Unknown") and (conf_val <= THRESH)

                if good:
                    # Draw ONLY for recognized faces
                    cv2.rectangle(frame, (X, Y), (X+W, Y+H), (0, 255, 0), 2)
                    tag = f"{label_text}" + (f" ({conf_val:.1f})" if args.show_conf else "")
                    cv2.putText(frame, tag, (X, max(22, Y-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(frame, tag, (X, max(22, Y-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                    cnt = stable_counts.get(label_text, 0) + 1
                    stable_counts[label_text] = cnt
                    if cnt >= args.stable and label_text not in marked:
                        marked.add(label_text)
                        log_attendance(label_text, conf_val)
                else:
                    # decay counters slightly
                    for k in list(stable_counts.keys()):
                        stable_counts[k] = max(0, stable_counts[k] - 1)

        hud = f"Session:{session_label}  Marked:{len(marked)}  Cam:{args.cam}"
        cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Recognize + Attendance (LBPH)", frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Attendance written to: {csv_path}")

if __name__ == "__main__":
    main()
