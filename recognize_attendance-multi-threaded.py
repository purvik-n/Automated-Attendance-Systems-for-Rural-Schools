# recognize_attendance.py — v3 (multithreaded, distance-friendly)
import cv2, json, csv, os, time, argparse, datetime, threading, queue
import numpy as np
import mediapipe as mp

# ---------------- Utils ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def preprocess_gray(img):
    clahe = cv2.createCLAHE(2.0, (8,8))
    eq = clahe.apply(img)
    return cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX)

def load_labels(path):
    with open(path, "r") as f:
        label_map = json.load(f)
    return {v: k for k, v in label_map.items()}  # int -> str

def open_cam(index: int, width=1920, height=1080, fps=30):
    tried = []
    for name in ("CAP_MSMF", "CAP_DSHOW", None):  # try MSMF -> DSHOW -> default
        flag = getattr(cv2, name, None) if name else None
        cap = cv2.VideoCapture(index) if flag is None else cv2.VideoCapture(index, flag)
        ok = cap.isOpened()
        tried.append(f"{(name or 'DEFAULT')}({'OK' if ok else 'FAIL'})")
        if ok:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            print(f"[INFO] Opened camera {index} via {(name or 'DEFAULT')}")
            return cap
        else:
            cap.release()
    print("[ERROR] Could not open camera. Tried:", " -> ".join(tried))
    return None

# ---------------- Worker (processing thread) ----------------
def processor_worker(in_q, out_q, stop_ev, args, csv_path):
    # Create detector(s) & recognizer inside the worker thread
    mp_fd = mp.solutions.face_detection
    det_near = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=args.conf)
    det_far  = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=args.conf)

    inv_labels = load_labels(args.labels)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.model)
    try:
        model_thresh = recognizer.getThreshold()
        if model_thresh <= 0: model_thresh = 70.0
    except:
        model_thresh = 70.0
    THRESH = args.thresh if args.thresh is not None else model_thresh
    print(f"[INFO] LBPH threshold = {THRESH:.1f}")

    marked = set()
    stable_counts = {}

    def log_attendance(student_id, conf):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ts, args.session, student_id, f"{conf:.2f}"])
        print(f"[MARKED] {student_id} @ {ts}  conf={conf:.2f}")

    def detect_all(rgb):
        """Return list of (X,Y,W,H,kps) based on detector mode."""
        h, w = rgb.shape[:2]
        out = []
        # try near first for 'auto'; if nothing found, try far
        if args.det in ("near","auto"):
            r = det_near.process(rgb)
            if r.detections:
                for d in r.detections:
                    bb = d.location_data.relative_bounding_box
                    X = max(0, int((bb.xmin - 0.06) * w))
                    Y = max(0, int((bb.ymin - 0.06) * h))
                    W = min(w - X, int((bb.width + 0.12) * w))
                    H = min(h - Y, int((bb.height + 0.12) * h))
                    out.append((X,Y,W,H,d.location_data.relative_keypoints))
        if (args.det in ("far","auto")) and (args.det == "far" or not out):
            r = det_far.process(rgb)
            if r.detections:
                for d in r.detections:
                    bb = d.location_data.relative_bounding_box
                    X = max(0, int((bb.xmin - 0.06) * w))
                    Y = max(0, int((bb.ymin - 0.06) * h))
                    W = min(w - X, int((bb.width + 0.12) * w))
                    H = min(h - Y, int((bb.height + 0.12) * h))
                    out.append((X,Y,W,H,d.location_data.relative_keypoints))
        return out

    while not stop_ev.is_set():
        try:
            frame = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_all(rgb)

        # distance-friendly gates (relative to frame height/box width)
        min_face_px = int(args.min_face_frac * h)

        # We’ll draw on a copy to keep main thread simple
        vis = frame.copy()

        for (X,Y,W,H,kps) in faces:
            # Optional debug: show yellow box for detections
            if args.debug_detect:
                cv2.rectangle(vis, (X,Y), (X+W,Y+H), (0,255,255), 1)

            if W <= 0 or H <= 0:
                continue
            if H < min_face_px:
                continue

            if not kps or len(kps) < 2:
                continue
            re = (int(kps[0].x * w), int(kps[0].y * h))  # right eye
            le = (int(kps[1].x * w), int(kps[1].y * h))  # left eye
            def inside(pt): return X <= pt[0] <= X+W and Y <= pt[1] <= Y+H
            if not (inside(re) and inside(le)):
                continue
            if abs(le[0]-re[0]) < args.min_eye_frac * W:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[Y:Y+H, X:X+W]
            if roi.size == 0:
                continue

            # Sharpness gate (reject smooth patches)
            if cv2.Laplacian(roi, cv2.CV_64F).var() < args.min_sharp:
                continue

            # Normalize crop
            hh, ww = roi.shape
            s = min(hh, ww); cy, cx = hh//2, ww//2
            roi_sq = roi[cy - s//2: cy + s//2, cx - s//2: cx + s//2]
            if roi_sq.size == 0:
                continue
            roi_200 = cv2.resize(roi_sq, (200,200), cv2.INTER_AREA)
            roi_200 = preprocess_gray(roi_200)

            pred_label, conf = recognizer.predict(roi_200)
            conf_val = float(conf)
            who = inv_labels.get(pred_label, "Unknown")
            good = (who != "Unknown") and (conf_val <= THRESH)

            if good:
                cv2.rectangle(vis, (X,Y), (X+W,Y+H), (0,255,0), 2)
                tag = f"{who}" + (f" ({conf_val:.1f})" if args.show_conf else "")
                cv2.putText(vis, tag, (X, max(22, Y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, tag, (X, max(22, Y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

                stable_counts[who] = stable_counts.get(who, 0) + 1
                if stable_counts[who] >= args.stable and who not in marked:
                    marked.add(who)
                    log_attendance(who, conf_val)
            else:
                # Light decay to avoid sticky counts
                for k in list(stable_counts.keys()):
                    stable_counts[k] = max(0, stable_counts[k]-1)

        # HUD (drawn in worker so display thread is dumb & fast)
        hud1 = f"Session:{args.session}  Marked:{len(marked)}  det:{args.det}"
        hud2 = f"conf:{args.conf:.2f} min_face:{int(args.min_face_frac*100)}% sharp:{args.min_sharp:.1f} eye:{args.min_eye_frac:.2f} TH:{THRESH:.1f}"
        for i, text in enumerate([hud1, hud2], start=1):
            y = 24*i
            cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # Push latest visualization, dropping stale frames so UI stays realtime
        try:
            while out_q.qsize() >= out_q.maxsize:
                out_q.get_nowait()
        except queue.Empty:
            pass
        out_q.put(vis)

# ---------------- Main (UI + capture thread) ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/lbph.yml")
    ap.add_argument("--labels", default="models/labels.json")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)

    # Detector & gates (distance-friendly defaults)
    ap.add_argument("--det", choices=["auto","near","far"], default="auto")
    ap.add_argument("--conf", type=float, default=0.35, help="Min detection confidence")
    ap.add_argument("--min_face_frac", type=float, default=0.08, help="Min face height as fraction of frame height")
    ap.add_argument("--min_eye_frac", type=float, default=0.18, help="Min eye dist as fraction of box width")
    ap.add_argument("--min_sharp", type=float, default=8.0, help="Min Laplacian variance (focus)")
    ap.add_argument("--debug_detect", action="store_true", help="Show yellow boxes for detections")

    # Recognition / attendance
    ap.add_argument("--thresh", type=float, default=None, help="Override LBPH threshold (lower=stricter)")
    ap.add_argument("--stable", type=int, default=5, help="Consecutive frames to confirm an ID")
    ap.add_argument("--show_conf", action="store_true", help="Show LBPH confidence on-screen")
    ap.add_argument("--session", default="Session", help="Session label")
    ap.add_argument("--outdir", default="attendance")
    args = ap.parse_args()

    # CSV
    ensure_dir(args.outdir)
    today = datetime.date.today().strftime("%Y-%m-%d")
    csv_path = os.path.join(args.outdir, f"attendance_{today}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "session", "student_id", "confidence"])

    # Camera (main thread)
    cap = open_cam(args.cam, args.width, args.height, args.fps)
    if cap is None:
        return

    # Queues & threads
    in_q  = queue.Queue(maxsize=2)   # backpressure: keep only newest frames
    out_q = queue.Queue(maxsize=2)
    stop_ev = threading.Event()

    # Capture thread (producer)
    def capture_loop():
        while not stop_ev.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue
            # Drop stale input frames
            try:
                while in_q.qsize() >= in_q.maxsize:
                    in_q.get_nowait()
            except queue.Empty:
                pass
            in_q.put(frame)

    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_proc = threading.Thread(target=processor_worker, args=(in_q, out_q, stop_ev, args, csv_path), daemon=True)

    t_cap.start()
    t_proc.start()

    # Display loop (consumer)
    cv2.namedWindow("Recognize + Attendance (LBPH) — v3", cv2.WINDOW_NORMAL)
    last_frame_time = time.time()

    try:
        while True:
            try:
                vis = out_q.get(timeout=0.1)
            except queue.Empty:
                # no processed frame yet; keep window responsive
                if (time.time() - last_frame_time) > 5 and stop_ev.is_set():
                    break
                # optional: show blank or last frame – skip
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                continue

            last_frame_time = time.time()
            cv2.imshow("Recognize + Attendance (LBPH) — v3", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
    finally:
        stop_ev.set()
        t_cap.join(timeout=1.0)
        t_proc.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        print(f"[DONE] Attendance written to: {csv_path}")

if __name__ == "__main__":
    main()
