# capture-frontend-v2.py
# Automated Rural Attendance monitoring system
# Unified App (Capture + Train + Recognize)
# Fixes: model/labels integrity check, eye-aligned recognition, prototype gating to reject unknowns
# Layout: 65% video / 35% controls, multithreaded, per-student counters
# Requires:
#   pip install PySide6 opencv-contrib-python mediapipe numpy

from __future__ import annotations
import os, sys, time, uuid, queue, json, csv, datetime, math
import cv2
import numpy as np
import mediapipe as mp

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot

APP_VERSION = "v3.4 (integrity+alignment+prototypes)"

# ----------------------- helpers -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    """CLAHE + normalize for lighting robustness. Input must be grayscale 200x200."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    return cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX)

def rotate_roi_by_eyes(roi_gray: np.ndarray, re: tuple[int,int], le: tuple[int,int]) -> np.ndarray:
    """
    Rotate ROI (grayscale) so eyes are horizontal.
    re/le are eye coords relative to ROI (x,y).
    Returns an aligned ROI, same size as input (replicate border).
    """
    (rx, ry), (lx, ly) = re, le
    dx, dy = (lx - rx), (ly - ry)
    angle = math.degrees(math.atan2(dy, dx))  # +ve ccw
    (h, w) = roi_gray.shape[:2]
    cx, cy = (rx + lx) / 2.0, (ry + ly) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    aligned = cv2.warpAffine(roi_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned

# ----------------------- camera (producer) -----------------------
class FrameGrabber(QtCore.QThread):
    error = Signal(str)
    info = Signal(str)

    def __init__(self, cam_index: int, width: int, height: int, fps_req: int, out_q: queue.Queue, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps_req = fps_req
        self.out_q = out_q
        self._stop = False
        self._backend_used = ""
        self._cap = None

    def stop(self):
        self._stop = True

    def _open_cam(self):
        tried = []
        for name in ("CAP_MSMF", "CAP_DSHOW", None):
            flag = getattr(cv2, name, None) if name else None
            cap = cv2.VideoCapture(self.cam_index) if flag is None else cv2.VideoCapture(self.cam_index, flag)
            ok = cap.isOpened()
            tried.append(f"{(name or 'DEFAULT')}({'OK' if ok else 'FAIL'})")
            if ok:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps_req)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self._backend_used = (name or 'DEFAULT')
                self.info.emit(f"[INFO] Camera {self.cam_index} via {self._backend_used}")
                return cap
            else:
                cap.release()
        self.error.emit("Could not open camera. Tried: " + " -> ".join(tried))
        return None

    def run(self):
        self._cap = self._open_cam()
        if self._cap is None:
            return
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            # Push newest frame; drop stale
            try:
                while self.out_q.qsize() >= self.out_q.maxsize:
                    self.out_q.get_nowait()
            except queue.Empty:
                pass
            self.out_q.put(frame)
        if self._cap:
            self._cap.release()

# ----------------------- CAPTURE (consumer) -----------------------
class CaptureProcessor(QtCore.QThread):
    frameReady = Signal(QtGui.QImage)
    statsReady = Signal(str)
    savedCountChanged = Signal(int)
    error = Signal(str)

    def __init__(self,
                 in_q: queue.Queue,
                 student_id: str,
                 out_root: str,
                 far_model: bool,
                 det_conf: float,
                 min_face_px: int,
                 save_every: int,
                 max_images: int,
                 det_downscale: float = 0.75,
                 cold_start_frames: int = 20,
                 parent=None):
        super().__init__(parent)
        self.in_q = in_q
        self.student_id = student_id
        self.out_root = out_root
        self.far_model = far_model
        self.det_conf = float(det_conf)
        self.min_face_px = int(min_face_px)
        self.save_every = max(1, int(save_every))
        self.max_images = max(1, int(max_images))
        self.det_downscale = float(det_downscale)
        self.cold_start_frames = int(cold_start_frames)

        self._stop = False
        self._capturing = False
        self._saved = 0

        self._fps_tick = time.time()
        self._fps_frames = 0
        self._detector = None

    @Slot(bool)
    def set_capturing(self, on: bool):
        self._capturing = bool(on)

    @Slot(float)
    def set_conf(self, conf: float):
        self.det_conf = float(conf)
        self._detector = None

    @Slot(bool)
    def set_far(self, far: bool):
        self.far_model = bool(far)
        self._detector = None

    @Slot(int)
    def set_min_face(self, px: int):
        self.min_face_px = int(px)

    @Slot(float)
    def set_downscale(self, f: float):
        self.det_downscale = max(0.4, min(1.0, float(f)))

    def stop(self):
        self._stop = True

    def _create_detector(self):
        try:
            mp_fd = mp.solutions.face_detection
            self._detector = mp_fd.FaceDetection(
                model_selection=1 if self.far_model else 0,
                min_detection_confidence=self.det_conf
            )
        except Exception as e:
            self.error.emit(f"Failed to create detector: {e}")
            self._detector = None

    def run(self):
        person_dir = os.path.join(self.out_root, self.student_id) if self.student_id else None
        if self.student_id:
            ensure_dir(person_dir)
        frames_seen = 0
        while not self._stop:
            try:
                frame = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            frames_seen += 1

            # Warm-up
            if self._detector is None and frames_seen <= self.cold_start_frames:
                vis_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(vis_rgb.data, vis_rgb.shape[1], vis_rgb.shape[0], vis_rgb.strides[0], QtGui.QImage.Format_RGB888)
                self.frameReady.emit(qimg.copy())
                if frames_seen == self.cold_start_frames:
                    self._create_detector()
                continue

            if self._detector is None:
                self._create_detector()
                vis_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(vis_rgb.data, vis_rgb.shape[1], vis_rgb.shape[0], vis_rgb.strides[0], QtGui.QImage.Format_RGB888)
                self.frameReady.emit(qimg.copy())
                continue

            # Detection
            h, w = frame.shape[:2]
            scale = self.det_downscale
            if scale < 1.0:
                small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = self._detector.process(rgb)

            boxes = []
            if result and result.detections:
                H, W = frame.shape[:2]
                for det in result.detections:
                    bb = det.location_data.relative_bounding_box
                    x, y, bw, bh = bb.xmin, bb.ymin, bb.width, bb.height
                    X = max(0, int((x - 0.08) * W))
                    Y = max(0, int((y - 0.08) * H))
                    WW = min(W - X, int((bw + 0.16) * W))
                    HH = min(H - Y, int((bh + 0.16) * H))
                    if WW > 0 and HH > 0 and HH >= self.min_face_px:
                        boxes.append((X, Y, WW, HH))

            vis = frame.copy()
            for (X, Y, WW, HH) in boxes:
                cv2.rectangle(vis, (X, Y), (X + WW, Y + HH), (0, 255, 0), 2)

            # Save logic
            if self._capturing and boxes and self._saved < self.max_images:
                if frames_seen % self.save_every == 0 and self.student_id:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for (X, Y, WW, HH) in boxes:
                        roi = gray[Y:Y + HH, X:X + WW]
                        if roi.size == 0: continue
                        hh, ww = roi.shape
                        s = min(hh, ww); cy, cx = hh // 2, ww // 2
                        roi = roi[cy - s // 2: cy + s // 2, cx - s // 2: cx + s // 2]
                        if roi.size == 0: continue
                        roi = cv2.resize(roi, (200, 200), cv2.INTER_AREA)
                        roi = preprocess_gray(roi)
                        fname = f"{self.student_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
                        try:
                            cv2.imwrite(os.path.join(person_dir, fname), roi)
                            self._saved += 1
                            self.savedCountChanged.emit(self._saved)
                        except Exception as e:
                            self.error.emit(f"Failed to save: {e}")
                        if self._saved >= self.max_images:
                            break

            # HUD
            self._fps_frames += 1
            if self._fps_frames >= 15:
                now = time.time(); dt = now - self._fps_tick
                fps = self._fps_frames / dt if dt > 0 else 0.0
                self._fps_tick = now; self._fps_frames = 0
                hud = (f"Det:{'FAR' if self.far_model else 'NEAR'} conf:{self.det_conf:.2f}  "
                       f"minFace:{self.min_face_px}px  Saved:{self._saved}/{self.max_images}  FPS:{fps:.1f}  scale:{scale:.2f}")
                self.statsReady.emit(hud)

            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(vis_rgb.data, vis_rgb.shape[1], vis_rgb.shape[0], vis_rgb.strides[0], QtGui.QImage.Format_RGB888)
            self.frameReady.emit(qimg.copy())

# ----------------------- CAPTURE TAB -----------------------
class CaptureTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CaptureTab")
        self.grabber: FrameGrabber | None = None
        self.proc: CaptureProcessor | None = None
        self.q_frames: queue.Queue = queue.Queue(maxsize=2)

        # Left: video
        self.videoLabel = QtWidgets.QLabel("Preview")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.videoLabel.setMinimumSize(960, 540)
        self.videoLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Right: controls
        self.studentEdit = QtWidgets.QLineEdit(); self.studentEdit.setPlaceholderText("Student ID (e.g., 1001)")
        self.camIndex = QtWidgets.QSpinBox(); self.camIndex.setRange(0,8); self.camIndex.setValue(0)
        self.widthBox  = QtWidgets.QComboBox(); self.widthBox.addItems(["1280","1920"]); self.widthBox.setCurrentText("1280")
        self.heightBox = QtWidgets.QComboBox(); self.heightBox.addItems(["720","1080"]); self.heightBox.setCurrentText("720")
        self.fpsBox    = QtWidgets.QSpinBox(); self.fpsBox.setRange(5,60); self.fpsBox.setValue(30)
        self.farCheck  = QtWidgets.QCheckBox("FAR model (2–5 m)")
        self.confSlider= QtWidgets.QSlider(Qt.Horizontal); self.confSlider.setRange(10,95); self.confSlider.setValue(40)
        self.confLabel = QtWidgets.QLabel("0.40")
        self.minFace   = QtWidgets.QSpinBox(); self.minFace.setRange(40,400); self.minFace.setValue(120)
        self.saveEvery = QtWidgets.QSpinBox(); self.saveEvery.setRange(1,60); self.saveEvery.setValue(5)
        self.maxImages = QtWidgets.QSpinBox(); self.maxImages.setRange(1,2000); self.maxImages.setValue(150)
        self.outRoot   = QtWidgets.QLineEdit("data/faces")
        self.scaleBox  = QtWidgets.QComboBox(); self.scaleBox.addItems(["1.0","0.75","0.5"]); self.scaleBox.setCurrentText("0.75")

        self.startBtn  = QtWidgets.QPushButton("Start Preview")
        self.stopBtn   = QtWidgets.QPushButton("Stop Preview")
        self.captureBtn= QtWidgets.QPushButton("Start Capture"); self.captureBtn.setCheckable(True)
        self.savedLabel= QtWidgets.QLabel("Saved: 0")
        self.statusBar = QtWidgets.QLabel(); self.statusBar.setStyleSheet("color:#888;")

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(6,6,6,6); form.setVerticalSpacing(4)
        form.addRow("Student ID:", self.studentEdit)
        form.addRow("Camera index:", self.camIndex)
        wh = QtWidgets.QHBoxLayout(); wh.addWidget(self.widthBox); wh.addWidget(QtWidgets.QLabel("×")); wh.addWidget(self.heightBox); wh.addWidget(QtWidgets.QLabel(" @ ")); wh.addWidget(self.fpsBox)
        form.addRow("Resolution:", self._wrap(wh))
        form.addRow(self.farCheck)
        cr = QtWidgets.QHBoxLayout(); cr.addWidget(self.confSlider); cr.addWidget(self.confLabel); form.addRow("Detector conf:", self._wrap(cr))
        form.addRow("Min face (px):", self.minFace)
        form.addRow("Save every N frames:", self.saveEvery)
        form.addRow("Max images:", self.maxImages)
        form.addRow("Output root:", self.outRoot)
        form.addRow("Detection scale:", self.scaleBox)
        form.addRow(self.startBtn); form.addRow(self.stopBtn); form.addRow(self.captureBtn)
        form.addRow(self.savedLabel); form.addRow(self.statusBar)

        rightPanel = QtWidgets.QWidget(); rightPanel.setLayout(form)
        rightScroll = QtWidgets.QScrollArea(); rightScroll.setWidgetResizable(True); rightScroll.setWidget(rightPanel)
        rightScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.videoLabel)
        self.splitter.addWidget(rightScroll)
        self.splitter.setStretchFactor(0, 65)
        self.splitter.setStretchFactor(1, 35)

        layout = QtWidgets.QHBoxLayout(self); layout.addWidget(self.splitter)
        QtCore.QTimer.singleShot(0, self._apply_split_sizes)

        self.startBtn.clicked.connect(self.start_preview)
        self.stopBtn.clicked.connect(self.stop_preview)
        self.captureBtn.toggled.connect(self.toggle_capture)
        self.confSlider.valueChanged.connect(self._conf_changed)
        self._update_buttons(False)

    def _apply_split_sizes(self):
        total = max(1, self.splitter.width())
        self.splitter.setSizes([int(total*0.65), int(total*0.35)])

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        QtCore.QTimer.singleShot(0, self._apply_split_sizes)
        return super().resizeEvent(e)

    def _wrap(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); w.setLayout(layout); return w

    def _conf_changed(self, val: int):
        conf = val / 100.0
        self.confLabel.setText(f"{conf:.2f}")
        if self.proc: self.proc.set_conf(conf)

    def _update_buttons(self, running: bool):
        self.startBtn.setEnabled(not running)
        self.stopBtn.setEnabled(running)
        self.captureBtn.setEnabled(running)

    @Slot()
    def start_preview(self):
        if self.grabber or self.proc: return
        cam = int(self.camIndex.value())
        width = int(self.widthBox.currentText()); height = int(self.heightBox.currentText()); fps = int(self.fpsBox.value())
        far = bool(self.farCheck.isChecked()); conf = self.confSlider.value()/100.0
        min_face_px = int(self.minFace.value()); every = int(self.saveEvery.value()); max_images = int(self.maxImages.value())
        out_root = self.outRoot.text().strip() or "data/faces"; student_id = self.studentEdit.text().strip()
        if not os.path.isdir(out_root): ensure_dir(out_root)

        self.grabber = FrameGrabber(cam, width, height, fps, self.q_frames)
        det_scale = float(self.scaleBox.currentText())
        self.proc = CaptureProcessor(self.q_frames, student_id, out_root, far, conf, min_face_px, every, max_images,
                                     det_downscale=det_scale, cold_start_frames=20)

        self.grabber.error.connect(self.on_error); self.grabber.info.connect(self.on_status)
        self.proc.frameReady.connect(self.on_frame); self.proc.statsReady.connect(self.on_status)
        self.proc.savedCountChanged.connect(self.on_saved_count); self.proc.error.connect(self.on_error)

        self.grabber.start(); self.proc.start()
        self._update_buttons(True); self.statusBar.setText("Starting preview…"); self.savedLabel.setText("Saved: 0")

    @Slot()
    def stop_preview(self):
        if self.proc: self.proc.stop(); self.proc.wait(1000); self.proc=None
        if self.grabber: self.grabber.stop(); self.grabber.wait(1000); self.grabber=None
        try:
            while self.q_frames.qsize(): self.q_frames.get_nowait()
        except Exception: pass
        self._update_buttons(False); self.statusBar.setText("Stopped.")

    @Slot(bool)
    def toggle_capture(self, checked: bool):
        if not self.proc:
            self.captureBtn.setChecked(False); return
        if checked and not self.studentEdit.text().strip():
            QtWidgets.QMessageBox.warning(self, "Missing Student ID", "Please enter a Student ID before capturing.")
            self.captureBtn.setChecked(False); return
        self.proc.set_capturing(checked)
        self.captureBtn.setText("Stop Capture" if checked else "Start Capture")

    @Slot(QtGui.QImage)
    def on_frame(self, qimg: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pix.scaled(self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(str)
    def on_status(self, text: str):
        self.statusBar.setText(text)

    @Slot(int)
    def on_saved_count(self, n: int):
        self.savedLabel.setText(f"Saved: {n}")

    @Slot(str)
    def on_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Error", msg)

# ----------------------- TRAIN (worker + tab) -----------------------
class TrainWorker(QtCore.QThread):
    progress = Signal(str)
    done = Signal(str, str, int)  # model_path, labels_path, num_images
    error = Signal(str)

    def __init__(self, data_root: str, model_path: str, labels_path: str,
                 radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=60,
                 parent=None):
        super().__init__(parent)
        self.data_root = data_root
        self.model_path = model_path
        self.labels_path = labels_path
        self.radius = int(radius)
        self.neighbors = int(neighbors)
        self.grid_x = int(grid_x)
        self.grid_y = int(grid_y)
        self.threshold = float(threshold)  # default a bit stricter than 70

    def run(self):
        try:
            if not os.path.isdir(self.data_root):
                raise RuntimeError(f"Data root not found: {self.data_root}")

            label_to_int: dict[str,int] = {}
            images: list[np.ndarray] = []
            labels: list[int] = []
            imgs_per_class: dict[int, list[np.ndarray]] = {}
            total = 0

            # Load & preprocess
            for name in sorted(os.listdir(self.data_root)):
                person_dir = os.path.join(self.data_root, name)
                if not os.path.isdir(person_dir):
                    continue
                if name not in label_to_int:
                    label_to_int[name] = len(label_to_int)
                lid = label_to_int[name]

                count = 0
                for fn in os.listdir(person_dir):
                    if not fn.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                        continue
                    p = os.path.join(person_dir, fn)
                    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    if img.shape != (200,200):
                        img = cv2.resize(img, (200,200), interpolation=cv2.INTER_AREA)
                    img = preprocess_gray(img)
                    images.append(img)
                    labels.append(lid)
                    imgs_per_class.setdefault(lid, []).append(img)
                    count += 1
                    total += 1
                self.progress.emit(f"Loaded {count} images for {name}")

            if total == 0:
                raise RuntimeError("No images found. Use Capture tab first.")

            # Train LBPH
            self.progress.emit(f"Training LBPH on {total} images …")
            rec = cv2.face.LBPHFaceRecognizer_create(
                radius=self.radius, neighbors=self.neighbors,
                grid_x=self.grid_x, grid_y=self.grid_y
            )
            rec.setThreshold(self.threshold)
            rec.train(images, np.array(labels))

            # Save model + labels
            ensure_dir(os.path.dirname(self.model_path) or ".")
            ensure_dir(os.path.dirname(self.labels_path) or ".")
            rec.write(self.model_path)
            with open(self.labels_path, "w") as f:
                json.dump(label_to_int, f, indent=2)

            # Build prototypes (per-class mean and RMSE 95p gate)
            num_labels = len(label_to_int)
            means = np.zeros((num_labels, 200, 200), dtype=np.float32)
            thr = np.zeros((num_labels,), dtype=np.float32)
            for lid, arr in imgs_per_class.items():
                A = np.stack(arr).astype(np.float32)
                mu = A.mean(axis=0)
                means[lid] = mu
                rmse = np.sqrt(((A - mu) ** 2).reshape(A.shape[0], -1).mean(axis=1))
                thr[lid] = np.percentile(rmse, 95) + 5.0  # margin
            base = os.path.splitext(self.model_path)[0]
            proto_path = base + "_proto.npz"
            np.savez(proto_path, means=means.astype(np.uint8), thr=thr.astype(np.float32))

            # Meta file: integrity guard
            meta = {
                "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "labels": label_to_int,
                "lbph": dict(radius=self.radius, neighbors=self.neighbors, grid_x=self.grid_x, grid_y=self.grid_y, threshold=self.threshold),
                "prototypes": os.path.basename(proto_path)
            }
            with open(base + "_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            self.done.emit(self.model_path, self.labels_path, total)
        except Exception as e:
            self.error.emit(str(e))

class TrainTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: TrainWorker | None = None

        self.dataRoot = QtWidgets.QLineEdit("data/faces")
        self.btnBrowseData = QtWidgets.QPushButton("Browse…")
        self.modelPath = QtWidgets.QLineEdit("models/lbph.yml")
        self.btnBrowseModel = QtWidgets.QPushButton("…")
        self.labelsPath = QtWidgets.QLineEdit("models/labels.json")
        self.btnBrowseLabels = QtWidgets.QPushButton("…")

        self.radius = QtWidgets.QSpinBox(); self.radius.setRange(1, 8); self.radius.setValue(1)
        self.neighbors = QtWidgets.QSpinBox(); self.neighbors.setRange(1, 32); self.neighbors.setValue(8)
        self.gridx = QtWidgets.QSpinBox(); self.gridx.setRange(1, 16); self.gridx.setValue(8)
        self.gridy = QtWidgets.QSpinBox(); self.gridy.setRange(1, 16); self.gridy.setValue(8)
        self.thresh = QtWidgets.QDoubleSpinBox(); self.thresh.setRange(1.0, 200.0); self.thresh.setValue(60.0); self.thresh.setDecimals(1)

        self.btnScan = QtWidgets.QPushButton("Scan Dataset")
        self.btnTrain = QtWidgets.QPushButton("Train LBPH")
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Data root:"), 0, 0); grid.addWidget(self.dataRoot, 0, 1); grid.addWidget(self.btnBrowseData, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Model path:"), 1, 0); grid.addWidget(self.modelPath, 1, 1); grid.addWidget(self.btnBrowseModel, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Labels path:"), 2, 0); grid.addWidget(self.labelsPath, 2, 1); grid.addWidget(self.btnBrowseLabels, 2, 2)

        h1 = QtWidgets.QHBoxLayout()
        for lab, w in [("radius", self.radius), ("neighbors", self.neighbors), ("grid_x", self.gridx), ("grid_y", self.gridy), ("threshold", self.thresh)]:
            h1.addWidget(QtWidgets.QLabel(lab)); h1.addWidget(w)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(grid); v.addLayout(h1); v.addWidget(self.btnScan); v.addWidget(self.btnTrain); v.addWidget(QtWidgets.QLabel("Output:")); v.addWidget(self.log)

        self.btnBrowseData.clicked.connect(self._pick_data)
        self.btnBrowseModel.clicked.connect(self._pick_model)
        self.btnBrowseLabels.clicked.connect(self._pick_labels)
        self.btnScan.clicked.connect(self._scan_dataset)
        self.btnTrain.clicked.connect(self._start_train)

    def _pick_data(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select data root", self.dataRoot.text())
        if d: self.dataRoot.setText(d)

    def _pick_model(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save model as", self.modelPath.text(), "LBPH (*.yml)")
        if p: self.modelPath.setText(p)

    def _pick_labels(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save labels as", self.labelsPath.text(), "JSON (*.json)")
        if p: self.labelsPath.setText(p)

    def _scan_dataset(self):
        root = self.dataRoot.text().strip()
        if not os.path.isdir(root):
            QtWidgets.QMessageBox.warning(self, "Missing", f"Folder not found: {root}"); return
        counts = []; total = 0
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if not os.path.isdir(p): continue
            c = sum(1 for fn in os.listdir(p) if fn.lower().endswith((".png",".jpg",".jpeg",".bmp")))
            counts.append((name, c)); total += c
        summary = "Dataset summary:\n" + "\n".join([f"  {n}: {c}" for n, c in counts]) + f"\nTOTAL: {total}\n"
        self.log.appendPlainText(summary)

    def _start_train(self):
        if self.worker:
            QtWidgets.QMessageBox.information(self, "Training", "Training already running…"); return
        root = self.dataRoot.text().strip(); model = self.modelPath.text().strip(); labels = self.labelsPath.text().strip()
        if not os.path.isdir(root):
            QtWidgets.QMessageBox.warning(self, "Missing", f"Folder not found: {root}"); return
        self.worker = TrainWorker(root, model, labels,
                                  radius=self.radius.value(), neighbors=self.neighbors.value(),
                                  grid_x=self.gridx.value(), grid_y=self.gridy.value(), threshold=self.thresh.value())
        self.worker.progress.connect(lambda s: self.log.appendPlainText(s))
        self.worker.done.connect(self._train_done); self.worker.error.connect(self._train_err)
        self.log.appendPlainText("Starting training…"); self.worker.start()

    def _train_done(self, model_path: str, labels_path: str, total: int):
        base = os.path.splitext(model_path)[0]
        self.log.appendPlainText(f"DONE: {total} images → {model_path} + {labels_path} + {base+'_proto.npz'} + {base+'_meta.json'}")
        self.worker = None

    def _train_err(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Training failed", msg); self.worker = None

# ----------------------- RECOGNIZE (worker + tab) -----------------------
class RecognizeProcessor(QtCore.QThread):
    frameReady = Signal(QtGui.QImage)
    statsReady = Signal(str)
    attendanceMarked = Signal(str, float, str)
    recognizedHit = Signal(str, int)
    error = Signal(str)

    def __init__(self, in_q: queue.Queue,
                 model_path: str, labels_path: str,
                 det_mode: str, det_conf: float, det_scale: float,
                 min_face_px: int, min_eye_frac: float, min_sharp: float,
                 threshold: float, stable: int, show_conf: bool, debug_detect: bool,
                 session: str, outdir: str,
                 parent=None):
        super().__init__(parent)
        self.in_q = in_q
        self.model_path = model_path
        self.labels_path = labels_path
        self.det_mode = det_mode
        self.det_conf = float(det_conf)
        self.det_scale = float(det_scale)
        self.min_face_px = int(min_face_px)
        self.min_eye_frac = float(min_eye_frac)
        self.min_sharp = float(min_sharp)
        self.threshold = float(threshold)
        self.stable = int(stable)
        self.show_conf = bool(show_conf)
        self.debug_detect = bool(debug_detect)
        self.session = session
        self.outdir = outdir

        self._stop = False
        self.hit_counts: dict[str,int] = {}
        self.last_hit: dict[str,float] = {}
        self.hit_cooldown = 1.0

        # tracking for stability per detection (prevents A/B flipping reaching stable)
        self.tracks = []  # list of dict(center=(x,y), label, count, marked)
        self.track_dist_thr = 0.25  # fraction of box size

    @Slot()
    def reset_counts(self):
        self.hit_counts.clear(); self.last_hit.clear()

    def stop(self):
        self._stop = True

    def _register_hit(self, who: str):
        now = time.time(); last = self.last_hit.get(who, 0.0)
        if (now - last) >= self.hit_cooldown:
            self.last_hit[who] = now
            self.hit_counts[who] = self.hit_counts.get(who, 0) + 1
            self.recognizedHit.emit(who, self.hit_counts[who])

    def run(self):
        try:
            # Load labels
            if not (os.path.exists(self.model_path) and os.path.exists(self.labels_path)):
                raise RuntimeError("Model/labels not found. Train first.")
            with open(self.labels_path, "r") as f:
                label_map = json.load(f)  # name -> int
            inv_labels = {v:k for k,v in label_map.items()}

            # Integrity check with meta (if present)
            base = os.path.splitext(self.model_path)[0]
            meta_path = base + "_meta.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if meta.get("labels") != label_map:
                        raise RuntimeError("Model/labels mismatch: lbph.yml and labels.json were not trained together. Re-train or pick matching pair.")
                except Exception:
                    raise

            # Load prototypes (optional but recommended)
            proto_path = base + "_proto.npz"
            proto_means = None; proto_thr = None
            if os.path.exists(proto_path):
                npz = np.load(proto_path)
                proto_means = npz["means"].astype(np.float32)
                proto_thr = npz["thr"].astype(np.float32)

            # LBPH model
            rec = cv2.face.LBPHFaceRecognizer_create()
            rec.read(self.model_path)
            try:
                model_thr = rec.getThreshold()
                if model_thr <= 0: model_thr = self.threshold
            except:
                model_thr = self.threshold

            # Detectors
            mp_fd = mp.solutions.face_detection
            det_near = mp_fd.FaceDetection(0, self.det_conf)
            det_far  = mp_fd.FaceDetection(1, self.det_conf)

            ensure_dir(self.outdir)
            today = datetime.date.today().strftime("%Y-%m-%d")
            csv_path = os.path.join(self.outdir, f"attendance_{today}.csv")
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(["timestamp","session","student_id","confidence"])

            fps_tick, fps_frames = time.time(), 0

            def mark(student_id: str, conf: float):
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([ts, self.session, student_id, f"{conf:.2f}"])
                self.attendanceMarked.emit(student_id, conf, ts)

            while not self._stop:
                try:
                    frame = self.in_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                H, W = frame.shape[:2]
                # Downscale for detection
                small = cv2.resize(frame, (int(W*self.det_scale), int(H*self.det_scale)), interpolation=cv2.INTER_LINEAR) if self.det_scale < 1.0 else frame
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                # Detect
                faces = []
                def add_dets(r):
                    if r and r.detections:
                        for d in r.detections:
                            bb = d.location_data.relative_bounding_box
                            X = max(0, int((bb.xmin - 0.06) * W))
                            Y = max(0, int((bb.ymin - 0.06) * H))
                            WW = min(W - X, int((bb.width + 0.12) * W))
                            HH = min(H - Y, int((bb.height + 0.12) * H))
                            faces.append((X, Y, WW, HH, d.location_data.relative_keypoints))

                if self.det_mode in ("near","auto"):
                    add_dets(det_near.process(rgb_small))
                if self.det_mode in ("far","auto") and (self.det_mode=="far" or not faces):
                    add_dets(det_far.process(rgb_small))

                vis = frame.copy()
                new_tracks = []

                for (X,Y,WW,HH,kps) in faces:
                    if WW<=0 or HH<=0 or HH < self.min_face_px: continue
                    if not kps or len(kps) < 2: continue
                    re = (int(kps[0].x * W), int(kps[0].y * H))
                    le = (int(kps[1].x * W), int(kps[1].y * H))
                    def inside(pt): return X <= pt[0] <= X+WW and Y <= pt[1] <= Y+HH
                    if not (inside(re) and inside(le)): continue
                    eye_frac = abs(le[0]-re[0]) / float(WW)
                    if eye_frac < self.min_eye_frac: continue

                    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi = gray_full[Y:Y+HH, X:X+WW]
                    if roi.size == 0: continue
                    if cv2.Laplacian(roi, cv2.CV_64F).var() < self.min_sharp: continue

                    # align by eyes within ROI
                    re_roi = (re[0]-X, re[1]-Y); le_roi = (le[0]-X, le[1]-Y)
                    roi_aligned = rotate_roi_by_eyes(roi, re_roi, le_roi)

                    # square center crop → 200x200, preprocess
                    hh, ww = roi_aligned.shape
                    s = min(hh, ww); cy, cx = hh//2, ww//2
                    roi_sq = roi_aligned[cy - s//2: cy + s//2, cx - s//2: cx + s//2]
                    if roi_sq.size == 0: continue
                    roi_200 = cv2.resize(roi_sq, (200,200), cv2.INTER_AREA)
                    roi_200 = preprocess_gray(roi_200)

                    pred, conf = rec.predict(roi_200)
                    conf = float(conf)
                    lid = int(pred)
                    who = inv_labels.get(lid, "Unknown")

                    good = (who != "Unknown") and (conf <= model_thr)

                    # Prototype gate (if present)
                    if good and proto_means is not None and 0 <= lid < proto_means.shape[0]:
                        mu = proto_means[lid]  # (200,200) float32
                        rmse = float(np.sqrt(np.mean((roi_200.astype(np.float32) - mu) ** 2)))
                        if rmse > float(proto_thr[lid]):
                            good = False  # reject as unknown

                    if self.debug_detect:
                        cv2.rectangle(vis, (X,Y), (X+WW,Y+HH), (0,255,255), 1)

                    # tracking-based stability
                    cx, cy = X + WW/2.0, Y + HH/2.0
                    assigned = None
                    for tr in self.tracks:
                        tx, ty = tr["center"]
                        if abs(cx - tx) <= self.track_dist_thr*WW and abs(cy - ty) <= self.track_dist_thr*HH:
                            assigned = tr; break
                    if assigned is None:
                        assigned = {"center": (cx,cy), "label": None, "count": 0, "marked": False}
                    assigned["center"] = (cx,cy)
                    if good:
                        # debounced hit
                        self._register_hit(who)
                        # label consensus
                        if assigned["label"] == who:
                            assigned["count"] += 1
                        else:
                            assigned["label"] = who
                            assigned["count"] = 1
                        # draw
                        cv2.rectangle(vis, (X,Y), (X+WW,Y+HH), (0,255,0), 2)
                        tag = f"{who}" + (f" ({conf:.1f})" if self.show_conf else "")
                        cv2.putText(vis, tag, (X, max(22, Y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(vis, tag, (X, max(22, Y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

                        if assigned["count"] >= self.stable and not assigned["marked"]:
                            assigned["marked"] = True
                            mark(who, conf)
                    else:
                        assigned["label"] = None
                        assigned["count"] = 0
                    new_tracks.append(assigned)

                self.tracks = new_tracks

                # HUD
                fps_frames += 1
                if fps_frames >= 15:
                    now = time.time(); dt = now - fps_tick
                    fps = fps_frames / dt if dt > 0 else 0.0
                    fps_tick, fps_frames = now, 0
                    self.statsReady.emit(
                        f"det:{self.det_mode} conf:{self.det_conf:.2f} min_face:{self.min_face_px}px sharp:{self.min_sharp:.1f} eye:{self.min_eye_frac:.2f} thr:{model_thr:.1f} FPS:{fps:.1f}"
                    )

                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(vis_rgb.data, vis_rgb.shape[1], vis_rgb.shape[0], vis_rgb.strides[0], QtGui.QImage.Format_RGB888)
                self.frameReady.emit(qimg.copy())
        except Exception as e:
            self.error.emit(str(e))

class RecognizeTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grabber: FrameGrabber | None = None
        self.proc: RecognizeProcessor | None = None
        self.q_frames = queue.Queue(maxsize=2)

        # Left: video + attendance table
        self.videoLabel = QtWidgets.QLabel("Preview")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.videoLabel.setMinimumSize(960, 540)
        self.videoLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time", "Session", "Student ID", "Confidence"])
        self.table.horizontalHeader().setStretchLastSection(True)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.videoLabel, 4)
        left.addWidget(self.table, 2)
        leftW = QtWidgets.QWidget(); leftW.setLayout(left)

        # Right controls
        self.modelPath = QtWidgets.QLineEdit("models/lbph.yml"); self.btnModel = QtWidgets.QPushButton("…")
        self.labelsPath = QtWidgets.QLineEdit("models/labels.json"); self.btnLabels = QtWidgets.QPushButton("…")
        self.sessionEdit = QtWidgets.QLineEdit("Session")
        self.outdirEdit = QtWidgets.QLineEdit("attendance")
        self.camIndex = QtWidgets.QSpinBox(); self.camIndex.setRange(0,8); self.camIndex.setValue(0)
        self.widthBox = QtWidgets.QComboBox(); self.widthBox.addItems(["1280","1920"]); self.widthBox.setCurrentText("1280")
        self.heightBox= QtWidgets.QComboBox(); self.heightBox.addItems(["720","1080"]); self.heightBox.setCurrentText("720")
        self.fpsBox = QtWidgets.QSpinBox(); self.fpsBox.setRange(5,60); self.fpsBox.setValue(30)

        self.detMode = QtWidgets.QComboBox(); self.detMode.addItems(["auto","near","far"]); self.detMode.setCurrentText("auto")
        self.confSlider = QtWidgets.QSlider(Qt.Horizontal); self.confSlider.setRange(10,95); self.confSlider.setValue(35)
        self.confLabel = QtWidgets.QLabel("0.35")
        self.minFace = QtWidgets.QSpinBox(); self.minFace.setRange(40,400); self.minFace.setValue(110)
        self.eyeFrac = QtWidgets.QDoubleSpinBox(); self.eyeFrac.setRange(0.05,0.6); self.eyeFrac.setSingleStep(0.01); self.eyeFrac.setValue(0.18)
        self.sharp = QtWidgets.QDoubleSpinBox(); self.sharp.setRange(0.0,200.0); self.sharp.setValue(8.0)
        self.scaleBox = QtWidgets.QComboBox(); self.scaleBox.addItems(["1.0","0.75","0.5"]); self.scaleBox.setCurrentText("0.75")

        self.threshold = QtWidgets.QDoubleSpinBox(); self.threshold.setRange(1.0,200.0); self.threshold.setValue(60.0)
        self.stable = QtWidgets.QSpinBox(); self.stable.setRange(1,20); self.stable.setValue(5)
        self.showConf = QtWidgets.QCheckBox("Show confidence")
        self.debugDet = QtWidgets.QCheckBox("Debug detections (yellow)")

        self.countTable = QtWidgets.QTableWidget(0, 2)
        self.countTable.setHorizontalHeaderLabels(["Student ID", "Hits"])
        self.countTable.horizontalHeader().setStretchLastSection(True)
        self.btnResetCounts = QtWidgets.QPushButton("Reset Counters")
        self._count_rows: dict[str,int] = {}

        self.btnStart = QtWidgets.QPushButton("Start Preview")
        self.btnStop = QtWidgets.QPushButton("Stop Preview")
        self.btnReset = QtWidgets.QPushButton("Reset Session Table")

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(6,6,6,6); form.setVerticalSpacing(4)
        form.addRow("Model:", self._row(self.modelPath, self.btnModel))
        form.addRow("Labels:", self._row(self.labelsPath, self.btnLabels))
        form.addRow("Session:", self.sessionEdit)
        form.addRow("Attendance dir:", self.outdirEdit)
        form.addRow("Camera index:", self.camIndex)
        wh = QtWidgets.QHBoxLayout(); wh.addWidget(self.widthBox); wh.addWidget(QtWidgets.QLabel("×")); wh.addWidget(self.heightBox); wh.addWidget(QtWidgets.QLabel(" @ ")); wh.addWidget(self.fpsBox)
        form.addRow("Resolution:", self._wrap(wh))
        cRow = QtWidgets.QHBoxLayout(); cRow.addWidget(self.confSlider); cRow.addWidget(self.confLabel)
        form.addRow("Detector conf:", self._wrap(cRow))
        form.addRow("Detector mode:", self.detMode)
        form.addRow("Min face (px):", self.minFace)
        form.addRow("Min eye frac:", self.eyeFrac)
        form.addRow("Min sharpness:", self.sharp)
        form.addRow("Detection scale:", self.scaleBox)
        form.addRow("LBPH threshold:", self.threshold)
        form.addRow("Stable frames:", self.stable)
        form.addRow(self.showConf); form.addRow(self.debugDet)
        form.addRow(QtWidgets.QLabel("Recognition Counters:")); form.addWidget(self.countTable); form.addRow(self.btnResetCounts)
        form.addRow(self.btnStart); form.addRow(self.btnStop); form.addRow(self.btnReset)

        rightPanel = QtWidgets.QWidget(); rightPanel.setLayout(form)
        rightScroll = QtWidgets.QScrollArea(); rightScroll.setWidgetResizable(True); rightScroll.setWidget(rightPanel)
        rightScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter.addWidget(leftW); self.splitter.addWidget(rightScroll)
        self.splitter.setStretchFactor(0, 65); self.splitter.setStretchFactor(1, 35)

        layout = QtWidgets.QHBoxLayout(self); layout.addWidget(self.splitter)
        QtCore.QTimer.singleShot(0, self._apply_split_sizes)

        self.btnModel.clicked.connect(self._pick_model)
        self.btnLabels.clicked.connect(self._pick_labels)
        self.btnStart.clicked.connect(self.start_preview)
        self.btnStop.clicked.connect(self.stop_preview)
        self.btnReset.clicked.connect(self._reset_table)
        self.btnResetCounts.clicked.connect(self._reset_counts)
        self.confSlider.valueChanged.connect(lambda v: self.confLabel.setText(f"{v/100.0:.2f}"))

    def _apply_split_sizes(self):
        total = max(1, self.splitter.width())
        self.splitter.setSizes([int(total*0.65), int(total*0.35)])

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        QtCore.QTimer.singleShot(0, self._apply_split_sizes)
        return super().resizeEvent(e)

    def _wrap(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); w.setLayout(layout); return w

    def _row(self, *widgets) -> QtWidgets.QWidget:
        h = QtWidgets.QHBoxLayout()
        for w in widgets: h.addWidget(w)
        return self._wrap(h)

    def _pick_model(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select model", self.modelPath.text(), "LBPH (*.yml)")
        if p: self.modelPath.setText(p)

    def _pick_labels(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select labels", self.labelsPath.text(), "JSON (*.json)")
        if p: self.labelsPath.setText(p)

    def _reset_table(self):
        self.table.setRowCount(0)

    def _reset_counts(self):
        self.countTable.setRowCount(0); self._count_rows = {}
        if self.proc: self.proc.reset_counts()

    @Slot()
    def start_preview(self):
        if self.grabber or self.proc: return
        cam = int(self.camIndex.value())
        width = int(self.widthBox.currentText()); height = int(self.heightBox.currentText()); fps = int(self.fpsBox.value())
        self.grabber = FrameGrabber(cam, width, height, fps, self.q_frames)

        det_mode = self.detMode.currentText(); det_conf = self.confSlider.value()/100.0; det_scale = float(self.scaleBox.currentText())
        proc = RecognizeProcessor(
            self.q_frames,
            model_path=self.modelPath.text().strip(),
            labels_path=self.labelsPath.text().strip(),
            det_mode=det_mode, det_conf=det_conf, det_scale=det_scale,
            min_face_px=int(self.minFace.value()), min_eye_frac=float(self.eyeFrac.value()), min_sharp=float(self.sharp.value()),
            threshold=float(self.threshold.value()), stable=int(self.stable.value()),
            show_conf=self.showConf.isChecked(), debug_detect=self.debugDet.isChecked(),
            session=self.sessionEdit.text().strip() or "Session", outdir=self.outdirEdit.text().strip() or "attendance"
        )
        self.proc = proc
        self.proc.frameReady.connect(self.on_frame)
        self.proc.statsReady.connect(lambda s: self.setToolTip(s))
        self.proc.attendanceMarked.connect(self.on_mark)
        self.proc.recognizedHit.connect(self.on_hit)
        self.proc.error.connect(self.on_error)

        self.grabber.start(); self.proc.start()

    @Slot()
    def stop_preview(self):
        if self.proc: self.proc.stop(); self.proc.wait(1000); self.proc=None
        if self.grabber: self.grabber.stop(); self.grabber.wait(1000); self.grabber=None
        try:
            while self.q_frames.qsize(): self.q_frames.get_nowait()
        except Exception: pass

    @Slot(QtGui.QImage)
    def on_frame(self, qimg: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pix.scaled(self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(str, float, str)
    def on_mark(self, sid: str, conf: float, ts: str):
        r = self.table.rowCount(); self.table.insertRow(r)
        for c, val in enumerate([ts, self.sessionEdit.text().strip() or "Session", sid, f"{conf:.1f}"]):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(str(val)))
        self.table.scrollToBottom()

    @Slot(str, int)
    def on_hit(self, sid: str, count: int):
        row = self._count_rows.get(sid, None)
        if row is None:
            row = self.countTable.rowCount(); self.countTable.insertRow(row)
            self._count_rows[sid] = row
            self.countTable.setItem(row, 0, QtWidgets.QTableWidgetItem(sid))
        self.countTable.setItem(row, 1, QtWidgets.QTableWidgetItem(str(count)))
        self.countTable.scrollToBottom()

    @Slot(str)
    def on_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Recognize error", msg)

# ----------------------- MAIN WINDOW -----------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automated Rural Attendance monitoring system")
        self.resize(1680, 980)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(CaptureTab(), "Capture")
        tabs.addTab(TrainTab(), "Train")
        tabs.addTab(RecognizeTab(), "Recognize & Attendance")
        self.setCentralWidget(tabs)
        self.setStyleSheet("""
        QPushButton{padding:6px 10px;border-radius:6px;}
        QPushButton:hover{background:#eef;}
        QLabel,QCheckBox,QComboBox,QSpinBox,QDoubleSpinBox,QLineEdit{font-size:12px;}
        QFormLayout{margin:0px;}
        """)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
