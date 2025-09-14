# Unified Face Attendance App — Capture UI (v2, **multithreaded**)
# -----------------------------------------------------------------
# Requirements:
#   pip install PySide6 opencv-contrib-python mediapipe numpy
#
# What this does
# - Single-window GUI (PySide6) for CAPTURE (Train/Recognize tabs will come next)
# - **True multithreading**: separate FrameGrabber (camera) + Processor (detect+save) threads
# - Bounded queue between threads (drops stale frames → smooth, low-latency UI)
# - Auto camera backend fallback (MSMF → DSHOW → DEFAULT)
# - Controls: Student ID, FAR/NEAR, detector conf, min face px, save-every-N, max images, resolution, FPS
# - Saves 200×200 CLAHE-normalized grayscale crops to data/faces/<student_id>/

from __future__ import annotations
import os, sys, time, uuid, queue
import cv2
import numpy as np
import mediapipe as mp

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot

APP_VERSION = "Capture UI v2.1 (multithreaded • fast-start)"

# ----------------------- helpers -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    """CLAHE + normalize for lighting robustness. Input must be grayscale."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(img)
    return cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX)

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
                # Try to reduce internal buffering if backend supports it
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
            # push newest frame, drop stale
            try:
                while self.out_q.qsize() >= self.out_q.maxsize:
                    self.out_q.get_nowait()
            except queue.Empty:
                pass
            self.out_q.put(frame)
        if self._cap:
            self._cap.release()

# ----------------------- processor (consumer) -----------------------
class Processor(QtCore.QThread):
    frameReady = Signal(QtGui.QImage)           # processed frame with overlays
    statsReady = Signal(str)                    # HUD/status
    savedCountChanged = Signal(int)             # count of saved images
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
                 cold_start_frames: int = 30,
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

        # runtime stats
        self._fps_tick = time.time()
        self._fps_frames = 0

        self._detector = None

    # public slots to adjust live
    @Slot(bool)
    def set_capturing(self, on: bool):
        self._capturing = bool(on)

    @Slot(float)
    def set_conf(self, conf: float):
        self.det_conf = float(conf)
        self._detector = None  # recreate with new conf lazily

    @Slot(bool)
    def set_far(self, far: bool):
        self.far_model = bool(far)
        self._detector = None  # recreate with new model lazily

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
        # Prepare output
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

            # --- Fast start: pass-through first N frames (no detection) ---
            if self._detector is None and frames_seen <= self.cold_start_frames:
                vis_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(vis_rgb.data, vis_rgb.shape[1], vis_rgb.shape[0], vis_rgb.strides[0], QtGui.QImage.Format_RGB888)
                self.frameReady.emit(qimg.copy())
                # create detector once after we've shown a few frames
                if frames_seen == self.cold_start_frames:
                    self._create_detector()
                continue

            if self._detector is None:
                self._create_detector()
                # still show raw while creating
                vis_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(vis_rgb.data, vis_rgb.shape[1], vis_rgb.shape[0], vis_rgb.strides[0], QtGui.QImage.Format_RGB888)
                self.frameReady.emit(qimg.copy())
                continue

            # Detection at reduced scale for speed
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
                # MediaPipe bounding boxes are relative → safe across scales
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

            # Save logic — save-every-N frames, up to max_images
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

            # FPS / HUD every ~15 frames
            self._fps_frames += 1
            if self._fps_frames >= 15:
                now = time.time(); dt = now - self._fps_tick
                fps = self._fps_frames / dt if dt > 0 else 0.0
                self._fps_tick = now; self._fps_frames = 0
                hud = (f"Det:{'FAR' if self.far_model else 'NEAR'} conf:{self.det_conf:.2f}  "
                       f"minFace:{self.min_face_px}px  Saved:{self._saved}/{self.max_images}  FPS:{fps:.1f}  scale:{scale:.2f}")
                self.statsReady.emit(hud)

            # Emit frame to UI
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(vis_rgb.data, vis_rgb.shape[1], vis_rgb.shape[0], vis_rgb.strides[0], QtGui.QImage.Format_RGB888)
            self.frameReady.emit(qimg.copy())

# ----------------------- UI -----------------------
class CaptureTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CaptureTab")
        self.grabber: FrameGrabber | None = None
        self.proc: Processor | None = None
        self.q_frames: queue.Queue = queue.Queue(maxsize=2)  # shared between threads

        # Left: video preview
        self.videoLabel = QtWidgets.QLabel()
        self.videoLabel.setFixedSize(960, 540)  # 16:9 preview
        self.videoLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setText("Preview")

        # Right: controls
        self.studentEdit = QtWidgets.QLineEdit()
        self.studentEdit.setPlaceholderText("Student ID (e.g., 1001)")
        self.camIndex = QtWidgets.QSpinBox(); self.camIndex.setRange(0, 8); self.camIndex.setValue(0)
        self.widthBox  = QtWidgets.QComboBox(); self.widthBox.addItems(["1280", "1920"]); self.widthBox.setCurrentText("1280")
        self.heightBox = QtWidgets.QComboBox(); self.heightBox.addItems(["720", "1080"]); self.heightBox.setCurrentText("720")
        self.fpsBox    = QtWidgets.QSpinBox(); self.fpsBox.setRange(5, 60); self.fpsBox.setValue(30)
        self.farCheck  = QtWidgets.QCheckBox("FAR model (2–5 m)")
        self.confSlider = QtWidgets.QSlider(Qt.Horizontal); self.confSlider.setRange(10, 95); self.confSlider.setValue(40)
        self.confLabel  = QtWidgets.QLabel("0.40")
        self.minFace    = QtWidgets.QSpinBox(); self.minFace.setRange(40, 400); self.minFace.setValue(120)
        self.saveEvery  = QtWidgets.QSpinBox(); self.saveEvery.setRange(1, 60); self.saveEvery.setValue(5)
        self.maxImages  = QtWidgets.QSpinBox(); self.maxImages.setRange(1, 2000); self.maxImages.setValue(150)
        self.outRoot    = QtWidgets.QLineEdit("data/faces")

        self.startBtn   = QtWidgets.QPushButton("Start Preview")
        self.stopBtn    = QtWidgets.QPushButton("Stop Preview")
        self.captureBtn = QtWidgets.QPushButton("Start Capture"); self.captureBtn.setCheckable(True)

        self.savedLabel = QtWidgets.QLabel("Saved: 0")
        self.statusBar  = QtWidgets.QLabel(); self.statusBar.setStyleSheet("color: #888;")

        form = QtWidgets.QFormLayout()
        form.addRow("Student ID:", self.studentEdit)
        form.addRow("Camera index:", self.camIndex)
        wh = QtWidgets.QHBoxLayout(); wh.addWidget(self.widthBox); wh.addWidget(QtWidgets.QLabel("×")); wh.addWidget(self.heightBox); wh.addWidget(QtWidgets.QLabel(" @ ")); wh.addWidget(self.fpsBox); form.addRow("Resolution:", self._wrap(wh))
        form.addRow(self.farCheck)
        confRow = QtWidgets.QHBoxLayout(); confRow.addWidget(self.confSlider); confRow.addWidget(self.confLabel); form.addRow("Detector conf:", self._wrap(confRow))
        form.addRow("Min face (px):", self.minFace)
        form.addRow("Save every N frames:", self.saveEvery)
        form.addRow("Max images:", self.maxImages)
        form.addRow("Output root:", self.outRoot)
        # Detection scale (speeds up startup & runtime)
        self.scaleBox = QtWidgets.QComboBox(); self.scaleBox.addItems(["1.0", "0.75", "0.5"]); self.scaleBox.setCurrentText("0.75")
        form.addRow("Detection scale:", self.scaleBox)
        form.addRow(self.startBtn)
        form.addRow(self.stopBtn)
        form.addRow(self.captureBtn)
        form.addRow(self.savedLabel)
        form.addRow(self.statusBar)

        right = QtWidgets.QWidget(); right.setLayout(form)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.videoLabel, 3)
        layout.addWidget(right, 2)

        # Connections
        self.startBtn.clicked.connect(self.start_preview)
        self.stopBtn.clicked.connect(self.stop_preview)
        self.captureBtn.toggled.connect(self.toggle_capture)
        self.confSlider.valueChanged.connect(self._conf_changed)

        self._update_buttons(running=False)

    def _wrap(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); w.setLayout(layout); return w

    def _conf_changed(self, val: int):
        conf = val / 100.0
        self.confLabel.setText(f"{conf:.2f}")
        if self.proc:
            self.proc.set_conf(conf)

    def _update_buttons(self, running: bool):
        self.startBtn.setEnabled(not running)
        self.stopBtn.setEnabled(running)
        self.captureBtn.setEnabled(running)

    @Slot()
    def start_preview(self):
        if self.grabber or self.proc:
            return
        cam = int(self.camIndex.value())
        width = int(self.widthBox.currentText())
        height = int(self.heightBox.currentText())
        fps = int(self.fpsBox.value())
        far = bool(self.farCheck.isChecked())
        conf = self.confSlider.value() / 100.0
        min_face_px = int(self.minFace.value())
        every = int(self.saveEvery.value())
        max_images = int(self.maxImages.value())
        out_root = self.outRoot.text().strip() or "data/faces"
        student_id = self.studentEdit.text().strip()
        if not os.path.isdir(out_root): ensure_dir(out_root)

        # threads share the same bounded queue
        self.grabber = FrameGrabber(cam, width, height, fps, self.q_frames)
        det_scale = float(self.scaleBox.currentText())
        self.proc    = Processor(self.q_frames, student_id, out_root, far, conf, min_face_px, every, max_images, det_downscale=det_scale, cold_start_frames=20)

        # Connect signals
        self.grabber.error.connect(self.on_error)
        self.grabber.info.connect(self.on_status)
        self.proc.frameReady.connect(self.on_frame)
        self.proc.statsReady.connect(self.on_status)
        self.proc.savedCountChanged.connect(self.on_saved_count)
        self.proc.error.connect(self.on_error)

        # Start threads
        self.grabber.start()
        self.proc.start()

        self._update_buttons(True)
        self.statusBar.setText("Starting preview…")
        self.savedLabel.setText("Saved: 0")

    @Slot()
    def stop_preview(self):
        if self.proc:
            self.proc.stop(); self.proc.wait(1000); self.proc = None
        if self.grabber:
            self.grabber.stop(); self.grabber.wait(1000); self.grabber = None
        # drain queue
        try:
            while self.q_frames.qsize(): self.q_frames.get_nowait()
        except Exception:
            pass
        self._update_buttons(False)
        self.statusBar.setText("Stopped.")

    @Slot(bool)
    def toggle_capture(self, checked: bool):
        if not self.proc:
            self.captureBtn.setChecked(False)
            return
        if checked and not self.studentEdit.text().strip():
            QtWidgets.QMessageBox.warning(self, "Missing Student ID", "Please enter a Student ID before capturing.")
            self.captureBtn.setChecked(False)
            return
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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Face Attendance — {APP_VERSION}")
        self.resize(1280, 720)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(CaptureTab(), "Capture")
        # TODO: add Train / Recognize tabs in next steps using the same threading pattern
        self.setCentralWidget(tabs)
        self.setStyleSheet("QPushButton{padding:6px 10px;border-radius:6px;} QPushButton:hover{background:#eef;} QLabel{font-size:13px}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
