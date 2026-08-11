"""
Jarvis Face Recognition — OpenCV LBPH with MariaDB storage.

Detects and recognizes faces from webcam using LBPH (Local Binary Patterns
Histograms) algorithm. Trained models stored as LONGBLOB in MariaDB.

Usage:
    recognizer = FaceRecognizer(camera_index=0, db_host='192.168.55.41')
    name = recognizer.recognize(frame)  # returns name or None
    recognizer.add(name, samples)  # train on samples
    names = recognizer.list_faces()  # list known faces
    recognizer.delete(name)  # remove a model
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pymysql

logger = logging.getLogger(__name__)


@dataclass
class FaceConfig:
    """Face recognition configuration."""

    camera_index: int = 0
    db_host: str | None = None  # set via JARVIS_DB_HOST or FaceRecognizer.__init__
    db_port: int = 3306
    db_user: str | None = None
    db_password: str | None = None  # set via JARVIS_DB_PASSWORD env var
    db_name: str | None = None
    # ONNX face detector (Ultra-Light-Fast-Generic-Face-Detector RFB-320).
    detector_model: str | None = None  # None = default ~/.hermes/version-RFB-320.onnx
    dnn_score_threshold: float = 0.6  # forward confidence cutoff (0..1)
    # Haar cascade fallback (kept for compatibility)
    cascade_path: str | None = None  # None = use OpenCV default
    confidence_threshold: float = 70.0  # LBPH distance threshold (lower = stricter)
    debounce_seconds: float = 2.0  # minimum time between same-name recognitions
    min_faces_to_add: int = 20  # min samples needed to train
    resize_dims: tuple = (100, 100)  # resize faces to these dims for LBPH
    # Sample quality gating (root cause of inconsistent recognition: blurry/AWB-bad
    # samples get trained in). Reject samples below these thresholds.
    sample_min_focus: float = 20.0  # min Laplacian variance of the grayscale ROI
    sample_min_brightness: float = 30.0  # min mean pixel value (too dark = reject)
    sample_max_brightness: float = 235.0  # max mean pixel value (blown out = reject)
    # Temporal voting: hold a sliding window of predictions per identity and only
    # emit a recognition once the majority of the window agrees. Kills flicker on
    # a 10fps loop where single frames mispredict.
    vote_window: int = 5  # frames of history per identity
    vote_majority: float = 0.6  # fraction of window that must agree


class FaceRecognizer:
    """
    OpenCV LBPH face recognizer with MariaDB model storage.

    Loads trained models from DB, detects faces via Haar cascades,
    and predicts identity using LBPH.
    """

    def __init__(self, config: FaceConfig | None = None):
        self.config = config or FaceConfig()
        # Resolve None values from env vars (no hardcoded fallbacks)
        if self.config.db_host is None:
            self.config.db_host = os.environ.get("JARVIS_DB_HOST")
        if self.config.db_user is None:
            self.config.db_user = os.environ.get("JARVIS_DB_USER")
        if self.config.db_password is None:
            self.config.db_password = os.environ.get("JARVIS_DB_PASSWORD")
        if self.config.db_name is None:
            self.config.db_name = os.environ.get("JARVIS_DB_NAME")
        self._models: dict[str, cv2.face.LBPHFaceRecognizer] = {}
        self._last_recognition: dict[str, float] = {}  # name -> last recognition time
        self._vote_buffers: dict[str, list[bool]] = {}  # name -> recent hit/miss window
        self._db = None
        self._cascade = None
        self._net = None
        self._last_db_error: str | None = None
        self._last_sample_error: str | None = None

    def _get_db(self) -> pymysql.Connection:
        """Get or create MariaDB connection."""
        if self._db is None or self._db._closed:
            if self.config.db_host is None or self.config.db_user is None:
                raise RuntimeError(
                    "DB host and user must be configured via env vars "
                    "(JARVIS_DB_HOST, JARVIS_DB_USER) or FaceConfig"
                )
            self._db = pymysql.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                user=self.config.db_user,
                password=self.config.db_password or "",
                database=self.config.db_name or "jarvis",
                cursorclass=pymysql.cursors.DictCursor,
            )
            # Ensure face tables exist (idempotent)
            with self._db.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_samples (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        sample LONGBLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_name (name)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_model (
                        name VARCHAR(100) PRIMARY KEY,
                        model LONGBLOB NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
            self._db.commit()
        return self._db

    def _get_cascade(self) -> cv2.CascadeClassifier:
        """Get or create Haar cascade classifier."""
        if self._cascade is None:
            path = self.config.cascade_path
            if path is None:
                # Try common paths
                candidates = [
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
                    os.path.expanduser("~/.hermes/haarcascade_frontalface_default.xml"),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        path = candidate
                        break
                if path is None:
                    logger.warning("No Haar cascade found, using OpenCV default")
                    path = None
            if path is not None:
                self._cascade = cv2.CascadeClassifier(path)
                logger.info(f"Loaded cascade from {path}")
            else:
                self._cascade = cv2.CascadeClassifier()
                logger.warning("No cascade file found — face detection will fail")
        return self._cascade

    def _get_net(self):
        """Load ONNX face detector model (OpenCV DNN)."""
        if self._net is None:
            model = self.config.detector_model
            if model is None:
                model = os.path.expanduser("~/.hermes/version-RFB-320.onnx")
            if os.path.exists(model):
                try:
                    self._net = cv2.dnn.readNetFromONNX(model)
                    logger.info(f"Loaded ONNX face detector from {model}")
                except Exception as e:
                    logger.error(f"Failed to load ONNX face detector: {e}")
                    self._net = None
            else:
                logger.warning(f"ONNX model not found: {model}")
                self._net = None
        return self._net

    def load_models(self) -> int:
        """
        Load all trained face models from MariaDB.

        Returns:
            Number of models loaded.
        """
        self._models.clear()
        try:
            db = self._get_db()
        except (RuntimeError, pymysql.err.OperationalError) as e:
            logger.warning(f"Face DB unavailable: {e}")
            return 0

        try:
            with db.cursor() as cur:
                cur.execute("SELECT name, model FROM face_model")
                rows = cur.fetchall()

            for row in rows:
                name = row["name"]
                model_blob = row["model"]
                model = self._load_model_blob(model_blob)
                if model is not None:
                    self._models[name] = model
                    logger.info(f"Loaded model for: {name}")

            logger.info(f"Loaded {len(self._models)} face models")
            return len(self._models)

        except Exception as e:
            logger.error(f"Failed to load face models: {e}")
            return 0

    def _load_model_blob(self, blob: bytes) -> cv2.face.LBPHFaceRecognizer | None:
        """Load an LBPH model from a LONGBLOB."""
        try:
            # Try pickle deserialization first
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                f.write(blob)
                temp_path = f.name

            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(temp_path)
            os.unlink(temp_path)
            return model

        except Exception:
            # Fallback: try writing as YML (OpenCV native format)
            try:
                with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
                    f.write(blob)
                    temp_path = f.name

                model = cv2.face.LBPHFaceRecognizer_create()
                model.read(temp_path)
                os.unlink(temp_path)
                return model

            except Exception as e:
                logger.error(f"Failed to load model blob: {e}")
                return None

    def detect_faces(self, frame: np.ndarray) -> list[tuple]:
        """Detect faces in a frame using ONNX (RFB-320) or Haar cascade fallback."""
        # Try ONNX first
        net = self._get_net()
        if net is not None:
            try:
                h, w = frame.shape[:2]
                # RFB-320 expects 320x240 input
                blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 240), (104, 117, 123), False, False)
                net.setInput(blob)
                scores, boxes = net.forward(
                    net.getUnconnectedOutLayersNames()
                )  # (1, 4420, 2), (1, 4420, 4)
                scores = scores[0, :, 1]  # face class scores (index 1), shape (4420,)
                boxes = boxes[0]  # (4420, 4) — cx, cy, w, h normalized
                # Collect raw boxes + confidences for NMS
                nms_boxes = []
                nms_scores = []
                for i in range(scores.shape[0]):
                    conf = float(scores[i])
                    if conf < self.config.dnn_score_threshold:
                        continue
                    cx, cy, bw, bh = boxes[i]
                    # Denormalize to original frame size
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    # Clamp
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        nms_boxes.append([x1, y1, x2, y2])
                        nms_scores.append(conf)
                # Apply Non-Maximum Suppression to remove overlapping detections
                if nms_boxes:
                    indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores,
                                              self.config.dnn_score_threshold, 0.3)
                    faces = []
                    for idx in indices.flatten():
                        x1, y1, x2, y2 = nms_boxes[idx]
                        faces.append((x1, y1, x2 - x1, y2 - y1))
                    return faces
                return []
            except Exception as e:
                logger.warning(f"ONNX face detection failed: {e}, falling back to Haar")
        # Fallback to Haar cascade
        cascade = self._get_cascade()
        if cascade.empty():
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(40, 40),
        )
        return [tuple(face) for face in faces]

    def recognize(self, frame: np.ndarray) -> tuple[str, float] | None:
        """
        Detect and recognize faces in a frame.

        Args:
            frame: BGR image from webcam

        Returns:
            (Recognized name, confidence) tuple where confidence is the
            LBPH distance (lower = better), or None if no face or
            confidence too low.
        """
        if not self._models:
            return None

        faces = self.detect_faces(frame)
        if not faces:
            # No faces present this frame — drift the vote buffers down so a
            # committed recognition decays rather than sticking.
            for buf in self._vote_buffers.values():
                if buf:
                    buf.pop(0)
            if len(self._vote_buffers) > 1:
                self._vote_buffers = {
                    k: v for k, v in self._vote_buffers.items() if v
                }
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pred = self._predict_frame(gray, faces)

        # ── Temporal voting ──────────────────────────────────────────
        # Push this frame's per-identity match into each buffer; a name is
        # only emitted when its sliding window crosses the majority threshold.
        if pred is None:
            for buf in self._vote_buffers.values():
                if buf:
                    buf.pop(0)
            return None

        name, conf = pred
        buf = self._vote_buffers.setdefault(name, [])
        buf.append(True)
        if len(buf) > self.config.vote_window:
            buf.pop(0)
        # Also decay non-selected buffers so alternating flicker can't sustain a wrong hit.
        for other, obuf in self._vote_buffers.items():
            if other != name and obuf:
                obuf.pop(0)

        if sum(buf) / float(self.config.vote_window) < self.config.vote_majority:
            return None

        # ── Confidence threshold + debounce ─────────────────────────
        if conf < self.config.confidence_threshold:
            now = time.time()
            last = self._last_recognition.get(name, 0)
            if now - last >= self.config.debounce_seconds:
                self._last_recognition[name] = now
                return (name, conf)

        return None

    def _predict_frame(self, gray: np.ndarray, faces: list[tuple]) -> tuple | None:
        """Predict the best (name, confidence) across all detected faces."""
        best_name = None
        best_conf = float("inf")

        for x, y, w, h in faces:
            roi = gray[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, self.config.resize_dims)

            for name, model in self._models.items():
                try:
                    _, conf = model.predict(roi)
                    if conf < best_conf:
                        best_conf = conf
                        best_name = name
                except Exception:
                    continue
        if best_name is None:
            return None
        return (best_name, best_conf)

    def add_face(self, name: str, frame: np.ndarray) -> bool:
        """
        Collect a face sample for a person.

        Samples are stored in a separate `face_samples` table. Call this multiple
        times with different angles/lighting, then call retrain_face(name) to
        train the LBPH model from all collected samples.
        """
        faces = self.detect_faces(frame)
        if not faces:
            logger.warning(f"No faces detected in frame for {name}")
            self._last_sample_error = "No face detected in frame"
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = faces[0]

        roi = gray[y : y + h, x : x + w]
        # Quality gate — reject blurry / too dark / blown-out samples so they
        # don't poison the LBPH model. Records WHY so the UI can show the real
        # reason instead of blaming the DB.
        focus = self._laplacian_variance(roi)
        mean = float(roi.mean()) if roi.size else 0.0
        if focus < self.config.sample_min_focus:
            logger.debug(f"Rejecting sample for {name}: blurry (focus={focus:.1f})")
            self._last_sample_error = (
                f"Sample too blurry (focus {focus:.0f}) — hold still"
            )
            return False
        if mean < self.config.sample_min_brightness or mean > self.config.sample_max_brightness:
            logger.debug(
                f"Rejecting sample for {name}: bad exposure (mean={mean:.1f})"
            )
            self._last_sample_error = (
                f"Bad exposure (mean {mean:.0f}) — adjust lighting"
            )
            return False
        self._last_sample_error = None

        roi = cv2.resize(roi, self.config.resize_dims)

        # Serialize ROI for DB storage
        sample_blob = self._serialize_roi(roi)
        if not sample_blob:
            logger.error("Failed to serialize face sample")
            return False

        # Store sample in DB
        try:
            db = self._get_db()
        except (RuntimeError, pymysql.err.OperationalError) as e:
            logger.debug(f"DB unavailable, cannot store face sample: {e}")
            self._last_db_error = str(e)
            return False
        try:
            with db.cursor() as cur:
                cur.execute(
                    "INSERT INTO face_samples (name, sample, created_at) VALUES (%s, %s, NOW())",
                    (name, sample_blob),
                )
                db.commit()
        except Exception as e:
            logger.error(f"Failed to store face sample: {e}")
            return False

        count = self._get_sample_count(name)
        logger.info(f"Added face sample for {name} ({count} total)")
        return True

    def _serialize_roi(self, roi: np.ndarray) -> bytes:
        """Serialize a face ROI to bytes for DB storage."""
        try:
            _, buf = cv2.imencode(".png", roi)
            return buf.tobytes()
        except Exception as e:
            logger.error(f"Failed to serialize ROI: {e}")
            return b""

    def _laplacian_variance(self, roi: np.ndarray) -> float:
        """Compute Laplacian variance as a focus/sharpness metric (higher = sharper)."""
        if roi.size == 0:
            return 0.0
        return float(cv2.Laplacian(roi, cv2.CV_64F).var())

    def _deserialize_roi(self, blob: bytes) -> np.ndarray | None:
        """Deserialize a face ROI from DB bytes."""
        try:
            arr = np.frombuffer(blob, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            logger.error(f"Failed to deserialize ROI: {e}")
            return None

    def _get_sample_count(self, name: str) -> int:
        """Count stored samples for a person."""
        try:
            db = self._get_db()
            with db.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM face_samples WHERE name = %s", (name,))
                return cur.fetchone()["cnt"]
        except Exception:
            return 0

    def retrain_face(self, name: str) -> bool:
        """
        Retrain LBPH model from all stored samples for a person.

        Requires face_samples table to exist with (name, sample) columns.
        """
        # Need at least one sample to retrain
        count = self._get_sample_count(name)
        if count < self.config.min_faces_to_add:
            logger.warning(f"Not enough samples for {name}: {count}/{self.config.min_faces_to_add}")
            return False

        try:
            db = self._get_db()
            with db.cursor() as cur:
                cur.execute("SELECT sample FROM face_samples WHERE name = %s ORDER BY id", (name,))
                rows = cur.fetchall()

            if not rows:
                logger.warning(f"No samples found for {name}")
                return False

            samples = []
            labels = []
            for row in rows:
                roi = self._deserialize_roi(row["sample"])
                if roi is not None:
                    samples.append(roi)
                    labels.append(0)  # All samples for same person share label 0

            if not samples:
                logger.warning(f"No valid samples for {name}")
                return False

            logger.info(f"Retraining model for {name} with {len(samples)} samples")
            new_model = cv2.face.LBPHFaceRecognizer_create()
            new_model.train(samples, np.array(labels))

            # Save to DB
            model_blob = self._save_model_blob(new_model)
            with db.cursor() as cur:
                cur.execute("UPDATE face_model SET model = %s WHERE name = %s", (model_blob, name))
                db.commit()

            # Reload all models
            self.load_models()
            logger.info(f"Model retrained for {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to retrain model for {name}: {e}")
            return False

    def maybe_retrain(self, name: str) -> bool:
        """Retrain if enough new samples have accumulated since last training.
        
        Returns True if retrained, False otherwise.
        """
        count = self._get_sample_count(name)
        if count < self.config.min_faces_to_add:
            return False
        # Simple heuristic: retrain every N samples (e.g., every 10)
        # In production you'd track last_train_count in the DB.
        if count % 10 == 0:
            return self.retrain_face(name)
        return False

    def train_new_face(self, name: str) -> bool:
        """
        Train and store a new face model from collected samples.

        Unlike retrain_face(), this creates a brand-new model entry (INSERT)
        rather than updating an existing one. Requires face_samples table
        to contain samples for the given name.
        """
        try:
            db = self._get_db()
            with db.cursor() as cur:
                cur.execute("SELECT sample FROM face_samples WHERE name = %s ORDER BY id", (name,))
                rows = cur.fetchall()

            if not rows:
                logger.warning(f"No samples found for {name}")
                return False

            samples = []
            labels = []
            for row in rows:
                roi = self._deserialize_roi(row["sample"])
                if roi is not None:
                    samples.append(roi)
                    labels.append(0)  # All samples for same person share label 0

            if not samples:
                logger.warning(f"No valid samples for {name}")
                return False

            logger.info(f"Training new model for {name} with {len(samples)} samples")
            new_model = cv2.face.LBPHFaceRecognizer_create()
            new_model.train(samples, np.array(labels))

            model_blob = self._save_model_blob(new_model)
            with db.cursor() as cur:
                cur.execute(
                    "INSERT INTO face_model (name, model) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE model = VALUES(model)",
                    (name, model_blob),
                )
                db.commit()

            # Reload all models so the new one takes effect
            self.load_models()
            logger.info(f"New model trained and stored for {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to train new model for {name}: {e}")
            return False

    def _save_model_blob(self, model: cv2.face.LBPHFaceRecognizer) -> bytes:
        """Serialize an LBPH model to bytes for DB storage."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                model.write(f.name)
                with open(f.name, "rb") as rf:
                    blob = rf.read()
                os.unlink(f.name)
                return blob
        except Exception:
            # Fallback: try YML format
            try:
                with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
                    model.write(f.name)
                    with open(f.name, "rb") as rf:
                        blob = rf.read()
                    os.unlink(f.name)
                    return blob
            except Exception as e:
                logger.error(f"Failed to serialize model: {e}")
                return b""

    def list_faces(self) -> list[str]:
        """List all known faces."""
        try:
            db = self._get_db()
        except (RuntimeError, pymysql.err.OperationalError):
            return []
        try:
            with db.cursor() as cur:
                cur.execute("SELECT name FROM face_model ORDER BY name")
                return [row["name"] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list faces: {e}")
            return []

    def delete_face(self, name: str) -> bool:
        """Delete a face model from DB."""
        try:
            db = self._get_db()
        except (RuntimeError, pymysql.err.OperationalError):
            return False
        try:
            with db.cursor() as cur:
                cur.execute("DELETE FROM face_model WHERE name = %s", (name,))
                db.commit()

            if name in self._models:
                del self._models[name]

            return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete face {name}: {e}")
            return False

    def close(self):
        """Close database connection."""
        if self._db is not None:
            try:
                self._db.close()
            except Exception as e:
                logger.debug(f"DB close (already closed?): {e}")
