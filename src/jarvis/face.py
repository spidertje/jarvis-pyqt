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

import io
import logging
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

import cv2
import pymysql
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceConfig:
    """Face recognition configuration."""
    camera_index: int = 0
    db_host: Optional[str] = None  # set via JARVIS_DB_HOST or FaceRecognizer.__init__
    db_port: int = 3306
    db_user: Optional[str] = None
    db_password: Optional[str] = None  # set via JARVIS_DB_PASSWORD env var
    db_name: Optional[str] = None
    cascade_path: Optional[str] = None  # None = use OpenCV default
    confidence_threshold: float = 80.0  # LBPH distance threshold (lower = stricter)
    debounce_seconds: float = 2.0  # minimum time between same-name recognitions
    min_faces_to_add: int = 20  # min samples needed to train
    resize_dims: tuple = (100, 100)  # resize faces to these dims for LBPH


class FaceRecognizer:
    """
    OpenCV LBPH face recognizer with MariaDB model storage.

    Loads trained models from DB, detects faces via Haar cascades,
    and predicts identity using LBPH.
    """

    def __init__(self, config: Optional[FaceConfig] = None):
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
        self._models: Dict[str, cv2.face.LBPHFaceRecognizer] = {}
        self._last_recognition: Dict[str, float] = {}  # name -> last recognition time
        self._db = None
        self._cascade = None

    def _get_db(self) -> pymysql.Connection:
        """Get or create MariaDB connection."""
        if self._db is None or self._db.closed:
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

    def load_models(self) -> int:
        """
        Load all trained face models from MariaDB.

        Returns:
            Number of models loaded.
        """
        self._models.clear()
        db = self._get_db()

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

    def _load_model_blob(self, blob: bytes) -> Optional[cv2.face.LBPHFaceRecognizer]:
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

    def detect_faces(self, frame: np.ndarray) -> List[tuple]:
        """
        Detect faces in a frame using Haar cascades.

        Args:
            frame: BGR image from webcam

        Returns:
            List of (x, y, w, h) tuples for detected faces.
        """
        cascade = self._get_cascade()
        if cascade.empty():
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        return [tuple(face) for face in faces]

    def recognize(self, frame: np.ndarray) -> Optional[str]:
        """
        Detect and recognize faces in a frame.

        Args:
            frame: BGR image from webcam

        Returns:
            Recognized name, or None if no face or confidence too low.
        """
        if not self._models:
            return None

        faces = self.detect_faces(frame)
        if not faces:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_name = None
        best_conf = float("inf")

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, self.config.resize_dims)

            for name, model in self._models.items():
                try:
                    _, conf = model.predict(roi)
                    if conf < best_conf:
                        best_conf = conf
                        best_name = name
                except Exception:
                    continue

        # Check confidence threshold and debounce
        if best_name and best_conf < self.config.confidence_threshold:
            now = time.time()
            last = self._last_recognition.get(best_name, 0)
            if now - last >= self.config.debounce_seconds:
                self._last_recognition[best_name] = now
                return best_name

        return None

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
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = faces[0]

        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, self.config.resize_dims)

        # Serialize ROI for DB storage
        sample_blob = self._serialize_roi(roi)
        if not sample_blob:
            logger.error("Failed to serialize face sample")
            return False

        # Store sample in DB
        db = self._get_db()
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO face_samples (name, sample, created_at) VALUES (%s, %s, NOW())",
                (name, sample_blob),
            )
            db.commit()

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

    def _deserialize_roi(self, blob: bytes) -> Optional[np.ndarray]:
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
        if name not in self._models:
            logger.error(f"No existing model for {name}")
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
            for i, row in enumerate(rows):
                roi = self._deserialize_roi(row["sample"])
                if roi is not None:
                    samples.append(roi)
                    labels.append(i)

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

    def list_faces(self) -> List[str]:
        """List all known faces."""
        try:
            db = self._get_db()
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
        if self._db and not self._db.closed:
            self._db.close()
