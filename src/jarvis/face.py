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
    db_host: str = "192.168.55.41"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = "rocklobster"
    db_name: str = "jarvis"
    cascade_path: Optional[str] = None  # None = use OpenCV default
    confidence_threshold: float = 80.0  # LBPH threshold (lower = stricter)
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
        # Face config uses env or defaults
        if self.config.db_host is None:
            self.config.db_host = os.environ.get("JARVIS_DB_HOST", "192.168.55.41")
        if self.config.db_user is None:
            self.config.db_user = os.environ.get("JARVIS_DB_USER", "root")
        if self.config.db_password is None:
            self.config.db_password = os.environ.get("JARVIS_DB_PASSWORD", "")
        self._models: Dict[str, cv2.face.LBPHFaceRecognizer] = {}
        self._last_recognition: Dict[str, float] = {}  # name -> last recognition time
        self._db = None
        self._cascade = None

    def _get_db(self) -> pymysql.Connection:
        """Get or create MariaDB connection."""
        if self._db is None or self._db.closed:
            self._db = pymysql.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                user=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name,
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
        Train a face model from a single frame.

        For best results, call this multiple times with different angles.
        """
        faces = self.detect_faces(frame)
        if not faces:
            logger.warning(f"No faces detected in frame for {name}")
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = faces[0]  # Use first detected face

        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, self.config.resize_dims)

        # Get or create model
        db = self._get_db()
        with db.cursor() as cur:
            cur.execute("SELECT id FROM face_model WHERE name = %s", (name,))
            row = cur.fetchone()

            if row:
                # Update existing model
                model = cv2.face.LBPHFaceRecognizer_create()
                model.read("/dev/null")  # placeholder
                try:
                    model.update([roi], [0])
                except Exception:
                    # Model already trained, can't update
                    logger.warning(f"Cannot update model for {name} (already trained)")
                    return False

                # Serialize and save
                model_blob = self._save_model_blob(model)
                cur.execute(
                    "UPDATE face_model SET model = %s WHERE name = %s",
                    (model_blob, name),
                )
            else:
                # Create new model
                model = cv2.face.LBPHFaceRecognizer_create()
                model.train([roi], [0])

                model_blob = self._save_model_blob(model)
                cur.execute(
                    "INSERT INTO face_model (name, model) VALUES (%s, %s)",
                    (name, model_blob),
                )

            db.commit()

        # Reload models to include new/updated model
        self.load_models()
        return True

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
