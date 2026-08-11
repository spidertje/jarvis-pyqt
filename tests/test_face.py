"""Tests for FaceConfig and FaceRecognizer."""

from unittest.mock import MagicMock, patch

import numpy as np

from jarvis.face import FaceConfig, FaceRecognizer


class TestFaceConfig:
    def test_defaults(self):
        cfg = FaceConfig()
        assert cfg.camera_index == 0
        assert cfg.db_port == 3306
        assert cfg.dnn_score_threshold == 0.6
        assert cfg.confidence_threshold == 70.0
        assert cfg.debounce_seconds == 2.0
        assert cfg.min_faces_to_add == 20
        assert cfg.resize_dims == (100, 100)

    def test_custom_values(self):
        cfg = FaceConfig(
            camera_index=2,
            db_host="10.0.0.1",
            confidence_threshold=50.0,
            min_faces_to_add=10,
        )
        assert cfg.camera_index == 2
        assert cfg.db_host == "10.0.0.1"
        assert cfg.confidence_threshold == 50.0
        assert cfg.min_faces_to_add == 10


class TestFaceRecognizerInit:
    def test_init_resolves_env_vars(self, monkeypatch):
        monkeypatch.setenv("JARVIS_DB_HOST", "env-host")
        monkeypatch.setenv("JARVIS_DB_USER", "env-user")
        monkeypatch.setenv("JARVIS_DB_PASSWORD", "env-pass")
        monkeypatch.setenv("JARVIS_DB_NAME", "env-db")
        rec = FaceRecognizer(FaceConfig(db_host=None, db_user=None))
        assert rec.config.db_host == "env-host"
        assert rec.config.db_user == "env-user"
        assert rec.config.db_password == "env-pass"
        assert rec.config.db_name == "env-db"

    def test_init_no_env_fallbacks(self):
        """FaceConfig with explicit values should not override from env."""
        rec = FaceRecognizer(FaceConfig(db_host="explicit-host", db_user="explicit-user"))
        assert rec.config.db_host == "explicit-host"
        assert rec.config.db_user == "explicit-user"


class TestSerializeDeserialize:
    def test_roi_round_trip(self):
        """_serialize_roi / _deserialize_roi should round-trip a face ROI."""
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        roi = np.zeros((100, 100), dtype=np.uint8)
        roi[50, 50] = 255
        blob = rec._serialize_roi(roi)
        assert len(blob) > 0
        restored = rec._deserialize_roi(blob)
        assert restored is not None
        assert restored.shape == (100, 100)

    def test_serialize_roi_failure_returns_empty(self):
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        blob = rec._serialize_roi(None)
        assert blob == b""


class TestDetectFaces:
    @patch("jarvis.face.FaceRecognizer._get_net", return_value=None)
    def test_detect_no_models_no_faces(self, mock_net):
        """When no cascade and no ONNX model, detect_faces returns empty."""
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        rec._cascade = MagicMock()
        rec._cascade.empty.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = rec.detect_faces(frame)
        assert faces == []

    @patch("jarvis.face.FaceRecognizer._get_net", return_value=None)
    @patch("jarvis.face.FaceRecognizer._get_cascade")
    def test_detect_haar_fallback(self, mock_cascade, mock_net):
        """Haar cascade fallback should return detected faces."""
        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[10, 20, 50, 60]])
        mock_cascade.return_value = cascade
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = rec.detect_faces(frame)
        assert len(faces) == 1
        assert faces[0] == (10, 20, 50, 60)

    def test_detect_onnx_applies_nms(self):
        """ONNX detection should apply NMS to remove overlapping boxes."""
        mock_net = MagicMock()
        mock_net.getUnconnectedOutLayersNames.return_value = ["scores", "boxes"]

        # 10 detections with high confidence, slightly overlapping
        scores_output = np.ones((1, 10, 2), dtype=np.float32)
        scores_output[0, :, 0] = 0.01  # background
        scores_output[0, :, 1] = 0.99  # face

        boxes_output = np.zeros((1, 10, 4), dtype=np.float32)
        for i in range(10):
            boxes_output[0, i] = [0.5 + i * 0.001, 0.5 + i * 0.001,
                                  0.2 + i * 0.0005, 0.3 + i * 0.0005]

        mock_net.forward.return_value = (scores_output, boxes_output)
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        rec._net = mock_net
        rec._cascade = MagicMock()
        rec._cascade.empty.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        faces = rec.detect_faces(frame)
        # NMS should reduce 10 overlapping detections to 1
        assert len(faces) == 1


class TestRecognize:
    def test_recognize_no_models(self):
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        rec._models = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.recognize(frame)
        assert result is None

    @patch("jarvis.face.FaceRecognizer.detect_faces")
    def test_recognize_no_faces(self, mock_detect):
        mock_detect.return_value = []
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        rec._models = {"alice": MagicMock()}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.recognize(frame)
        assert result is None

    @patch("jarvis.face.FaceRecognizer.detect_faces")
    def test_recognize_success(self, mock_detect):
        """recognize should return (name, conf) for best match below threshold."""
        mock_detect.return_value = [(10, 20, 50, 60)]
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        model = MagicMock()
        model.predict.return_value = (0, 50.0)  # label, confidence
        rec._models = {"alice": model}
        rec.config.confidence_threshold = 70.0
        # Temporal voting needs a whole window of matches before it emits.
        # Seed the vote buffer already-satisfied so a single call returns immediately.
        rec._vote_buffers["alice"] = [True] * rec.config.vote_window
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.recognize(frame)
        assert result is not None
        assert result[0] == "alice"
        assert result[1] == 50.0

    @patch("jarvis.face.FaceRecognizer.detect_faces")
    def test_recognize_above_threshold(self, mock_detect):
        """recognize should return None when confidence above threshold."""
        mock_detect.return_value = [(10, 20, 50, 60)]
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        model = MagicMock()
        model.predict.return_value = (0, 80.0)
        rec._models = {"alice": model}
        rec.config.confidence_threshold = 70.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.recognize(frame)
        assert result is None


class TestAddFace:
    @patch("jarvis.face.FaceRecognizer.detect_faces")
    def test_add_face_no_faces(self, mock_detect):
        mock_detect.return_value = []
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        result = rec.add_face("bob", np.zeros((100, 100, 3), dtype=np.uint8))
        assert result is False


class TestListDeleteFaces:
    def test_list_faces_db_unavailable(self, monkeypatch):
        # Clear DB env vars — config.py's load_dotenv() picks up ~/jarvis-pyqt/.env
        for key in ["JARVIS_DB_HOST", "JARVIS_DB_USER", "JARVIS_DB_PASSWORD",
                    "JARVIS_DB_NAME", "JARVIS_DB_PORT"]:
            monkeypatch.delenv(key, raising=False)
        rec = FaceRecognizer(FaceConfig(db_host=None))
        assert rec.list_faces() == []

    def test_delete_face_db_unavailable(self):
        rec = FaceRecognizer(FaceConfig(db_host=None))
        assert rec.delete_face("alice") is False


class TestClose:
    def test_close_no_db(self):
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        rec._db = None
        rec.close()  # should not raise

    def test_close_with_db(self):
        rec = FaceRecognizer(FaceConfig(db_host="localhost"))
        rec._db = MagicMock()
        rec.close()
        rec._db.close.assert_called_once()


class TestGetDbTableCreation:
    def test_get_db_creates_face_tables(self):
        """_get_db should create both face_samples and face_model tables."""
        """_get_db should create both face_samples and face_model tables."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn._closed = False

        with patch("pymysql.connect", return_value=mock_conn):
            rec = FaceRecognizer(FaceConfig(db_host="localhost", db_user="user"))
            rec._get_db()

        # Two CREATE TABLE IF NOT EXISTS statements should have been executed
        execute_calls = mock_cursor.execute.call_args_list
        sql_statements = [call[0][0] for call in execute_calls]
        assert any("face_samples" in s for s in sql_statements)
        assert any("face_model" in s for s in sql_statements)
