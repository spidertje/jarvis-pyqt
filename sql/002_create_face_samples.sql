-- Migration: Add face_samples table for training sample storage.
-- Required by FaceRecognizer.add_face() and retrain_face().
-- Run: mysql -u alex -p jarvis < sql/002_create_face_samples.sql

CREATE TABLE IF NOT EXISTS face_samples (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    sample LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
