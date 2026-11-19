-- 005: searchable chat history (created by the app on first DB use too —
-- this script is for manual/provisioned installs).
CREATE TABLE IF NOT EXISTS chat_messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    profile_name VARCHAR(100) NOT NULL,
    role VARCHAR(16) NOT NULL,
    content MEDIUMTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_profile_time (profile_name, created_at),
    INDEX idx_profile_id (profile_name, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
