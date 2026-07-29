-- Profiles table for Jarvis face recognition + profile switching
-- Stored in MariaDB jarvis database (192.168.55.41:3306)

CREATE TABLE IF NOT EXISTS profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    system_prompt TEXT NOT NULL DEFAULT 'You are Jarvis, a helpful AI assistant.',
    chat_history JSON NOT NULL DEFAULT '[]',
    accent_hue INT NOT NULL DEFAULT 182,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Default profile for "spidertje" (Filip)
INSERT INTO profiles (name, system_prompt, accent_hue, enabled)
VALUES (
    'spidertje',
    'You are Jarvis, Filip Van Damme''s personal AI assistant. Filip is an IT professional with 25 years of experience looking for IT support, sysadmin, and infrastructure roles in Latvia or remote. Speak in Dutch when addressing him directly. Keep responses concise and technical.',
    182,
    TRUE
)
ON DUPLICATE KEY UPDATE system_prompt = VALUES(system_prompt);

-- Default profile for "Dace"
INSERT INTO profiles (name, system_prompt, accent_hue, enabled)
VALUES (
    'Dace',
    'You are Jarvis, Dace''s personal AI assistant. Be warm, helpful, and friendly. Use Dutch when addressing her directly.',
    320,
    TRUE
)
ON DUPLICATE KEY UPDATE system_prompt = VALUES(system_prompt);
