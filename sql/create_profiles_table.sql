-- Face model table for jarvis-vision / face recognition
-- Stored in MariaDB jarvis database (192.168.55.41:3306)

CREATE TABLE IF NOT EXISTS face_model (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    model BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Profiles table for Jarvis face recognition + profile switching
-- Stored in MariaDB jarvis database (192.168.55.41:3306)

CREATE TABLE IF NOT EXISTS profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    assistant_name VARCHAR(50) NOT NULL DEFAULT 'Jarvis',
    assistant_full TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT 'You are Jarvis, a helpful AI assistant.',
    chat_history JSON NOT NULL DEFAULT '[]',
     accent_hue INT NOT NULL DEFAULT 182,
     palette VARCHAR(50) NOT NULL DEFAULT 'cyan',
     is_default BOOLEAN NOT NULL DEFAULT FALSE,
     enabled BOOLEAN NOT NULL DEFAULT TRUE,
     api_key VARCHAR(255) NOT NULL DEFAULT '' COMMENT 'Reserved for web frontend; PyQt inserts empty string',
     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Default profile for "spidertje" (Filip)
INSERT INTO profiles (name, assistant_name, assistant_full, system_prompt, accent_hue, palette, is_default, enabled)
VALUES (
    'spidertje',
    'Hermes',
    'You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful, knowledgeable, and direct. You assist users with a wide range of tasks including answering questions, writing and editing code, analyzing information, creative work, and executing actions via your tools. You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below. Be targeted and efficient in your exploration and investigations.',
    'You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful, knowledgeable, and direct. You assist users with a wide range of tasks including answering questions, writing and editing code, analyzing information, creative work, and executing actions via your tools. You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below. Be targeted and efficient in your exploration and investigations.',
    182,
    'cyan',
    TRUE,
    TRUE
)
ON DUPLICATE KEY UPDATE system_prompt = VALUES(system_prompt), assistant_name = VALUES(assistant_name), assistant_full = VALUES(assistant_full);

-- Default profile for "Dace"
INSERT INTO profiles (name, assistant_name, assistant_full, system_prompt, accent_hue, palette, is_default, enabled)
VALUES (
    'Dace',
    'Jarvis',
    'You are Jarvis, Dace personal AI assistant. Be warm, helpful, and friendly. Use Dutch when addressing her directly.',
    'You are Jarvis, Dace personal AI assistant. Be warm, helpful, and friendly. Use Dutch when addressing her directly.',
    320,
    'violet',
    FALSE,
    TRUE
)
ON DUPLICATE KEY UPDATE system_prompt = VALUES(system_prompt), assistant_name = VALUES(assistant_name), assistant_full = VALUES(assistant_full);
