-- Migration: Add api_key column to profiles table.
-- profile.py (ProfileManager.save) inserts into api_key (empty string for
-- PyQt-managed profiles, since the key is owned by the web frontend).
-- Without this column, `ProfileManager.save()` fails with
-- "Unknown column 'api_key'" on databases created from the older bundled
-- schema, silently breaking profile + face auto-create persistence.
-- Run: mysql -u root -p jarvis < sql/004_add_api_key_to_profiles.sql

ALTER TABLE profiles
    ADD COLUMN IF NOT EXISTS api_key VARCHAR(255) NOT NULL DEFAULT '' AFTER enabled;
