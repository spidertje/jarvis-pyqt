-- Migration: Add face_name column to profiles table
-- Links a face identity (from face_model.name) to a profile
-- Run: mysql -u alex -p jarvis < sql/003_add_face_name_to_profiles.sql

ALTER TABLE profiles
    ADD COLUMN IF NOT EXISTS face_name VARCHAR(100) UNIQUE NULL AFTER name;

-- Backfill: if you want existing profiles to auto-link, set face_name = name
-- (uncomment if you want profiles that share the face identity name)
-- UPDATE profiles SET face_name = name WHERE face_name IS NULL;
