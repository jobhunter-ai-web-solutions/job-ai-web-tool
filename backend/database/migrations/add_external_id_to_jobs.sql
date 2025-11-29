-- Migration: Add external_id column to jobs table
-- This will store Adzuna's unique job ID to prevent duplicates
-- when the same job is returned in different searches

USE jobhunter_ai;

-- Add external_id column if it doesn't exist
ALTER TABLE jobs 
ADD COLUMN IF NOT EXISTS external_id VARCHAR(100) NULL AFTER job_id,
ADD UNIQUE INDEX IF NOT EXISTS idx_jobs_external_id (external_id);

-- Add comment to document the column
ALTER TABLE jobs 
MODIFY COLUMN external_id VARCHAR(100) NULL COMMENT 'External job ID from API provider (e.g., Adzuna ID)';


