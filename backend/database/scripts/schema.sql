-- SQL schema and database setup
CREATE DATABASE IF NOT EXISTS jobhunter_ai;
USE jobhunter_ai;

-- 1. USERS
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    title VARCHAR(100),
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('jobseeker', 'recruiter', 'admin') DEFAULT 'jobseeker',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. USER PROFILES
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id INT PRIMARY KEY,
    experience_level ENUM('entry','mid','senior','executive') DEFAULT 'entry',
    bio TEXT,
    location VARCHAR(150),
    desired_industry VARCHAR(150),
    phone VARCHAR(50),
    job_preferences JSON,
    desired_salary DECIMAL(10,2),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_profiles_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT uq_user_profiles_user UNIQUE (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DELIMITER $$
CREATE TRIGGER trg_users_after_insert
AFTER INSERT ON users
FOR EACH ROW
BEGIN
    INSERT IGNORE INTO user_profiles (user_id) VALUES (NEW.id);
END$$
DELIMITER ;

-- 3. USER SKILLS
CREATE TABLE IF NOT EXISTS user_skills (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    skill_name VARCHAR(100) NOT NULL,
    proficiency ENUM('beginner','intermediate','advanced','expert') DEFAULT 'beginner',
    years_experience INT DEFAULT 0,
    CONSTRAINT fk_user_skills_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_skill_per_user (user_id, skill_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. EDUCATION
CREATE TABLE IF NOT EXISTS education (
    edu_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    institution VARCHAR(150),
    degree VARCHAR(100),
    field_of_study VARCHAR(100),
    graduation_year YEAR,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. CERTIFICATIONS
CREATE TABLE IF NOT EXISTS certifications (
    cert_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    certification_name VARCHAR(150),
    issuer VARCHAR(150),
    year_obtained YEAR,
    expiration_date DATE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. RESUMES
CREATE TABLE IF NOT EXISTS resumes (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  resume_text LONGTEXT NOT NULL,
  file_name VARCHAR(255) NULL,
  parsed_sections JSON NULL,
  parsed_contacts JSON NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_resumes_user_created (user_id, created_at DESC),
  CONSTRAINT fk_resumes_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 7. JOBS
CREATE TABLE IF NOT EXISTS jobs (
    job_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(150) NOT NULL,
    company_name VARCHAR(150),
    industry VARCHAR(100),
    description TEXT,
    location VARCHAR(100),
    requirements TEXT,
    url VARCHAR(255),
    salary_range VARCHAR(100),
    source ENUM('internal','api') DEFAULT 'internal',
    posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_job_unique (title, company_name, location, url)
);

-- 8. JOB RECOMMENDATIONS
CREATE TABLE IF NOT EXISTS job_recommendations (
    rec_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    job_id INT NOT NULL,
    match_score DECIMAL(5,2),
    generated_resume LONGTEXT,
    generated_cover_letter LONGTEXT,
    recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 9. SAVED JOBS
CREATE TABLE IF NOT EXISTS saved_jobs (
    saved_job_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    job_id INT NOT NULL,
    date_saved TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes VARCHAR(255),
    CONSTRAINT fk_savedjobs_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_savedjobs_job  FOREIGN KEY (job_id)  REFERENCES jobs(job_id) ON DELETE CASCADE,
    UNIQUE KEY uq_user_job (user_id, job_id)
);

-- 10. APPLIED JOBS
CREATE TABLE IF NOT EXISTS applied_jobs (
    applied_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    job_id INT NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_appliedjobs_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_appliedjobs_job FOREIGN KEY (job_id)
        REFERENCES jobs(job_id) ON DELETE CASCADE,
    UNIQUE KEY uq_applied_user_job (user_id, job_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------------
-- Migration: Add external_id to jobs table
USE jobhunter_ai;

-- Add external_id column if it doesn't exist
-- Add external_id column and index if they do not exist (portable across MySQL/MariaDB)
-- Use INFORMATION_SCHEMA checks + prepared statements so this script is idempotent

SET @schema_name = DATABASE();

-- Add column if missing
SELECT COUNT(*) INTO @col_count
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = @schema_name AND TABLE_NAME = 'jobs' AND COLUMN_NAME = 'external_id';

SET @add_col_sql = IF(@col_count = 0,
    'ALTER TABLE jobs ADD COLUMN external_id VARCHAR(100) NULL AFTER job_id',
    'SELECT "external_id column already exists"');

PREPARE stmt_col FROM @add_col_sql;
EXECUTE stmt_col;
DEALLOCATE PREPARE stmt_col;

-- Add unique index if missing
SELECT COUNT(*) INTO @idx_count
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = @schema_name AND TABLE_NAME = 'jobs' AND INDEX_NAME = 'idx_jobs_external_id';

SET @add_idx_sql = IF(@idx_count = 0,
    'ALTER TABLE jobs ADD UNIQUE INDEX idx_jobs_external_id (external_id)',
    'SELECT "idx_jobs_external_id already exists"');

PREPARE stmt_idx FROM @add_idx_sql;
EXECUTE stmt_idx;
DEALLOCATE PREPARE stmt_idx;

-- Ensure the column has a helpful comment (this is safe to run repeatedly)
ALTER TABLE jobs
    MODIFY COLUMN external_id VARCHAR(100) NULL COMMENT 'External job ID from API provider (e.g., Adzuna ID)';
