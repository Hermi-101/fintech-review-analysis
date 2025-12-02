-- File: db/schema.sql

-- Database: bank_reviews
-- This script should be run *after* creating the 'bank_reviews' database.

-- 1. DROP TABLES if they exist to allow for clean re-creation
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS banks;

-- 2. Banks Table: Stores information about the banks
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) UNIQUE NOT NULL,
    app_name VARCHAR(100)
);

-- 3. Reviews Table: Stores the scraped and processed review data
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    -- Foreign Key link to the banks table
    bank_id INTEGER NOT NULL REFERENCES banks(bank_id),
    review_text TEXT,
    rating INTEGER NOT NULL,
    review_date DATE,
    sentiment_label VARCHAR(50),
    sentiment_score NUMERIC(5, 4), -- Store scores up to 4 decimal places (e.g., 0.9999)
    identified_themes TEXT, -- Store the comma-separated themes
    source VARCHAR(50) DEFAULT 'Google Play Store' NOT NULL
);

-- Initial Data Insert: Insert the three primary banks
INSERT INTO banks (bank_name, app_name) VALUES
('Commercial Bank of Ethiopia', 'CBE Mobile Banking'),
('Bank of Abyssinia', 'Bank of Abyssinia Mobile App'),
('Dashen Bank', 'Dashen Mobile Banking');