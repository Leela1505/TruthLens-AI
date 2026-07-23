-- TruthLens AI Database Schema
-- Compatible with MySQL 5.7+ / MySQL 8.0+ / MariaDB

CREATE DATABASE IF NOT EXISTS truthlens_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE truthlens_db;

-- --------------------------------------------------------
-- Table structure for table `users`
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(50) NOT NULL UNIQUE,
    `email` VARCHAR(100) NOT NULL UNIQUE,
    `password_hash` VARCHAR(255) NOT NULL,
    `role` ENUM('user', 'admin') DEFAULT 'user',
    `full_name` VARCHAR(100) DEFAULT '',
    `bio` TEXT,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------
-- Table structure for table `predictions`
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS `predictions` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `user_id` INT NOT NULL,
    `news_title` VARCHAR(255) DEFAULT 'Untitled News',
    `news_text` TEXT NOT NULL,
    `prediction` VARCHAR(10) NOT NULL, -- 'REAL' or 'FAKE'
    `confidence` FLOAT NOT NULL,       -- Percentage e.g. 96.5
    `explanation` TEXT,                -- JSON string or keyword explanation
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------
-- Insert Default Seed Data
-- Default Admin User: admin / admin123
-- Password hash generated with werkzeug pbkdf2:sha256
-- --------------------------------------------------------
INSERT INTO `users` (`username`, `email`, `password_hash`, `role`, `full_name`) 
VALUES (
    'admin', 
    'admin@truthlens.ai', 
    'scrypt:32768:8:1$uH34sYQvFvjA6V6L$99026330058b763321487caeeae299532bf143588916dcfbc6d5d568ea46a9a089d1b702ec8c9ec9ca66b26cfed144f800be744bf8365f5ee1edbc20f5be8709', 
    'admin', 
    'TruthLens Administrator'
) ON DUPLICATE KEY UPDATE `id`=`id`;

INSERT INTO `users` (`username`, `email`, `password_hash`, `role`, `full_name`) 
VALUES (
    'demouser', 
    'user@truthlens.ai', 
    'scrypt:32768:8:1$uH34sYQvFvjA6V6L$99026330058b763321487caeeae299532bf143588916dcfbc6d5d568ea46a9a089d1b702ec8c9ec9ca66b26cfed144f800be744bf8365f5ee1edbc20f5be8709', 
    'user', 
    'Demo Analyst'
) ON DUPLICATE KEY UPDATE `id`=`id`;

-- Sample predictions
INSERT INTO `predictions` (`user_id`, `news_title`, `news_text`, `prediction`, `confidence`, `explanation`, `created_at`) 
VALUES 
(1, 'Global Climate Summit Reaches Landmark Accord', 'Representatives from 190 countries signed a historic climate pact today pledging net zero emissions by 2050. Science advisors confirmed data alignment.', 'REAL', 97.4, 'Key indicators: summit, accord, representatives, pact, climate, science', NOW() - INTERVAL 2 DAY),
(2, 'Secret Alien Base Found Under Antarctic Ice Sheet', 'Breaking scandal! Anonymous source claims massive alien spacecraft hidden beneath 2 miles of ice since 1947. NASA covering up truth!', 'FAKE', 98.9, 'Key indicators: breaking scandal, anonymous source, alien spacecraft, covering up, truth!', NOW() - INTERVAL 1 DAY);
