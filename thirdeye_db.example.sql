-- =========================================================
-- ThirdEye — Database Schema & Sample Setup
-- =========================================================
-- Database: thirdeye_db
-- Compatibility: MySQL 5.7+ / MySQL 8.0+ / MariaDB 10.3+
-- =========================================================

CREATE DATABASE IF NOT EXISTS `thirdeye_db` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `thirdeye_db`;

-- ---------------------------------------------------------
-- Table structure for table `users`
-- ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `full_name` varchar(100) DEFAULT NULL,
  `username` varchar(80) NOT NULL,
  `password` varchar(200) NOT NULL,
  `role` varchar(20) NOT NULL DEFAULT 'user',
  PRIMARY KEY (`id`),
  UNIQUE KEY `username_UNIQUE` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------
-- Table structure for table `persons`
-- ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS `persons` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `person_details` text,
  `image_path` varchar(200) NOT NULL,
  `face_encoding` longtext NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------
-- Sample Seed Data for Local Development
-- (Note: Create admin securely using `python create_admin.py`)
-- ---------------------------------------------------------

-- Sample test person record with synthetic zero vector placeholder
-- (Replace or add records through the ThirdEye Admin interface)
INSERT INTO `persons` (`id`, `name`, `person_details`, `image_path`, `face_encoding`) VALUES
(1, 'Demo Subject', 'Synthetic reference record for system verification', 'test_person.jpg', '0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0');
