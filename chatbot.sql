-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 16, 2024 at 04:48 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.0.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `chatbot`
--

-- --------------------------------------------------------

--
-- Table structure for table `jawaban`
--

CREATE TABLE `jawaban` (
  `id` int(11) NOT NULL,
  `pertanyaanId` int(11) NOT NULL,
  `jawaban` varchar(191) NOT NULL,
  `createdAt` datetime(3) NOT NULL DEFAULT current_timestamp(3),
  `cosine` double NOT NULL,
  `score` double NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `jawaban`
--

INSERT INTO `jawaban` (`id`, `pertanyaanId`, `jawaban`, `createdAt`, `cosine`, `score`) VALUES
(2, 1, 'Bidikmisi adalah bantuan biaya pendidikan yang diberikan pemerintah dan ditujukan bagi lulusan Sekolah Menengah Atas (SMA) atau sederajat yang memiliki potensi akademik baik, tetapi memiliki ', '2024-05-16 21:14:50.654', 0, 1),
(3, 2, 'Kampus Merdeka merupakan kebijakan Menteri Pendidikan dan Kebudayaan Nadiem Makarim yang membebaskan mahasiswa untuk mengikuti kegiatan di luar program studinya selama 1 semester atau setara ', '2024-05-16 21:14:53.676', 1.0000000000000002, 1),
(4, 3, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:15:51.664', 1, 0.00020999999999999968),
(5, 4, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:16:55.049', 1, 1),
(6, 5, 'Kampus Merdeka merupakan kebijakan Menteri Pendidikan dan Kebudayaan Nadiem Makarim yang membebaskan mahasiswa untuk mengikuti kegiatan di luar program studinya selama 1 semester atau setara ', '2024-05-16 21:34:22.020', 1.0000000000000002, 2.1),
(7, 6, 'Kampus Merdeka merupakan kebijakan Menteri Pendidikan dan Kebudayaan Nadiem Makarim yang membebaskan mahasiswa untuk mengikuti kegiatan di luar program studinya selama 1 semester atau setara ', '2024-05-16 21:34:35.230', 1.0000000000000002, 2.1),
(8, 7, 'Kampus Merdeka merupakan kebijakan Menteri Pendidikan dan Kebudayaan Nadiem Makarim yang membebaskan mahasiswa untuk mengikuti kegiatan di luar program studinya selama 1 semester atau setara ', '2024-05-16 21:34:45.107', 1.0000000000000002, 1),
(9, 8, 'Kampus Merdeka merupakan kebijakan Menteri Pendidikan dan Kebudayaan Nadiem Makarim yang membebaskan mahasiswa untuk mengikuti kegiatan di luar program studinya selama 1 semester atau setara ', '2024-05-16 21:34:54.014', 1.0000000000000002, 9.261),
(10, 9, 'Kampus Merdeka merupakan kebijakan Menteri Pendidikan dan Kebudayaan Nadiem Makarim yang membebaskan mahasiswa untuk mengikuti kegiatan di luar program studinya selama 1 semester atau setara ', '2024-05-16 21:35:05.275', 1.0000000000000002, 1),
(11, 10, 'Kampus Merdeka merupakan kebijakan Menteri Pendidikan dan Kebudayaan Nadiem Makarim yang membebaskan mahasiswa untuk mengikuti kegiatan di luar program studinya selama 1 semester atau setara ', '2024-05-16 21:35:37.923', 1.0000000000000002, 1),
(12, 11, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:35:50.347', 1, 2.1),
(13, 12, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:36:01.438', 1, 4.41),
(14, 13, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:36:18.115', 1, 3502.7750054222097),
(15, 14, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:36:29.326', 1, 1),
(16, 15, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:38:29.048', 1, 1),
(17, 16, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:39:09.884', 1, 0.0000001000000000000001),
(18, 17, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:39:19.102', 1, 0.0009999999999999992),
(19, 18, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:39:34.046', 1, 1),
(20, 19, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:39:39.657', 1, 0.00009999999999999994),
(21, 20, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-16 21:39:52.435', 1, 1);

-- --------------------------------------------------------

--
-- Table structure for table `pertanyaan`
--

CREATE TABLE `pertanyaan` (
  `id` int(11) NOT NULL,
  `pertanyaan` varchar(191) NOT NULL,
  `createdAt` datetime(3) NOT NULL DEFAULT current_timestamp(3)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `pertanyaan`
--

INSERT INTO `pertanyaan` (`id`, `pertanyaan`, `createdAt`) VALUES
(4, 'hai', '2024-05-16 21:14:50.628'),
(5, 'apa itu kampus merdeka', '2024-05-16 21:14:53.655'),
(6, 'apa itu pens', '2024-05-16 21:15:51.569'),
(7, 'apa itu pens', '2024-05-16 21:16:54.964'),
(9, 'apa itu kampus merdeka', '2024-05-16 21:34:21.993'),
(10, 'apa itu kampus merdeka', '2024-05-16 21:34:35.211'),
(11, 'apa itu kampus merdeka', '2024-05-16 21:34:45.092'),
(12, 'apa itu kampus merdeka', '2024-05-16 21:34:53.999'),
(13, 'apa itu kampus merdeka', '2024-05-16 21:35:05.261'),
(14, 'apa itu kampus merdeka', '2024-05-16 21:35:37.909'),
(15, 'apa itu pens', '2024-05-16 21:35:50.261'),
(16, 'apa itu pens', '2024-05-16 21:36:01.351'),
(17, 'apa itu pens', '2024-05-16 21:36:18.038'),
(18, 'apa itu pens', '2024-05-16 21:36:29.243'),
(19, 'apa itu pens', '2024-05-16 21:38:28.947'),
(20, 'apa itu pens', '2024-05-16 21:39:09.798'),
(21, 'apa itu pens', '2024-05-16 21:39:19.021'),
(22, 'apa itu pens', '2024-05-16 21:39:33.962'),
(23, 'apa itu pens', '2024-05-16 21:39:39.561'),
(24, 'apa itu pens', '2024-05-16 21:39:52.355');

-- --------------------------------------------------------

--
-- Table structure for table `_prisma_migrations`
--

CREATE TABLE `_prisma_migrations` (
  `id` varchar(36) NOT NULL,
  `checksum` varchar(64) NOT NULL,
  `finished_at` datetime(3) DEFAULT NULL,
  `migration_name` varchar(255) NOT NULL,
  `logs` text DEFAULT NULL,
  `rolled_back_at` datetime(3) DEFAULT NULL,
  `started_at` datetime(3) NOT NULL DEFAULT current_timestamp(3),
  `applied_steps_count` int(10) UNSIGNED NOT NULL DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `_prisma_migrations`
--

INSERT INTO `_prisma_migrations` (`id`, `checksum`, `finished_at`, `migration_name`, `logs`, `rolled_back_at`, `started_at`, `applied_steps_count`) VALUES
('ddf5edca-bd5f-4a70-b585-e26ee4a9dbcd', '3ec31105df6599fc63409dc828d198a09c1919889aaa4333cb2f860fa137680f', '2024-05-16 14:11:44.141', '20240516141144_', NULL, NULL, '2024-05-16 14:11:44.095', 1);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `jawaban`
--
ALTER TABLE `jawaban`
  ADD PRIMARY KEY (`id`),
  ADD KEY `Jawaban_pertanyaanId_fkey` (`pertanyaanId`);

--
-- Indexes for table `pertanyaan`
--
ALTER TABLE `pertanyaan`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `_prisma_migrations`
--
ALTER TABLE `_prisma_migrations`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `jawaban`
--
ALTER TABLE `jawaban`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=22;

--
-- AUTO_INCREMENT for table `pertanyaan`
--
ALTER TABLE `pertanyaan`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=25;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `jawaban`
--
ALTER TABLE `jawaban`
  ADD CONSTRAINT `Jawaban_pertanyaanId_fkey` FOREIGN KEY (`pertanyaanId`) REFERENCES `pertanyaan` (`id`) ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
