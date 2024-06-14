-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 11, 2024 at 03:24 PM
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
(1, 1, 'Bidikmisi adalah bantuan biaya pendidikan yang diberikan pemerintah dan ditujukan bagi lulusan Sekolah Menengah Atas (SMA) atau sederajat yang memiliki potensi akademik baik, tetapi memiliki ', '2024-05-27 16:40:30.325', 1, 3.993),
(2, 1, 'Bidik Misi adalah Program bantuan pemerintah', '2024-05-27 16:40:30.325', 1, 3),
(3, 1, 'Bidik Misi Adalah Beasiswa', '2024-05-27 16:40:30.325', 1, 3),
(4, 2, 'meningkatkan akses dan kesempatan belajar di perguruan tinggi bagi peserta didik yang memiliki berpotensi akademik baik tetapi memiliki keterbatasan ekonomi', '2024-05-27 16:41:10.973', 1, 0.36299999999999977),
(5, 2, 'Mensejahterakan Mahasiswa', '2024-05-27 16:41:10.973', 1, 0.2999999999999998),
(6, 2, 'Meluluskan Mahasiswa', '2024-05-27 16:41:10.973', 1, 0.2999999999999998),
(7, 3, 'Mahasiswa berbeasiswa Bidik Misi dibebaskan melakukan pembayaran UKT/SPP , Ikoma namun tetap harus melakukan daftar ulang online', '2024-05-27 16:43:05.076', 1, 3.3),
(8, 3, 'Melalui ATM', '2024-05-27 16:43:05.076', 1, 3),
(9, 3, 'Bisa M banking', '2024-05-27 16:43:05.076', 1, 3),
(10, 4, 'Tidak, Anda harus login ke laman http://online.pens.ac.id pada menu daftar ulang klik tombol DAFTAR ULANG, lalu cetak bukti daftar ulang anda sebagai bukti anda sudah melakukan daftar ulang.', '2024-05-27 16:44:43.718', 1.0000000000000002, 3.3),
(11, 4, 'Otomatis', '2024-05-27 16:44:43.718', 1.0000000000000002, 3),
(12, 4, 'Tidak bisa', '2024-05-27 16:44:43.718', 1.0000000000000002, 3),
(13, 5, 'Karena jumlah yang dibayarkan tiap mahasiswa dan angkatan berbeda maka jumlah itu silahkan dicek di menu daftar ulang laman http://online.mis.eepis-its.edu atau ATM/Internet Banking/TELLER', '2024-05-27 16:44:54.644', 1.0000000000000002, 3.63),
(14, 5, 'Sesuai Nominal', '2024-05-27 16:44:54.644', 1.0000000000000002, 3),
(15, 5, 'Konsultasikan ke BAAK', '2024-05-27 16:44:54.644', 1.0000000000000002, 3),
(16, 6, 'Tidak Bisa', '2024-05-27 16:45:09.101', 1, 3.3),
(17, 6, 'Kurang tau', '2024-05-27 16:45:09.101', 1, 3),
(18, 6, 'Belum Bisa', '2024-05-27 16:45:09.101', 1, 3),
(19, 7, 'Untuk mendaftar beasiswa adaro dari PENS dapat melalui link https://intip.in/Adaro2020/', '2024-05-27 16:45:15.060', 1, 3.3),
(20, 7, 'Lewat Kemahasiswaan', '2024-05-27 16:45:15.060', 1, 3),
(21, 7, 'lewat PT djarum', '2024-05-27 16:45:15.060', 1, 3),
(22, 8, 'Untuk mendaftar beasiswa djarum plus dapat melalui website https://djarumbeasiswaplus.org/', '2024-05-27 16:45:23.757', 1, 3.3),
(23, 8, 'Lewat Website terterra', '2024-05-27 16:45:23.757', 1, 3),
(24, 8, 'Ditanyakan ke kemahasiswaan', '2024-05-27 16:45:23.757', 1, 3),
(25, 9, 'Gedung Kemahasiswaan PENS berlokasi di D4 Lantai 1 di Politeknik Elektronika Negeri Surabaya', '2024-05-27 16:45:31.631', 1.0000000000000002, 0.13784918959071632),
(26, 9, 'Di gedung PENS', '2024-05-27 16:45:31.631', 1.0000000000000002, 0.02999999999999997),
(27, 9, 'Di gedung D4 PENS', '2024-05-27 16:45:31.631', 1.0000000000000002, 0.02999999999999997),
(28, 10, 'Jalan Raya ITS - Kampus PENS Sukolilo Surabaya 60111, Indonesia', '2024-05-27 16:45:41.449', 1, 3.3),
(29, 10, 'Sukolilo ITS', '2024-05-27 16:45:41.449', 1, 3),
(30, 10, 'Jadi satu dengan gedung ITS', '2024-05-27 16:45:41.449', 1, 3),
(31, 11, 'Persyaratan pendaftaran adalah:\nSedang menempuh pendidikan Strata 1/Diploma 4 di semester IV, dari semua disiplin ilmu.\nIPK minimum 3.00 pada semester III, serta dapat mempertahankan IPK mini', '2024-05-27 16:45:47.489', 1, 3),
(32, 11, 'persyaratan terrtera di Dokumen', '2024-05-27 16:45:47.489', 1, 3),
(33, 11, 'Detail Persyaratan anda dapat melihat di Website', '2024-05-27 16:45:47.489', 1, 3),
(34, 13, 'Pendaftar melakukan pendaftaran online dan mengisi dengan lengkap formulir pendaftaran (melalui link website resmi yang akan dibuka pada bulan 20 Maret - 27 Mei 2023).\r\nPendaftar akan mendapa', '2024-05-27 16:51:50.574', 1.0000000000000002, 3),
(35, 13, 'Lewat Kantor POS', '2024-05-27 16:51:50.574', 1.0000000000000002, 3),
(36, 13, 'Lewat PENS', '2024-05-27 16:51:50.574', 1.0000000000000002, 3),
(37, 14, 'Program Djarum Beasiswa Plus dibuka 1 kali dalam 1 tahun untuk mahasiswa yang tengah menempuh semester IV. Untuk tahun 2023, pendaftaran akan dibuka pada 20 Maret 2023 dan ditutup pada 27 Mei', '2024-05-27 16:53:16.080', 1.0000000000000002, 3.3),
(38, 14, 'Dibuka setaun sekali', '2024-05-27 16:53:16.080', 1.0000000000000002, 3),
(39, 14, 'Awal awal Januari', '2024-05-27 16:53:16.080', 1.0000000000000002, 3),
(40, 15, 'Kegiatan mahasiswa PENS yang meliputi penalaran adalah GEMASTIK, NPEO, PKM dan MAWAPRES', '2024-05-27 16:53:21.931', 1, 3.3),
(41, 15, 'Mulai dari UKM, Organsiasi hingga robot', '2024-05-27 16:53:21.931', 1, 3),
(42, 15, 'Ada banyak sekali', '2024-05-27 16:53:21.931', 1, 3),
(43, 16, 'Futsal, Volly , Tari, Tenis Meja, Badminton, Tae Kwon Do, Basket dan ENT', '2024-05-27 16:53:28.682', 1, 3.3),
(44, 16, 'Ada banyak mulai olahraga hingga nalar', '2024-05-27 16:53:28.682', 1, 3),
(45, 16, 'Menalar dan olahraga', '2024-05-27 16:53:28.682', 1, 3),
(46, 17, 'Badan Eksekutif Mahasiswa, Dewan Perwakilan Mahasiswa, Dewan Konstitusi Mahasiswa, Lembaga Minat Bakat, Unit Kegiatan Kerohanian Islam, Unit Kegiatan Kristen Katolik dan Himpunan Mahasiswa', '2024-05-27 16:53:33.352', 1, 3.3),
(47, 17, 'Organisasi Mahasiswa hima dan bem', '2024-05-27 16:53:33.352', 1, 3),
(48, 17, 'ada banyak', '2024-05-27 16:53:33.352', 1, 3),
(49, 18, 'Mulai dari beasiswa, Konsultasi Psikologi dan Klinik untuk dokter umum.', '2024-05-27 16:53:38.015', 1.0000000000000002, 3.63),
(50, 18, 'Bidang akademis', '2024-05-27 16:53:38.015', 1.0000000000000002, 3),
(51, 18, 'Bidang kemahsiswaan', '2024-05-27 16:53:38.015', 1.0000000000000002, 3),
(52, 19, '\r\nBeasiswa adalah bentuk bantuan keuangan yang diberikan kepada seseorang untuk membantu membiayai pendidikannya. Beasiswa bisa diberikan oleh pemerintah, lembaga pendidikan, organisasi non-p', '2024-05-27 16:53:59.566', 1, 3),
(53, 19, 'beasiswa adalah untuk meringankan mahasiswa', '2024-05-27 16:53:59.566', 1, 3),
(54, 19, 'Beasiswa adalah bantuan mahasiswa', '2024-05-27 16:53:59.566', 1, 3),
(55, 20, 'Politeknik Elektronika Negeri Surabaya atau yang biasa dikenal dengan PENS merupakan salah satu perguruan tinggi negeri terbaik yang menyelenggarakan pendidikan vokasi pada bidang elektronik ', '2024-05-27 16:54:04.544', 1, 3.993),
(56, 20, 'Kampus Vokasi', '2024-05-27 16:54:04.544', 1, 0.2999999999999998),
(57, 20, 'Kampus nomor 1', '2024-05-27 16:54:04.544', 1, 0.2999999999999998),
(58, 21, 'organisasi mahasiswa tingkat jurusan atau program studi yang memiliki tujuan sama dengan perguruan tinggi untuk mengembangkan minat bakat mahasiswa baik di bidang akademik maupun non akademik', '2024-05-27 16:54:08.618', 1.0000000000000002, 3.63),
(59, 21, 'Organisasi Mahasiswa', '2024-05-27 16:54:08.618', 1.0000000000000002, 3),
(60, 21, 'organissasi untuk pengembangan diri ', '2024-05-27 16:54:08.618', 1.0000000000000002, 3),
(61, 22, 'Kompetisi Mahasiswa Informatika Politeknik Nasional KMIPN merupakan ajang bergengsi untuk Politeknik se-Indonesia di bidang Informatika. Pada tahun ini KMIPN akan dilaksanakan di Jakarta.', '2024-05-27 16:54:12.627', 1, 3.63),
(62, 22, 'Lomba Teknologi', '2024-05-27 16:54:12.627', 1, 3),
(63, 22, 'Lomba untuk mahasiswa', '2024-05-27 16:54:12.627', 1, 3),
(64, 23, 'GEMASTIK atau Pagelaran Mahasiswa Nasional Bidang Teknologi Informasi dan Komunikasi, merupakan program Pusat Prestasi Nasional, Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi', '2024-05-27 16:54:16.150', 1, 3.3),
(65, 23, 'Lomba diadakan kemendikbud', '2024-05-27 16:54:16.150', 1, 3),
(66, 23, 'Lomba tahunan yang diadakan pens', '2024-05-27 16:54:16.150', 1, 3),
(67, 24, 'Program Mahasiswa Wirausaha PMW merupakan program yang diperuntukkan bagi mahasiswa dalam menciptakan aktivitas usaha.', '2024-05-27 16:54:19.415', 1, 3.3),
(68, 24, 'Program wirausaha mahasiswa', '2024-05-27 16:54:19.415', 1, 3),
(69, 24, 'Lomba kewirausahaan', '2024-05-27 16:54:19.415', 1, 3),
(70, 25, 'Program Pembinaan Mahasiswa Wirausaha P2MW adalah program pengembangan usaha mahasiswa yang telah memiliki usaha. Program ini memberikan bantuan dana pengembangan dan pembinaan dengan melakuk', '2024-05-27 16:54:23.299', 1, 3.63),
(71, 25, 'Program kewirausahaan', '2024-05-27 16:54:23.299', 1, 3),
(72, 25, 'Lomba mahasiswa', '2024-05-27 16:54:23.299', 1, 3),
(73, 26, 'Kompetisi Mahasiswa Informatika Politeknik Nasional KMIPN VI akan dilaksanakan pada 1�3 Juli 2024. Pendaftaran dan pengajuan proposal akan dilakukan secara daring pada 22 Maret�13 Mei 2024', '2024-05-27 16:54:29.629', 1, 3.63),
(74, 26, 'Setiap Tahun', '2024-05-27 16:54:29.629', 1, 3),
(75, 26, 'diadakan tahun 2024', '2024-05-27 16:54:29.629', 1, 3),
(76, 27, 'Tahap 1, Pendaftaran Perguruan Tinggi : 12 Mei - 10 Juli 2023 \r\n khusus Divisi II Keamanan Siber : 12 Mei  7 Juli 2023 \r\nTahap 2, pendaftaran tim untuk semua divisi lomba : 12 Mei - 10 Juli 2', '2024-05-27 16:55:05.305', 1, 3),
(77, 27, 'Serangkaian Tahun 2024 ', '2024-05-27 16:55:05.305', 1, 3),
(78, 27, 'Lebih lengkap ada di guidebook', '2024-05-27 16:55:05.305', 1, 3),
(79, 28, 'Sosialisasi : 31 Maret 2023\r\nPengumpulan Proposal : 17 April 2023\r\nPengumuman Proposal Lolos Seleksi : 05 Mei 2023\r\nPresentasi Proposal : 08 Mei 2023\r\nPengumuman Tim Lolos Seleksi : 12 Mei 20', '2024-05-27 16:55:12.857', 1, 3),
(80, 28, 'Diadakan Tahun 2024 ', '2024-05-27 16:55:12.857', 1, 3),
(81, 28, 'Selengkapnya diumumkan oleh kemahasiswaan', '2024-05-27 16:55:12.857', 1, 3),
(82, 28, 'Sosialisasi : 31 Maret 2023\r\nPengumpulan Proposal : 17 April 2023\r\nPengumuman Proposal Lolos Seleksi : 05 Mei 2023\r\nPresentasi Proposal : 08 Mei 2023\r\nPengumuman Tim Lolos Seleksi : 12 Mei 20', '2024-05-27 16:57:21.099', 1, 3),
(83, 30, 'Pendaftaran Paling Lambat 8 Maret 2024', '2024-05-27 16:57:29.663', 1, 3),
(84, 30, 'Selengkapnya ada di kemahasiswaan', '2024-05-27 16:57:29.663', 1, 3),
(85, 30, 'akan diinformasikan melalui kemahasiswaan dan pens blast', '2024-05-27 16:57:29.663', 1, 3),
(86, 31, 'Terdapat GEMASTIK,KMIPN,PORSENI,PMW,P2MW', '2024-05-27 16:57:35.560', 1.0000000000000002, 3.3),
(87, 31, 'Selengkapnya ada di kemahasiswaan', '2024-05-27 16:57:35.560', 1.0000000000000002, 3),
(88, 31, 'ada beberapa yang diadakan', '2024-05-27 16:57:35.560', 1.0000000000000002, 3),
(89, 32, 'Kampus Merdeka merupakan kebijakan Menteri Pendidikan dan Kebudayaan Nadiem Makarim yang membebaskan mahasiswa untuk mengikuti kegiatan di luar program studinya selama 1 semester atau setara ', '2024-05-27 16:57:41.826', 1.0000000000000002, 3.3),
(90, 32, 'Program Dari Kemendikbud', '2024-05-27 16:57:41.826', 1.0000000000000002, 3),
(91, 32, 'Program dari kemendikbud', '2024-05-27 16:57:41.826', 1.0000000000000002, 3),
(92, 33, 'Terdapat program Kampus Merdeka dari Kemendikbud, Kemensos, serta BUMN', '2024-05-27 16:57:48.253', 1, 4.3923),
(93, 33, 'MSIB dan SI ', '2024-05-27 16:57:48.253', 1, 3),
(94, 33, 'Magang saja', '2024-05-27 16:57:48.253', 1, 3),
(95, 34, 'Badan Eksekutif Mahasiswa BEM adalah organisasi mahasiswa intra kampus yang merupakan lembaga eksekutif di tingkat Universitas atau Institusi. Dalam melaksanakan program � programnya, BEM mem', '2024-05-27 16:57:52.178', 1.0000000000000002, 3.63),
(96, 34, 'Organisasi Mahasiswa', '2024-05-27 16:57:52.178', 1.0000000000000002, 3),
(97, 34, 'Organisasi yang ada di PENS', '2024-05-27 16:57:52.178', 1.0000000000000002, 3),
(98, 35, 'Program Kreativitas Mahasiswa PKM adalah program yang bertujuan untuk meningkatkan mutu mahasiswa di perguruan tinggi. PKM diharapkan dapat meningkatkan soft skill dan kompetensi mahasiswa In', '2024-05-27 16:57:57.209', 1, 3.3),
(99, 35, 'Lomba untuk PENS', '2024-05-27 16:57:57.209', 1, 3),
(100, 35, 'Lomba untuk Mahasiswa PENS', '2024-05-27 16:57:57.209', 1, 3),
(101, 36, 'PKM memiliki lima jenis, yaitu\r\nPKM Penelitian PKMP\r\nPKM-AI\r\nPKM-K Kewirausahaan\r\nPKM-M Pengabdian Masyarakat\r\nPKM-T Teknologi\r\nPKM-KC Karsa Cipta\r\nKapan Pelaksanaan PKM?,Pelaksanaan PKM seme', '2024-05-27 16:58:00.890', 1, 3),
(102, 36, 'Banyak Sekali', '2024-05-27 16:58:00.890', 1, 3),
(103, 36, 'ada 5 jenis', '2024-05-27 16:58:00.890', 1, 3),
(104, 37, 'Untuk mendaftar di Politeknik Elektronika Negeri Surabaya (PENS), Anda perlu mengikuti prosedur pendaftaran yang telah ditetapkan oleh lembaga tersebut. Berikut langkah-langkah umum untuk men', '2024-05-27 16:58:04.337', 1.0000000000000002, 3.63),
(105, 37, 'Lewat SBMPTN', '2024-05-27 16:58:04.337', 1.0000000000000002, 3),
(106, 37, 'Lewat SNMPTN', '2024-05-27 16:58:04.337', 1.0000000000000002, 3),
(107, 38, 'Politeknik Elektronika Negeri Surabaya (PENS) menawarkan berbagai program studi atau jurusan dalam bidang teknik elektronika dan informatika. Beberapa program studi yang tersedia di PENS anta', '2024-05-27 16:58:08.155', 1, 3.3),
(108, 38, 'Ada 5 jurusan', '2024-05-27 16:58:08.155', 1, 3),
(109, 38, 'Ada banyak fakultas', '2024-05-27 16:58:08.155', 1, 3),
(110, 39, 'Pendaftaran KMIPN biasanya dilakukan secara daring melalui platform resmi acara tersebut. Calon peserta perlu mengisi formulir pendaftaran dan mengajukan proposal proyek mereka sesuai dengan ', '2024-05-27 16:58:11.478', 1.0000000000000002, 3.993),
(111, 39, 'Lewat form yang disediakan', '2024-05-27 16:58:11.478', 1.0000000000000002, 3),
(112, 39, 'Lewat kemahasiswaan', '2024-05-27 16:58:11.478', 1.0000000000000002, 3),
(113, 40, 'PORSENI adalah singkatan dari Pekan Olahraga dan Seni yang merupakan rangkaian kegiatan kompetisi olahraga dan seni antar mahasiswa Politeknik seluruh indonesia', '2024-05-27 16:58:16.282', 1, 3.63),
(114, 40, 'Lomba Olahraga', '2024-05-27 16:58:16.282', 1, 3),
(115, 40, 'Lomba Kebugaran', '2024-05-27 16:58:16.282', 1, 3),
(116, 41, 'Mahasiswa dapat mengakses melalui online.mis.pens.ac.id', '2024-05-27 16:58:21.734', 1, 3.993),
(117, 41, 'Lewat Website terterra', '2024-05-27 16:58:21.734', 1, 3),
(118, 41, 'melihat melalui website tertenty', '2024-05-27 16:58:21.734', 1, 3),
(119, 42, 'PENS adalah singkatan dari Politeknik Elektronika Negeri Surabaya. PENS adalah institusi pendidikan tinggi dengan bidang keahlian yang meliputi Teknik Informatika, Teknik Elektro Industri, Te', '2024-05-27 16:58:25.679', 1, 5.8461513),
(120, 42, 'Politeknik ITS', '2024-05-27 16:58:25.679', 1, 0.2999999999999998),
(121, 42, 'Kampus Vokasi', '2024-05-27 16:58:25.679', 1, 0.2999999999999998),
(122, 43, 'Awal sejarah PENS dimulai pada tahun 1985. Saat itu, tim studi awal Japan International Cooperation Agency (JICA) untuk bantuan dan kerjasama teknik yang dikepalai oleh Prof. Y. Naito dari To', '2024-05-27 16:58:29.857', 1, 4.3923),
(123, 43, 'Didirikan oleh Orang Jepang', '2024-05-27 16:58:29.857', 1, 3),
(124, 43, 'Didirikan kerjasama jepang', '2024-05-27 16:58:29.857', 1, 3),
(125, 44, '26 ruang kelas dan lebih dari 44 laboratorium atau studio\r\nHall\r\nTheater\r\nAuditorium\r\nPerpustakaan (D3 dan D4)\r\nRuang Sidang\r\nRuang Virtual Conference\r\nRuang Tugas Akhir untuk masing-masing j', '2024-05-27 16:58:34.927', 1, 3),
(126, 44, 'Ada banyak tergantung jurusan yang dipilih', '2024-05-27 16:58:34.927', 1, 3),
(127, 44, 'ada banyak', '2024-05-27 16:58:34.927', 1, 3),
(128, 46, 'HIMAELKA, HIMATELKOM, HIMIT, HIMAELIN, HIMAMMB, HIMAENERGI, dan HMCE', '2024-05-27 16:59:47.404', 1, 4.3923),
(129, 46, 'Secara lengkap ada di website kemahasiswaan PENS', '2024-05-27 16:59:47.404', 1, 3),
(130, 46, 'Ada vanyak macamnya', '2024-05-27 16:59:47.404', 1, 3),
(131, 47, '\"PENS Rider\" Komunitas Pengendara Motor\r\n\"Janaka\" (Japanese nakama), Komunitas Pecinta Budaya Jepang[30]\r\nKomunitas Tim Medis[31]\r\nKomunitas Debat Bahasa Inggris\r\nKomunitas PENS Reggae\r\n\"Game', '2024-05-27 16:59:59.267', 1, 3),
(132, 47, 'Selengkapnya ada di kemahasiswaan', '2024-05-27 16:59:59.267', 1, 3),
(133, 47, 'Ada banyak  sekali', '2024-05-27 16:59:59.267', 1, 3),
(134, 48, '2023\nJUARA UMUM KONTES ROBOT INDONESIA 2023\nKRI NASIONAL 2023 JUARA 1 KRSTI ( ERISA )\nKRI NASIONAL 2023 JUARA 1 KRSBI-B ( ERSOW )\nKRI NASIONAL 2023 JUARA 1 KRAI ( EIRA )\nKRI NASIONAL 2023 JUA', '2024-05-27 17:00:03.741', 1, 3),
(135, 48, 'Kebanaykan juara robot', '2024-05-27 17:00:03.741', 1, 3),
(136, 48, 'Ada banyak sekali dan dapat dicheck di website tertentu', '2024-05-27 17:00:03.741', 1, 3),
(137, 49, 'Politeknik Elektronika Negeri Surabaya merupakan perguruan tinggi politeknik yang meniru gaya pembelajaran Jepang dengan jumlah SKS dan praktikum yang tinggi di setiap semesternya. Untuk seme', '2024-05-27 17:00:08.321', 1, 5.314683),
(138, 49, 'Asik Sekali', '2024-05-27 17:00:08.321', 1, 3),
(139, 49, 'keren', '2024-05-27 17:00:08.321', 1, 3);

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
(1, 'apa itu bidikmisi?', '2024-05-27 16:32:12.719'),
(2, 'apa tujuan bidikmisi?', '2024-05-27 16:41:10.683'),
(3, 'Bagaimana pembayaran untuk mahasiswa yang berbeasiswa Bidikmisi?', '2024-05-27 16:43:04.643'),
(4, 'Saya sudah melakukan pembayaran SPP/UKT, Ikoma. Apakah sudah otomatis daftar ulang?', '2024-05-27 16:44:43.275'),
(5, 'Berapakah jumlah harus dibayarkan untuk UKT/SPP dan IKOMA?', '2024-05-27 16:44:54.212'),
(6, 'Pembayaran melalui ATM bersama apakah bisa? ', '2024-05-27 16:45:09.012'),
(7, 'Bagaimana cara untuk mendaftar beasiswa Adaro ?', '2024-05-27 16:45:14.850'),
(8, 'Bagaimana cara mendaftar beasiwa Djarum Plus?', '2024-05-27 16:45:23.387'),
(9, 'Dimana saya bisa menemukan Kemahasiswaan PENS ? ', '2024-05-27 16:45:31.318'),
(10, 'PENS berlokasi dimana?', '2024-05-27 16:45:41.168'),
(11, 'Apa Persyaratan pendaftaran beasiswa Djarum beasiswaplus', '2024-05-27 16:45:47.018'),
(13, 'Bagaimana Alur pendaftaran Djarum Beasiswa Plus', '2024-05-27 16:51:50.367'),
(14, 'Kapan Pendaftaran Beasiswa Djarum beasiswa Plus dibuka?', '2024-05-27 16:53:15.849'),
(15, 'Kegiatan penalaran apa saja yang ada di PENS?', '2024-05-27 16:53:21.619'),
(16, 'Kegiatan UKM apa saja yang ada di PENS?', '2024-05-27 16:53:28.496'),
(17, 'Organisasi Mahasiswa apa saja yang ada di PENS?', '2024-05-27 16:53:33.155'),
(18, 'Apa saja kemahasiswaan PENS di bidang Kesejahteraan ?', '2024-05-27 16:53:37.829'),
(19, 'apa itu beasiswa', '2024-05-27 16:53:59.492'),
(20, 'apa itu Politeknik Elektronika Negeri Surabaya?', '2024-05-27 16:54:04.330'),
(21, 'Apa itu Himpunan Mahasiswa ?', '2024-05-27 16:54:08.570'),
(22, 'Apa itu KMIPN?', '2024-05-27 16:54:12.433'),
(23, 'Apa itu Gemastik?', '2024-05-27 16:54:15.917'),
(24, 'Apa itu PMW?', '2024-05-27 16:54:19.365'),
(25, 'Apa itu PWMV ?', '2024-05-27 16:54:23.039'),
(26, 'Kapan KMIPN dilaksanakan?', '2024-05-27 16:54:29.353'),
(27, 'Kapan Periode Gemastik dilaksanakan?', '2024-05-27 16:55:05.081'),
(28, 'Kapan Periode PMW dilakukan?', '2024-05-27 16:55:12.807'),
(30, 'Kapan Pendaftaran PMW?', '2024-05-27 16:57:29.589'),
(31, 'Apa saja Perlombaan yang diadakan tahun ini?', '2024-05-27 16:57:35.486'),
(32, 'Apa itu Kampus Merdeka?', '2024-05-27 16:57:41.784'),
(33, 'Apa saja program Kampus Merdeka?', '2024-05-27 16:57:48.213'),
(34, 'Apa yang dimaksud dengan Badan Eksekutif Mahasiswa (BEM) di PENS?', '2024-05-27 16:57:52.009'),
(35, 'Apa itu PKM?', '2024-05-27 16:57:57.165'),
(36, 'Apa saja jenis PKM ?', '2024-05-27 16:58:00.837'),
(37, 'Bagaimana cara mendaftar PENS?', '2024-05-27 16:58:04.152'),
(38, 'Terdapat Jurusan apa saja di PENS?', '2024-05-27 16:58:07.977'),
(39, 'Bagaimana cara pendaftaran untuk Kompetisi Mahasiswa Informatika Politeknik Nasional (KMIPN)?', '2024-05-27 16:58:11.269'),
(40, 'Apa yang dimaksud dengan PORSENI di PENS?', '2024-05-27 16:58:15.774'),
(41, 'Cara melihat jadwal Kuliah Secara Online di PENS?', '2024-05-27 16:58:21.321'),
(42, 'Apa itu PENS?', '2024-05-27 16:58:25.508'),
(43, 'Sejarah PENS?', '2024-05-27 16:58:29.690'),
(44, 'Fasilitas PENS?', '2024-05-27 16:58:34.744'),
(46, 'Himpunan yang ada di PENS?', '2024-05-27 16:59:47.229'),
(47, 'Komunitas yang ada di PENS?', '2024-05-27 16:59:59.113'),
(48, 'Prestasi PENS di bidang robotika?', '2024-05-27 17:00:03.586'),
(49, 'Bagaimana Kegiatan Belajar di PENS?', '2024-05-27 17:00:08.152'),
(51, 'Kapan Periode Gemastik dilaksanakan?', '2024-06-11 20:04:34.152');

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
('6db71ff1-5449-4466-9625-e4908b541da2', 'c777551c13154b5267fc603c198f10690c3ab2bbba92ab321676daea34cc9924', '2024-05-27 09:26:13.301', '20240517185011_uuid', NULL, NULL, '2024-05-27 09:26:12.638', 1),
('71e86f9e-d219-4cad-bc5e-ef62372728ea', '95ce56c2765106e23c192dd863ddec8dfcb7ebdd88514769fe0d7cea077e38f0', '2024-05-27 09:26:12.538', '20240517151112_', NULL, NULL, '2024-05-27 09:26:10.825', 1),
('c513269f-820e-4669-995f-51b8a3b65fc2', '2279a1f31f9d7a86da0d29a04e90dbc8ebf09b964a3e17c8141cffffbc36462a', '2024-05-27 09:26:10.811', '20240517150828_', NULL, NULL, '2024-05-27 09:26:10.718', 1),
('c72bb8ad-9743-4a3d-9aed-4a823200b9a0', '8d6f1a96e721be6d68cea5024de447f15e771b3b3431dabbeb43436fd25ed2ac', '2024-05-27 09:26:10.705', '20240516141144_', NULL, NULL, '2024-05-27 09:26:10.004', 1),
('f405431c-abb9-4109-a8af-971f4374afb6', 'e9bb57d298eb731e62eb588d10490ffef68c44364cca6dedbc1be59eb2116abf', '2024-05-27 09:26:15.422', '20240517185513_', NULL, NULL, '2024-05-27 09:26:13.326', 1),
('fc3db524-da15-4f78-a544-170941f67201', '6f8dd9c4d9900e5da93b7fa52b91241197562b65b8a4eb1d0d878b58e4fb9e6c', '2024-05-27 09:26:12.626', '20240517184709_', NULL, NULL, '2024-05-27 09:26:12.550', 1);

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
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=179;

--
-- AUTO_INCREMENT for table `pertanyaan`
--
ALTER TABLE `pertanyaan`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=52;

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
