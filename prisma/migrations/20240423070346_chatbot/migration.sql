-- CreateTable
CREATE TABLE `Pertanyaan` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `pertanyaan` VARCHAR(191) NOT NULL,

    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- CreateTable
CREATE TABLE `Jawaban` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `id_pertanyaan` INTEGER NOT NULL,
    `jawaban` VARCHAR(191) NOT NULL,
    `score` DOUBLE NOT NULL,

    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
