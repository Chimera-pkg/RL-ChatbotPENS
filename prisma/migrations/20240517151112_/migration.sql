/*
  Warnings:

  - The primary key for the `pertanyaan` table will be changed. If it partially fails, the table could be left without primary key constraint.

*/
-- DropForeignKey
ALTER TABLE `jawaban` DROP FOREIGN KEY `Jawaban_pertanyaanId_fkey`;

-- AlterTable
ALTER TABLE `jawaban` MODIFY `pertanyaanId` VARCHAR(191) NOT NULL;

-- AlterTable
ALTER TABLE `pertanyaan` DROP PRIMARY KEY,
    MODIFY `id` VARCHAR(191) NOT NULL,
    ADD PRIMARY KEY (`id`);

-- AddForeignKey
ALTER TABLE `Jawaban` ADD CONSTRAINT `Jawaban_pertanyaanId_fkey` FOREIGN KEY (`pertanyaanId`) REFERENCES `Pertanyaan`(`id`) ON DELETE RESTRICT ON UPDATE CASCADE;
