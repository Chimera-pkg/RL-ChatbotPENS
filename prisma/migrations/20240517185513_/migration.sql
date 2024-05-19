/*
  Warnings:

  - The primary key for the `jawaban` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to alter the column `id` on the `jawaban` table. The data in that column could be lost. The data in that column will be cast from `VarChar(191)` to `Int`.
  - You are about to alter the column `pertanyaanId` on the `jawaban` table. The data in that column could be lost. The data in that column will be cast from `VarChar(191)` to `Int`.
  - The primary key for the `pertanyaan` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to alter the column `id` on the `pertanyaan` table. The data in that column could be lost. The data in that column will be cast from `VarChar(191)` to `Int`.

*/
-- DropForeignKey
ALTER TABLE `jawaban` DROP FOREIGN KEY `Jawaban_pertanyaanId_fkey`;

-- AlterTable
ALTER TABLE `jawaban` DROP PRIMARY KEY,
    MODIFY `id` INTEGER NOT NULL AUTO_INCREMENT,
    MODIFY `pertanyaanId` INTEGER NOT NULL,
    ADD PRIMARY KEY (`id`);

-- AlterTable
ALTER TABLE `pertanyaan` DROP PRIMARY KEY,
    MODIFY `id` INTEGER NOT NULL AUTO_INCREMENT,
    ADD PRIMARY KEY (`id`);

-- AddForeignKey
ALTER TABLE `Jawaban` ADD CONSTRAINT `Jawaban_pertanyaanId_fkey` FOREIGN KEY (`pertanyaanId`) REFERENCES `Pertanyaan`(`id`) ON DELETE RESTRICT ON UPDATE CASCADE;
