/*
  Warnings:

  - You are about to drop the column `pertanyaanId` on the `jawaban` table. All the data in the column will be lost.

*/
-- DropForeignKey
ALTER TABLE `jawaban` DROP FOREIGN KEY `Jawaban_pertanyaanId_fkey`;

-- AlterTable
ALTER TABLE `jawaban` DROP COLUMN `pertanyaanId`;
