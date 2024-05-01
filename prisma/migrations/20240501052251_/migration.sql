/*
  Warnings:

  - You are about to drop the column `id_pertanyaan` on the `jawaban` table. All the data in the column will be lost.
  - You are about to drop the column `score` on the `jawaban` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE `jawaban` DROP COLUMN `id_pertanyaan`,
    DROP COLUMN `score`,
    ADD COLUMN `createdAt` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3);
