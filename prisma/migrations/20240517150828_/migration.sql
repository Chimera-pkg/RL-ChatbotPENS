/*
  Warnings:

  - Added the required column `jawaban2` to the `Jawaban` table without a default value. This is not possible if the table is not empty.
  - Added the required column `jawaban3` to the `Jawaban` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE `jawaban` ADD COLUMN `jawaban2` VARCHAR(191) NOT NULL,
    ADD COLUMN `jawaban3` VARCHAR(191) NOT NULL;
