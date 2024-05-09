/*
  Warnings:

  - Added the required column `cosine` to the `Jawaban` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE `jawaban` ADD COLUMN `cosine` DOUBLE NOT NULL;
