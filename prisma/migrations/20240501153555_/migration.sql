/*
  Warnings:

  - Added the required column `score` to the `Jawaban` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE `jawaban` ADD COLUMN `score` DOUBLE NOT NULL;
