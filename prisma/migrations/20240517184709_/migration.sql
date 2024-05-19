/*
  Warnings:

  - You are about to drop the column `jawaban2` on the `jawaban` table. All the data in the column will be lost.
  - You are about to drop the column `jawaban3` on the `jawaban` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE `jawaban` DROP COLUMN `jawaban2`,
    DROP COLUMN `jawaban3`;
