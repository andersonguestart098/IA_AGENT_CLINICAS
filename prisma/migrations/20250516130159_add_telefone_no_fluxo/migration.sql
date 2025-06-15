/*
  Warnings:

  - Added the required column `telefone` to the `FluxoConversa` table without a default value. This is not possible if the table is not empty.

*/
-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_FluxoConversa" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "telefone" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "etapaAtual" TEXT NOT NULL,
    "dadosParciais" TEXT NOT NULL,
    "tipoFluxo" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'em_andamento',
    "criadoEm" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "new_FluxoConversa" ("criadoEm", "dadosParciais", "etapaAtual", "id", "sessionId", "status", "tipoFluxo", "userId") SELECT "criadoEm", "dadosParciais", "etapaAtual", "id", "sessionId", "status", "tipoFluxo", "userId" FROM "FluxoConversa";
DROP TABLE "FluxoConversa";
ALTER TABLE "new_FluxoConversa" RENAME TO "FluxoConversa";
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
