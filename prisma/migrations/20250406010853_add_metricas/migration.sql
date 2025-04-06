-- CreateTable
CREATE TABLE "Metricas" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "totalFeedbacks" INTEGER NOT NULL,
    "acertos" INTEGER NOT NULL,
    "taxaAcerto" REAL NOT NULL,
    "usadosTreino" INTEGER NOT NULL,
    "percentualUsado" REAL NOT NULL,
    "criadoEm" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
