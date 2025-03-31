import sqlite3

try:
    conn = sqlite3.connect(r"C:\IA-CEMEAR\IA_CEMEAR\prisma\dev.db")
    print("✅ Conectado com sucesso ao banco SQLite!")
    conn.close()
except Exception as e:
    print(f"❌ Erro ao conectar: {e}")
