from sentence_transformers import SentenceTransformer
import json

# Força o uso da CPU
modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

def gerar_embedding(texto: str) -> str:
    texto = texto.strip()[:8192]
    vetor = modelo.encode(texto).tolist()
    return json.dumps(vetor)
