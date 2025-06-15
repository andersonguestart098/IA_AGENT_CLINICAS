import asyncio
import json
import numpy as np
from sklearn.preprocessing import normalize
from prisma import Prisma
from models.embeddings import gerar_embedding

async def atualizar_embedding_endereco():
    prisma = Prisma()
    await prisma.connect()

    doc_id = 1  # ID do documento com endereço
    doc = await prisma.knowledgebase.find_unique(where={"id": doc_id})
    if not doc:
        print(f"❌ Documento ID {doc_id} não encontrado.")
        await prisma.disconnect()
        return

    print(f"📄 Conteúdo atual:\n{doc.conteudo}\n")

    try:
        novo_embedding = gerar_embedding(doc.conteudo)
        vetor = np.array(json.loads(novo_embedding), dtype=np.float32).reshape(1, -1)
        vetor = normalize(vetor, axis=1)

        await prisma.knowledgebase.update(
            where={"id": doc_id},
            data={"embedding": json.dumps(novo_embedding)}
        )

        print("✅ Embedding atualizado com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao atualizar embedding: {e}")

    await prisma.disconnect()

asyncio.run(atualizar_embedding_endereco())
