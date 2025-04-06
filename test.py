import asyncio
from prisma import Prisma

async def test():
    db = Prisma(auto_register=True)
    try:
        await db.connect()
        print("✅ Conectado com sucesso!")

        # Criar uma entrada em KnowledgeBase
        kb = await db.knowledgebase.create({
            "origem": "Manual da Planta",
            "conteudo": "Informações sobre manutenção de equipamentos."
        })
        print(f"KnowledgeBase criada: ID {kb.id}, Origem: {kb.origem}")

        # Criar um Feedback relacionado ao KnowledgeBase
        feedback = await db.feedback.create({
            "question": "Como fazer a manutenção?",
            "answer": "Siga o manual.",
            "feedback": "Resposta útil",
            "acerto": True,
            "knowledgeBaseId": kb.id,
            "contextoUsuario": "Usuário na linha de produção"
        })
        print(f"Feedback criado: ID {feedback.id}, Pergunta: {feedback.question}")

        # Listar todos os Feedbacks com suas relações
        feedbacks = await db.feedback.find_many(include={"knowledgeBase": True})
        for f in feedbacks:
            kb_info = f.knowledgeBase.origem if f.knowledgeBase else "Sem KB"
            print(f"Feedback ID: {f.id}, Pergunta: {f.question}, KB Origem: {kb_info}")

    except Exception as e:
        print(f"Erro: {e}")
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(test())