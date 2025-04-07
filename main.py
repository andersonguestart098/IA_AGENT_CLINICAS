import os
import traceback
import logging
import sys
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from llama_cpp import Llama
from prisma import Prisma
from datetime import datetime
from pathlib import Path
import pytesseract
from PIL import Image
from io import BytesIO
import mimetypes
import pdfplumber
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.embeddings import gerar_embedding
from fastapi.responses import HTMLResponse


# Carregar variáveis de ambiente
load_dotenv()

# Configurar o Prisma
prisma = Prisma(auto_register=True)

# Configurar o DATABASE_URL diretamente (ajuste o caminho se necessário)
os.environ["DATABASE_URL"] = "file:D:/IA2/CEMEAR-IA-FIX/prisma/dev.db"

# Criar diretórios
Path("uploads").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Configurar o Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configurar logging com UTF-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/cemear.log", encoding="utf-8"),
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))
    ]
)
logger = logging.getLogger(__name__)

# Inicializar o FastAPI
app = FastAPI()

# Inicializar o LLM
llm = Llama(
    model_path="D:/huggingface_cache/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=35,
    n_threads=16,
    use_mlock=True,
    verbose=True,
)

# Função para extrair texto de arquivos
async def extrair_texto(content: bytes, file: UploadFile) -> str:
    texto = ""
    file_ext = Path(file.filename).suffix.lower()
    content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or ""

    if content_type.startswith("image/") or file_ext in [".png", ".jpg", ".jpeg"]:
        image = Image.open(BytesIO(content))
        texto = pytesseract.image_to_string(image)
    elif content_type == "application/pdf" or file_ext == ".pdf":
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages:
                texto += page.extract_text() or ""
        if not texto.strip():
            images = convert_from_bytes(content)
            for image in images:
                texto += pytesseract.image_to_string(image)
    return texto

# Eventos de inicialização e encerramento
@app.on_event("startup")
async def startup():
    await prisma.connect()
    logger.info("✅ Prisma conectado com sucesso.")
    try:
        logger.info(f"📂 Banco em uso: {os.environ['DATABASE_URL']}")
    except Exception:
        logger.warning("⚠️ Não foi possível identificar o caminho do banco.")

@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()
    logger.info("🛑 Prisma desconectado.")

# Modelo para requisições de chat
class PromptRequest(BaseModel):
    question: str


# Endpoint de chat
@app.post("/chat")
async def chat_endpoint(data: PromptRequest):
    try:
        logger.info(f"🤖 Pergunta recebida no /chat: {data.question}")

        # Buscar vetores salvos com embedding
        todos = await prisma.knowledgebase.find_many(where={"embedding": {"not": None}})
        embedding_pergunta = np.array(json.loads(gerar_embedding(data.question))).reshape(1, -1)

        # Calcular similaridade entre a pergunta e os vetores
        relevantes = []
        for k in todos:
            try:
                vetor = np.array(json.loads(k.embedding)).reshape(1, -1)
                score = float(cosine_similarity(embedding_pergunta, vetor)[0][0])
                relevantes.append((k, score))
            except Exception as e:
                logger.warning(f"⚠️ Erro ao calcular similaridade com ID {k.id}: {e}")
                continue

        # Top 5 mais relevantes
        top = sorted(relevantes, key=lambda x: x[1], reverse=True)[:5]
        conteudos_unicos = list(dict.fromkeys(k.conteudo.strip()[:500] for k, _ in top))
        contexto_base = "\n\n".join(f"- {c}" for c in conteudos_unicos)

        logger.info("🔎 Top 5 similaridades:")
        for k, score in top:
            logger.info(f"🧩 ID: {k.id} | Score: {round(score, 4)} | Origem: {k.origem}")

        # Prompt com instruções reforçadas e sinal de fim
        prompt = f"""
Você é um assistente técnico institucional da empresa Cemear. Sua função é fornecer respostas objetivas, técnicas e detalhadas com base nos documentos oficiais abaixo.

🧠 Base de Conhecimento:
{contexto_base}

📩 Pergunta do usuário:
{data.question}

🎯 Instruções:
- Responda de forma técnica, precisa e baseada exclusivamente no conteúdo fornecido.
- Nunca repita a pergunta ou resposta.
- Não gere múltiplas versões da mesma resposta.
- Responda em **uma única vez**, com no máximo 6 parágrafos.
- Nunca invente informações. Se não souber, diga: "Não encontrei essa informação nos documentos técnicos."
- 🚫 Finalize sua resposta com a frase: **"Fim da resposta."**

✍️ Redija agora a resposta com base nas instruções acima:
"""

        resposta_raw = llm(
            prompt,
            max_tokens=2048,
            temperature=0.4,
            top_p=0.8,
            repeat_penalty=1.8,  # penalização forte para evitar repetição
            presence_penalty=1.0
        )["choices"][0]["text"].strip()

        # Pós-processamento: remover linhas duplicadas
        linhas_unicas = list(dict.fromkeys(resposta_raw.splitlines()))
        resposta_final = "\n".join(linhas_unicas).strip()

        logger.info(f"🧠 Resposta gerada: {resposta_final}")
        return {"answer": resposta_final}

    except Exception:
        logger.error("Erro no LLM: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro ao processar a pergunta.")



# Endpoint para upload de conhecimento
@app.post("/upload")
async def upload_conhecimento(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_ext = Path(file.filename).suffix.lower()
        texto = content.decode("utf-8", errors="ignore") if file_ext == ".txt" else await extrair_texto(content, file)

        if not texto.strip():
            raise HTTPException(status_code=422, detail="Não foi possível extrair texto do arquivo.")

        embedding_json = gerar_embedding(texto)  # <- string JSON com o vetor

        kb = await prisma.knowledgebase.create({
            "origem": file.filename,
            "conteudo": texto,
            "embedding": embedding_json
        })

        return {"status": "sucesso", "resumo": texto[:200], "id": kb.id}
    except Exception:
        logger.error("Erro ao salvar conhecimento: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno ao processar o upload.")


# Endpoint para upload de planta
@app.post("/upload-planta")
async def upload_planta(file: UploadFile = File(...), contexto: str = Form("")):
    try:
        content = await file.read()
        texto = await extrair_texto(content, file)
        if not texto.strip():
            raise HTTPException(status_code=422, detail="Não foi possível extrair texto da planta.")

        largura, altura = 5.0, 3.0  # Valores fixos como exemplo
        area = largura * altura
        materiais = {
            "area_total_m2": round(area, 2),
            "perimetro_total_m": round(2 * (largura + altura), 2),
            "montantes": int(area * 2),
            "guias": int(largura * 2),
            "placas_gesso": int(area / 1.8),
            "parafusos": int(area * 15),
            "fitas": int(area / 50),
            "massa": int(area / 20)
        }

        prompt = f"""
Você é um engenheiro da Cemear.

Contexto:
{contexto}

Texto da planta:
{texto[:1200]}...

Gere uma resposta técnica com materiais estimados.
"""
        resposta = llm(prompt, max_tokens=1024, temperature=0.7, top_p=0.9)["choices"][0]["text"].strip()
        kb = await prisma.knowledgebase.create({"origem": file.filename, "conteudo": texto})

        return {
            "status": "sucesso",
            "materiais_estimados": materiais,
            "medidas_detectadas": {"largura_metros": largura, "altura_metros": altura},
            "resposta_ia": resposta,
            "resumo": texto[:200],
            "knowledgeBaseId": kb.id
        }
    except Exception:
        logger.error("Erro ao processar planta: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno ao processar a planta.")

# Endpoint para interpretação de upload
@app.post("/upload-interpreta")
async def upload_interpreta(file: UploadFile = File(...), contexto: str = Form("")):
    try:
        content = await file.read()
        texto = await extrair_texto(content, file)
        if not texto.strip() and not contexto.strip():
            raise HTTPException(status_code=422, detail="Não foi possível interpretar nada.")

        prompt = (
            f"O usuário disse: {contexto}\nArquivo: {texto[:1200]}" 
            if texto and contexto 
            else (f"Conteúdo do arquivo: {texto[:1200]}" if texto else f"Contexto do usuário: {contexto}")
        )
        resposta = llm(prompt, max_tokens=1024, temperature=0.7, top_p=0.9)["choices"][0]["text"].strip()

        return {"status": "sucesso", "resposta_ia": resposta, "resumo": texto[:200] if texto else contexto[:200]}
    except Exception:
        logger.error("Erro ao interpretar: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno ao interpretar.")
    
@app.post("/calcular-materiais")
async def calcular_materiais_manualmente(
    area: float = Form(...),
    perimetro: float = Form(...),
    altura_rebaixo: float = Form(0.4),
    contexto: str = Form("")
):
    from math import ceil

    try:
        logger.info(f"📐 Cálculo manual solicitado - Área: {area}, Perímetro: {perimetro}, Rebaixo: {altura_rebaixo}")
        logger.info(f"📝 Contexto: {contexto}")

        # Cálculos técnicos
        f530 = (area * 1.8) / 3
        massa = area * 0.55
        fita = (area * 1.5) / 150
        gn25 = area * 16
        cantoneira = (perimetro * 1.05) / 3
        pendural = f530 * 3
        arame10 = (pendural * altura_rebaixo) / 14
        parafuso_bucha = (cantoneira * 6) + pendural
        parafuso_13mm_pa = f530 * 2

        # Resposta com base no contexto
        contexto_lower = contexto.lower()

        if "f530" in contexto_lower or "montante" in contexto_lower:
            resposta = f"Com base na área e perímetro fornecidos, a quantidade estimada de montantes F530 é de {ceil(f530)} barras de 3 metros."
        elif "massa" in contexto_lower:
            resposta = f"Com base na área fornecida, a quantidade estimada de massa é de {massa:.2f} kg."
        elif "fita" in contexto_lower:
            resposta = f"Com base na área fornecida, a quantidade estimada de fita de papel é de {ceil(fita)} rolos."
        elif "gn25" in contexto_lower:
            resposta = f"Com base na área fornecida, a quantidade estimada de parafusos GN25 é de {int(gn25)} unidades."
        elif "cantoneira" in contexto_lower:
            resposta = f"Com base no perímetro fornecido, a quantidade estimada de cantoneiras tabica é de {ceil(cantoneira)} barras de 3 metros."
        elif "pendural" in contexto_lower:
            resposta = f"Com base na área fornecida, a quantidade estimada de pendurais é de {ceil(pendural)} unidades."
        elif "arame" in contexto_lower:
            resposta = f"Com base na área e altura do rebaixo, a quantidade estimada de arame 10 é de {arame10:.2f} kg."
        elif "bucha" in contexto_lower:
            resposta = f"Com base nos cálculos, a quantidade estimada de parafusos com bucha é de {ceil(parafuso_bucha)} unidades."
        elif "parafuso 13mm" in contexto_lower or "pa" in contexto_lower:
            resposta = f"Com base na área fornecida, a quantidade estimada de parafusos 13mm PA é de {ceil(parafuso_13mm_pa)} unidades."
        else:
            resposta = "Não encontrei essa informação nos documentos técnicos."

        return {
            "status": "sucesso",
            "resposta_ia": resposta
        }

    except Exception as e:
        logger.error(f"Erro no cálculo manual com RAG: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao calcular com base no banco de conhecimento.")



# Modelo para requisições de feedback
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str
    contextoUsuario: str = ""
    origemPlanta: str = ""
    knowledgeBaseId: int | None = None

# Endpoint para receber feedback
@app.post("/feedback")
async def receber_feedback(data: FeedbackRequest):
    try:
        # Prompt para classificar feedback com LLaMA
        prompt_classificacao = f"""
Classifique o feedback a seguir com base nas regras:

Feedback do usuário:
"{data.feedback}"

Regras:
- Se o feedback indicar que a resposta foi correta, diga apenas: ACERTO
- Se o feedback indicar erro ou insatisfação, diga apenas: ERRO
- Se for neutro, elogio genérico ou não der pra saber, diga apenas: NEUTRO

IMPORTANTE: Retorne SOMENTE uma destas palavras em MAIÚSCULO: ACERTO, ERRO ou NEUTRO. Sem explicações, sem pontuação, sem prefixo.
"""

        # Chamada ao LLaMA com temperatura 0 (determinístico)
        classificacao_raw = llm(
            prompt_classificacao,
            max_tokens=5,
            temperature=0.0
        )["choices"][0]["text"].strip().upper()

        # Sanitização da saída para detectar a intenção correta
        if "ACERTO" in classificacao_raw:
            acerto = True
            classificacao_final = "ACERTO"
        elif "ERRO" in classificacao_raw:
            acerto = False
            classificacao_final = "ERRO"
        else:
            acerto = False
            classificacao_final = "NEUTRO"

        # Criação do feedback no banco
        created = await prisma.feedback.create({
            "question": data.question,
            "answer": data.answer,
            "feedback": data.feedback,
            "acerto": acerto,
            "contextoUsuario": data.contextoUsuario,
            "origemPlanta": data.origemPlanta,
            "knowledgeBaseId": data.knowledgeBaseId
        })

        logger.info(f"📥 Feedback salvo: ID {created.id} | Classificação: {classificacao_final}")
        return {"status": "ok", "id": created.id, "classificacao": classificacao_final}

    except Exception as e:
        logger.error(f"Erro ao salvar feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {str(e)}")



# Endpoint para aprendizado por reforço
@app.post("/reforco")
async def aprendizado_reforco():
    try:
        feedbacks = await prisma.feedback.find_many(where={"acerto": True, "usada_para_treinamento": False})
        contador = 0

        for fb in feedbacks:
            content = f"Pergunta: {fb.question}\nResposta: {fb.answer}"
            try:
                embedding_json = gerar_embedding(content)  # gera vetor como string JSON
                kb = await prisma.knowledgebase.create({
                    "origem": f"feedback:{fb.id}",
                    "conteudo": content,
                    "embedding": embedding_json
                })
                await prisma.feedback.update(
                    where={"id": fb.id},
                    data={"usada_para_treinamento": True, "knowledgeBaseId": kb.id}
                )
                contador += 1
            except Exception as e:
                logger.error(f"Erro ao vetorizar feedback {fb.id}: {str(e)}")
                continue

        logger.info(f"Reforço aplicado em {contador} feedbacks.")
        return {"status": "reforço aplicado", "quantidade": contador}
    except Exception:
        logger.error("Erro no reforço: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro ao aplicar reforço.")

# Endpoint para testar feedbacks
@app.get("/test-feedback")
async def test_feedback():
    try:
        feedbacks = await prisma.feedback.find_many(
            order={"contextoUsuario": "asc"},
            take=100,
            skip=0,
            include={"knowledgeBase": True}
        )
        return {
            "status": "success",
            "data": [
                {
                    "id": f.id,
                    "question": f.question,
                    "answer": f.answer,
                    "feedback": f.feedback,
                    "acerto": f.acerto,
                    "contextoUsuario": f.contextoUsuario,
                    "origemPlanta": f.origemPlanta,
                    "knowledgeBase": f.knowledgeBase.conteudo if f.knowledgeBase else None
                }
                for f in feedbacks
            ]
        }
    except Exception as e:
        logger.error(f"Erro ao buscar feedbacks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar feedbacks: {str(e)}")
    
@app.get("/metrics")
async def gerar_metricas():
    try:
        total = await prisma.feedback.count()
        acertos = await prisma.feedback.count(where={"acerto": True})
        usados = await prisma.feedback.count(where={"usada_para_treinamento": True})

        taxa = round((acertos / total) * 100, 2) if total else 0
        usado_pct = round((usados / total) * 100, 2) if total else 0

        ultima = await prisma.metricas.find_first(order={"criadoEm": "desc"})

        # Calcula a evolução se houver registro anterior
        evolucao_taxa = round(taxa - ultima.taxaAcerto, 2) if ultima else None
        evolucao_usados = round(usado_pct - ultima.percentualUsado, 2) if ultima else None

        # Salva nova métrica
        await prisma.metricas.create({
            "totalFeedbacks": total,
            "acertos": acertos,
            "taxaAcerto": taxa,
            "usadosTreino": usados,
            "percentualUsado": usado_pct
        })

        return {
            "total_feedbacks": total,
            "acertos": acertos,
            "taxa_acerto": taxa,
            "feedbacks_usados": usados,
            "percentual_usado": usado_pct,
            "evolucao_taxa_acerto": evolucao_taxa,
            "evolucao_percentual_usado": evolucao_usados
        }

    except Exception:
        logger.error("Erro ao gerar métricas: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro ao gerar métricas.")
    
@app.get("/dashboard", response_class=HTMLResponse)
async def exibir_dashboard():
    total = await prisma.feedback.count()
    acertos = await prisma.feedback.count(where={"acerto": True})
    usados = await prisma.feedback.count(where={"usada_para_treinamento": True})

    taxa_acerto = round((acertos / total) * 100, 2) if total else 0
    percentual_usado = round((usados / total) * 100, 2) if total else 0

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard IA - Cemear</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100 text-gray-800">
        <div class="container mx-auto p-8">
            <h1 class="text-3xl font-bold mb-4">📊 Dashboard da IA - Cemear</h1>
            <div class="grid grid-cols-2 gap-4">
                <div class="bg-white shadow-md rounded-lg p-6">
                    <h2 class="text-lg font-semibold">Feedbacks totais</h2>
                    <p class="text-2xl">{total}</p>
                </div>
                <div class="bg-white shadow-md rounded-lg p-6">
                    <h2 class="text-lg font-semibold">Feedbacks com acerto</h2>
                    <p class="text-2xl">{acertos} ({taxa_acerto}%)</p>
                </div>
                <div class="bg-white shadow-md rounded-lg p-6">
                    <h2 class="text-lg font-semibold">Usados para treinamento</h2>
                    <p class="text-2xl">{usados} ({percentual_usado}%)</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# Executar a aplicação
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)