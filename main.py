import os
import traceback
import logging
import sys
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from llama_cpp import Llama
from generated import Prisma
from datetime import datetime
from pathlib import Path
import pytesseract
from PIL import Image
from io import BytesIO
import mimetypes
import pdfplumber
from pdf2image import convert_from_bytes
from generated.models import Feedback, KnowledgeBase
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
prisma = Prisma()

import os
os.environ["PRISMA_QUERY_ENGINE_BINARY"] = os.path.abspath("prisma-query-engine-windows.exe")



# Diretórios
Path("uploads").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# OCR (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Logger UTF-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/cemear.log", encoding="utf-8"),
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))
    ]
)
logger = logging.getLogger(__name__)

# App e LLM
app = FastAPI()
prisma = Prisma()

llm = Llama(
    model_path="models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=0,
    n_threads=8,
    use_mlock=True,
    verbose=True,
)

# OCR
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

@app.on_event("startup")
async def startup():
    await prisma.connect()
    logger.info("✅ Prisma conectado com sucesso.")
    try:
        logger.info(f"📂 Banco em uso: {prisma._client._engine._engine_config.database_url}")
    except Exception:
        logger.warning("⚠️ Não foi possível identificar o caminho do banco via Prisma.")

@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()
    logger.info("🛑 Prisma desconectado.")

class PromptRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(data: PromptRequest):
    try:
        logger.info(f"🤖 Pergunta recebida no /chat: {data.question}")
        knowledge = await prisma.knowledgebase.find_many(take=5, order={"timestamp": "desc"})
        contexto_base = "\n\n".join(f"- {k.conteudo.strip()[:500]}" for k in knowledge if k.conteudo.strip())

        prompt = f"""
Você é um assistente técnico da empresa Cemear.
Use somente os dados abaixo para responder a pergunta.

Base de Conhecimento:
{contexto_base}

Pergunta:
{data.question}

Regras:
- Seja técnico, direto e objetivo.
- Se não souber, diga: "Não encontrei essa informação nos documentos."
"""
        resposta = llm(prompt, max_tokens=1024, temperature=0.7, top_p=0.9)["choices"][0]["text"].strip()
        logger.info(f"🧠 Resposta gerada: {resposta}")
        return {"answer": resposta}
    except Exception:
        logger.error("Erro no LLM: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro ao processar a pergunta.")

@app.post("/upload")
async def upload_conhecimento(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_ext = Path(file.filename).suffix.lower()
        texto = content.decode("utf-8", errors="ignore") if file_ext == ".txt" else await extrair_texto(content, file)
        if not texto.strip():
            raise HTTPException(status_code=422, detail="Não foi possível extrair texto do arquivo.")
        await prisma.knowledgebase.create({"origem": file.filename, "conteudo": texto})
        return {"status": "sucesso", "resumo": texto[:200]}
    except Exception:
        logger.error("Erro ao salvar conhecimento: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno ao processar o upload.")

@app.post("/upload-planta")
async def upload_planta(file: UploadFile = File(...), contexto: str = Form("")):
    try:
        content = await file.read()
        texto = await extrair_texto(content, file)
        if not texto.strip():
            raise HTTPException(status_code=422, detail="Não foi possível extrair texto da planta.")

        largura, altura = 5.0, 3.0
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
        await prisma.knowledgebase.create({"origem": file.filename, "conteudo": texto})

        return {
            "status": "sucesso",
            "materiais_estimados": materiais,
            "medidas_detectadas": {"largura_metros": largura, "altura_metros": altura},
            "resposta_ia": resposta,
            "resumo": texto[:200]
        }
    except Exception:
        logger.error("Erro ao processar planta: %s", traceback.format_exc())
        raise HTTPException(status_code=500)

@app.post("/upload-interpreta")
async def upload_interpreta(file: UploadFile = File(...), contexto: str = Form("")):
    try:
        content = await file.read()
        texto = await extrair_texto(content, file)
        if not texto.strip() and not contexto.strip():
            raise HTTPException(status_code=422, detail="Não foi possível interpretar nada.")

        prompt = f"O usuário disse: {contexto}\nArquivo: {texto[:1200]}" if texto and contexto else (f"Conteúdo do arquivo: {texto[:1200]}" if texto else f"Contexto do usuário: {contexto}")
        resposta = llm(prompt, max_tokens=1024, temperature=0.7, top_p=0.9)["choices"][0]["text"].strip()

        return {"status": "sucesso", "resposta_ia": resposta, "resumo": texto[:200] if texto else contexto[:200]}
    except Exception:
        logger.error("Erro ao interpretar: %s", traceback.format_exc())
        raise HTTPException(status_code=500)

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str
    contextoUsuario: str = ""
    origemPlanta: str = ""
    knowledgeBaseId: int | None = None

@app.post("/feedback")
async def receber_feedback(data: FeedbackRequest):
    try:
        created = await prisma.feedback.create({
            "question": data.question,
            "answer": data.answer,
            "feedback": data.feedback,
            "acerto": data.feedback == "certa",
            "contextoUsuario": data.contextoUsuario,
            "origemPlanta": data.origemPlanta,
            "knowledgeBaseId": data.knowledgeBaseId
        })
        return {"status": "ok", "id": created.id}
    except Exception as e:
        logger.error(f"Erro ao salvar feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {str(e)}")

@app.post("/reforco")
async def aprendizado_reforco():
    try:
        feedbacks = await prisma.feedback.find_many(where={"acerto": True, "usada_para_treinamento": False})
        for fb in feedbacks:
            content = f"Pergunta: {fb.question}\nResposta: {fb.answer}"
            kb = await prisma.knowledgebase.create({"origem": f"feedback:{fb.id}", "conteudo": content})
            await prisma.feedback.update(where={"id": fb.id}, data={"usada_para_treinamento": True, "knowledgeBaseId": kb.id})
        return {"status": "reforco aplicado", "quantidade": len(feedbacks)}
    except Exception:
        logger.error("Erro no reforço: %s", traceback.format_exc())
        raise HTTPException(status_code=500)

@app.get("/test-feedback")
async def test_feedback():
    try:
        feedbacks = await prisma.feedback.find_many(order=[{"contextoUsuario": "asc"}], take=100, skip=0, include={"knowledgeBase": True})
        return {"status": "success", "data": feedbacks}
    except Exception as e:
        logger.error(f"Erro ao buscar feedbacks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar feedbacks: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
