import os
import traceback
import logging
import sys
import io
from functools import lru_cache
from transformers import AutoTokenizer
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from prisma import Prisma
from pathlib import Path
import pytesseract
from PIL import Image
from io import BytesIO
import mimetypes
import pdfplumber
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import json
import faiss
from sklearn.preprocessing import normalize
from passlib.context import CryptContext
import secrets
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import datetime
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import asyncio
import time
from faster_whisper import WhisperModel
from tempfile import NamedTemporaryFile
from fastapi.staticfiles import StaticFiles
import secrets
import socketio  # noqa: F401  # usado indiretamente pelo nome



from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Carregar variáveis de ambiente
load_dotenv()

# Configurar o Prisma
prisma = Prisma(auto_register=True)

# Configurar o DATABASE_URL diretamente
os.environ["DATABASE_URL"] = "file:./prisma/dev.db"

# Criar diretórios
Path("uploads").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Configurar o Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

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

# Configurações da API Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

WPP_RAW_TOKEN = os.getenv("WPP_TOKEN")

NUMERO_GESTOR = os.getenv("WPP_GESTOR_VENDAS")

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Teste de saída
print(f"🔑 CHAVE MISTRAL: {MISTRAL_API_KEY[:8]}...")  # só os primeiros dígitos
if not MISTRAL_API_KEY:
    raise ValueError("❌ A variável MISTRAL_API_KEY não foi carregada.")

MISTRAL_MODEL = "mistral-large-latest"  # ou o modelo que você está usando

# Habilitar CORS para permitir acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fila_usuarios = defaultdict(asyncio.Queue)      # Fila de mensagens por telefone
locks_usuarios = defaultdict(asyncio.Lock)      # Lock por telefone (1 mensagem por vez)

# Modelo para embeddings
embedding_model = None


sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
# Envolver o FastAPI com o Socket.IO
socket_app = socketio.ASGIApp(sio, app)

# Evento para nova conexão
@sio.on("connect")
async def connect(sid, environ):
    print(f"🔌 Cliente conectado: {sid}")

# Evento para desconexão
@sio.on("disconnect")
async def disconnect(sid):
    print(f"❌ Cliente desconectado: {sid}")

# Evento customizado para enviar pedido em tempo real
@sio.on("atualizar_pedido")
async def atualizar_pedido(sid, data):
    session_id = data.get("sessionId")
    novo_item = data.get("item")
    if session_id and novo_item:
        # atualizar banco
        await adicionar_item_pedido(session_id, novo_item)
        print(f"📦 Pedido atualizado: {novo_item} (sessão: {session_id})")

        # emitir evento para todos os clientes conectados
        await sio.emit("pedido_atualizado", {
            "sessionId": session_id,
            "item": novo_item
        })

def load_embedding_model():
    """Carrega o modelo de embeddings."""
    global embedding_model
    try:
        if embedding_model is None:
            logger.info("🔄 Carregando modelo de embeddings...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Modelo de embeddings carregado com sucesso.")
        return embedding_model
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo de embeddings: {e}")
        raise

def gerar_embedding(texto):
    """
    Gera embedding para o texto usando SentenceTransformer.
    
    Args:
        texto: Texto para gerar embedding
        
    Returns:
        String JSON com o vetor de embedding
    """
    try:
        model = load_embedding_model()
        embedding = model.encode(texto)
        return json.dumps(embedding.tolist())
    except Exception as e:
        logger.error(f"❌ Erro ao gerar embedding: {e}")
        raise

def mensagem_inicial_simples(texto: str) -> bool:
    texto = texto.strip().lower()
    return texto in {"oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "tudo bem?", "e aí", "oie", "opa"}


def precisa_atendimento_humano(*args, **kwargs) -> bool:
    logger.info("🤖 Modo totem ativado - atendimento 100% automatizado.")
    return False


def chamar_mistral_api(prompt: str, temperature: float = 0.5, max_tokens: int = 300, system_override: str = None) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = system_override or (
        "Você é um atendente animado e simpático de um restaurante fast-food em um totem de autoatendimento por voz.\n"
        "Sua missão é atender o cliente com entusiasmo e cordialidade.\n"
        "Use frases curtas, com energia positiva e educação, como se fosse um atendente experiente que ama o que faz.\n"
        "Sempre confirme os pedidos, sugira bebidas ou sobremesas, e pergunte com empolgação se o cliente deseja mais alguma coisa."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    body = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if temperature > 0.0:
        body["top_p"] = 0.95

    for tentativa in range(5):
        try:
            logger.info(f"🔄 Tentativa {tentativa+1} | Requisição para Mistral (temp={temperature}, max_tokens={max_tokens})")

            res = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=30
            )

            if res.status_code == 429:
                logger.warning("⏳ Limite de requisições (429). Aguardando para retry...")
                time.sleep(2 + tentativa * 2)
                continue

            if res.status_code == 400:
                logger.error(f"❌ Erro 400 (Bad Request). Prompt:\n{prompt[:300]}...")
                return "OUTRO"

            res.raise_for_status()

            resposta_bruta = res.json()
            conteudo = resposta_bruta.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not conteudo:
                raise Exception("Resposta vazia.")

            return conteudo

        except Exception as e:
            logger.error(f"❌ Erro inesperado na chamada da Mistral API: {e}")
            time.sleep(2 + tentativa * 2)

    logger.error("❌ Falha após múltiplas tentativas com a Mistral API.")
    return "Estamos enfrentando instabilidade momentânea. Por favor, tente novamente em breve."

def classificar_intencao_mistral(pergunta: str) -> str:
    if not pergunta or not pergunta.strip():
        logger.warning("⚠️ Pergunta vazia na classificação de intenção.")
        return "OUTRO"

    pergunta = pergunta.strip()

    system_prompt = (
        "Você é um classificador de intenção para um sistema de pedidos de fast food por voz em um totem drive-thru.\n"
        "Classifique a intenção da frase do cliente usando UMA das seguintes categorias, SEM explicações:\n\n"
        "SAUDACAO - Ex: 'oi', 'boa tarde', 'alô'\n"
        "FAZER_PEDIDO - Quando o cliente pede comida ou bebida\n"
        "FINALIZAR_PEDIDO - Ex: 'só isso', 'pode fechar', 'quero pagar'\n"
        "CANCELAR - Quando o cliente desiste do pedido\n"
        "OUTRO - Qualquer outra coisa"
    )

    user_prompt = f"Frase do cliente: {pergunta}\nQual a intenção principal?"

    try:
        resposta = chamar_mistral_api(
            prompt=user_prompt,
            temperature=0.0,
            max_tokens=5,
            system_override=system_prompt
        )

        resposta = resposta.upper().strip()
        logger.info(f"🍔 Intenção classificada: '{resposta}'")

        categorias_validas = {"SAUDACAO", "FAZER_PEDIDO", "FINALIZAR_PEDIDO", "CANCELAR", "OUTRO"}
        return resposta if resposta in categorias_validas else "OUTRO"

    except Exception as e:
        logger.error(f"❌ Erro na classificação de intenção: {e}")
        return "OUTRO"


async def processar_mensagem_whatsapp(dados: dict):
    try:
        if not prisma.is_connected():
            await prisma.connect()

        telefone = dados.get("from")
        mensagem = dados.get("content") or dados.get("body") or dados.get("message", {}).get("text", "") or ""
        nome = dados.get("notifyName") or dados.get("contact", {}).get("name", "")
        session = dados.get("session") or os.getenv("WPP_SESSION") or "NERDWHATS_AMERICA"

        if telefone.endswith("@c.us"):
            telefone = telefone.replace("@c.us", "")

        if not mensagem.strip():
            return

        fluxo = await prisma.fluxoconversa.find_first(
            where={"telefone": telefone, "status": "em_andamento"}
        )

        if not fluxo:
            fluxo = await prisma.fluxoconversa.create(
                data={
                    "telefone": telefone,
                    "sessionId": session,
                    "userId": telefone,
                    "etapaAtual": "inicio",
                    "dadosParciais": "[]",
                    "tipoFluxo": "whatsapp",
                    "status": "em_andamento"
                }
            )

        try:
            historico = json.loads(fluxo.dadosParciais)
            if not isinstance(historico, list):
                historico = []
        except json.JSONDecodeError:
            historico = []

        mensagem_limpa = mensagem.strip().lower()

        def mensagem_inicial_simples(texto: str) -> bool:
            texto = texto.strip().lower()
            return texto in {"oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "tudo bem?", "e aí", "oie", "opa"}

        if mensagem_inicial_simples(mensagem):
            resposta = "Olá! Seja bem-vindo à Cemear. Meu nome é Cemy, como posso te ajudar?"
            
            historico.append({
                "tipo": "entrada",
                "conteudo": mensagem,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })

            historico.append({
                "tipo": "saida",
                "conteudo": resposta,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })

            await prisma.fluxoconversa.update(
                where={"id": fluxo.id},
                data={"dadosParciais": json.dumps(historico, ensure_ascii=False)}
            )

            token = get_token_dinamico()
            if not token:
                logger.error("❌ Token WPP inválido.")
                return

            headers = {
                "Authorization": f"Bearer {token.strip()}",
                "Content-Type": "application/json"
            }

            payload = {
                "phone": telefone,
                "message": resposta,
                "isGroup": False
            }

            envio = requests.post(
                f"http://wpp-server:21465/api/{session}/send-message",
                json=payload,
                headers=headers,
                timeout=10
            )

            logger.info(f"📤 Saudação simples respondida. Status envio: {envio.status_code}")
            return

        historico.append({
            "tipo": "entrada",
            "conteudo": mensagem,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        contexto_historico = ""
        if historico:
            recent_messages = historico[-5:]
            contexto_historico = "HISTÓRICO RECENTE:\n" + "\n".join([
                f"{'Usuário' if msg['tipo'] == 'entrada' else 'Cemear'}: {msg['conteudo']}"
                for msg in recent_messages
            ]) + "\n\n"

        resposta = ""
        if faiss_index.ntotal == 0 or not faiss_docs:
            resposta = "No momento, a base de conhecimento está indisponível. Por favor, tente novamente mais tarde."
        else:
            intencao = classificar_intencao_mistral(mensagem)
            logger.info(f"🧠 Intenção classificada: {intencao}")

            if intencao == "RECLAMACAO":
                resposta = "Lamento pelo ocorrido. Um de nossos atendentes entrará em contato em breve."
                await prisma.fluxoconversa.update(
                    where={"id": fluxo.id},
                    data={"etapaAtual": "aguardando_humano"}
                )

            elif intencao == "ORCAMENTO":
                resposta = "Certo! Encaminharei suas informações para um de nossos consultores. Aguarde o contato. 👍"

                logger.info(f"📨 Enviando lead para gestor via notificar_gestor_contato...")
                resultado_envio = await asyncio.to_thread(notificar_gestor_contato, telefone, mensagem, nome)
                logger.info(f"📨 Resultado envio ao gestor: {'✅ SUCESSO' if resultado_envio else '❌ FALHA'}")

                await prisma.fluxoconversa.update(
                    where={"id": fluxo.id},
                    data={"etapaAtual": "aguardando_vendedor"}
                )

            else:
                emb_pergunta = np.array(json.loads(gerar_embedding(mensagem)), dtype=np.float32).reshape(1, -1)
                emb_pergunta = normalize(emb_pergunta, axis=1)
                k = min(1, faiss_index.ntotal)
                D, I = faiss_index.search(emb_pergunta, k)
                resultados = []
                for i, score in zip(I[0], D[0]):
                    if i == -1 or i >= len(id_map): continue
                    doc_id = id_map[i]
                    doc = next((d for d in faiss_docs if d.id == doc_id), None)
                    if doc:
                        resultados.append((doc, float(score)))
                resultados_filtrados = [(doc, score) for doc, score in resultados if score > 0.3]

                if precisa_atendimento_humano(mensagem, resultados_filtrados, intencao):
                    resposta = "Sua pergunta requer análise especializada. Um de nossos técnicos entrará em contato."
                    await prisma.fluxoconversa.update(
                        where={"id": fluxo.id},
                        data={"etapaAtual": "aguardando_humano"}
                    )
                elif not resultados_filtrados:
                    resposta = chamar_mistral_api(
                        mensagem,
                        system_override=f"""
                        Você é um atendente técnico da Cemear (pisos vinílicos e laminados).

                        Regras:
                        - Use no máximo 3 frases curtas.
                        - Não adicione introduções ou encerramentos.
                        - Responda com base apenas na pergunta.
                        - Se não souber, diga: \"Preciso consultar um especialista sobre isso.\"

                        Mensagem do usuário:
                        {mensagem}
                        """.strip(),
                        temperature=0.3,
                        max_tokens=300
                    )
                else:
                    contexto = resultados_filtrados[0][0].conteudo
                    resposta = chamar_mistral_api(
                        mensagem,
                        system_override=f"""
                        Você é um atendente técnico da Cemear (pisos vinílicos e laminados).

                        Regras:
                        - Use no máximo 3 frases curtas.
                        - Responda APENAS com informações do documento.
                        - Não adicione introduções ou encerramentos.
                        - Linguagem direta e profissional.
                        - Se não souber, diga: \"Preciso consultar um especialista sobre isso.\"

                        {contexto_historico}

                        DOCUMENTO:
                        {contexto}

                        PERGUNTA:
                        {mensagem}
                        """.strip(),
                        temperature=0.3,
                        max_tokens=300
                    )

        resposta = resposta.strip() if resposta else "Não consegui processar sua pergunta. Pode reformular?"

        historico.append({
            "tipo": "saida",
            "conteudo": resposta,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        await prisma.fluxoconversa.update(
            where={"id": fluxo.id},
            data={"dadosParciais": json.dumps(historico, ensure_ascii=False)}
        )

        token = get_token_dinamico()
        if not token:
            logger.error("❌ Token WPP inválido.")
            return

        headers = {
            "Authorization": f"Bearer {token.strip()}",
            "Content-Type": "application/json"
        }

        payload = {
            "phone": telefone,
            "message": resposta,
            "isGroup": False
        }

        envio = requests.post(
            f"http://wpp-server:21465/api/{session}/send-message",
            json=payload,
            headers=headers,
            timeout=10
        )

        logger.info(f"📤 Envio WhatsApp status {envio.status_code}")

    except Exception as e:
        logger.error(f"❌ Erro no processamento WhatsApp:\n{traceback.format_exc()}")

async def processar_fila_usuario(telefone):
    if locks_usuarios[telefone].locked():
        return  # já está sendo processado

    async with locks_usuarios[telefone]:
        while not fila_usuarios[telefone].empty():
            dados = await fila_usuarios[telefone].get()
            try:
                logger.info(f"🚀 Processando mensagem de {telefone}")
                await processar_mensagem_whatsapp(dados)
            except Exception as e:
                logger.error(f"❌ Erro ao processar fila de {telefone}: {e}")


def notificar_gestor_contato(telefone_cliente: str, mensagem_cliente: str, nome_cliente: str = "") -> bool:
    """
    Envia notificação para o gestor de vendas com os dados do lead.
    """
    try:
        from datetime import datetime

        token = get_token_dinamico()
        if not token:
            logger.error("❌ Token WPP dinâmico não gerado.")
            return False

        WPP_SESSION = os.getenv("WPP_SESSION", "NERDWHATS_AMERICA")
        WPP_GESTOR = os.getenv("WPP_GESTOR_VENDAS")
        WPP_SERVER = os.getenv("WPP_SERVER_URL", "http://wpp-server:21465")  # default interno

        if not WPP_GESTOR:
            logger.error("❌ Número do gestor (WPP_GESTOR_VENDAS) não configurado no .env.")
            return False

        numero_cliente_formatado = telefone_cliente.replace("@c.us", "")
        nome_formatado = nome_cliente or "Nome não informado"
        hora = datetime.now().strftime("%d/%m/%Y %H:%M")

        mensagem = (
            f"📩 *NOVO LEAD DE ORÇAMENTO - IA CEMEAR*\n\n"
            f"👤 Nome: *{nome_formatado}*\n"
            f"📱 Telefone: *{numero_cliente_formatado}*\n"
            f"⏰ Data/Hora: {hora}\n\n"
            f"💬 *Mensagem:*\n_{mensagem_cliente.strip()[:300]}_\n\n"
            f"⚠️ Contato encaminhado pela IA. Por favor, responder o quanto antes."
        )

        headers = {
            "Authorization": f"Bearer {token.strip()}",
            "Content-Type": "application/json"
        }

        payload = {
            "phone": WPP_GESTOR,
            "message": mensagem,
            "isGroup": False
        }

        url = f"{WPP_SERVER}/api/{WPP_SESSION}/send-message"
        logger.info(f"📨 Enviando lead para gestor {WPP_GESTOR} via {url}")

        res = requests.post(url, json=payload, headers=headers, timeout=15)

        if res.status_code in [200, 201]:
            logger.info("✅ Lead enviado com sucesso para o gestor de vendas.")
            return True
        else:
            logger.error(f"❌ Erro HTTP {res.status_code} ao enviar lead: {res.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Erro ao enviar notificação de lead: {e}")
        return False


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Função para extrair texto de arquivos
async def extrair_texto(content: bytes, file: UploadFile) -> str:
    """Extrai texto de diferentes tipos de arquivo."""
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

def get_token_dinamico():
    session = os.getenv("WPP_SESSION")
    secret = os.getenv("WPP_SECRET_KEY")  # você pode adicionar isso ao .env

    url = f"http://wpp-server:21465/api/{session}/{secret}/generate-token"
    try:
        res = requests.post(url)
        res.raise_for_status()
        data = res.json()
        return data.get("token")
    except Exception as e:
        logger.error(f"❌ Erro ao buscar token dinâmico: {e}")
        return None

async def adicionar_item_pedido(session_id: str, novo_item: str):
    fluxo = await prisma.fluxoconversa.find_first(where={"sessionId": session_id})
    if not fluxo:
        return

    try:
        pedido = json.loads(fluxo.pedido or "[]")
    except:
        pedido = []

    pedido.append(novo_item)

    await prisma.fluxoconversa.update(
        where={"id": fluxo.id},
        data={"pedido": json.dumps(pedido, ensure_ascii=False)}
    )


# FAISS com produto interno (ideal para vetores normalizados)
faiss_index = faiss.IndexFlatIP(384)  # 384 = dimensão do embedding
id_map = {}
faiss_docs = []

@app.on_event("startup")
async def startup():
    """Inicialização da aplicação."""
    await prisma.connect()
    logger.info("✅ Prisma conectado com sucesso.")
    try:
        logger.info(f"📂 Banco em uso: {os.environ['DATABASE_URL']}")
    except Exception:
        logger.warning("⚠️ Não foi possível identificar o caminho do banco.")

    # Carregar modelo de embeddings
    load_embedding_model()

    # Limpar índice FAISS existente
    global faiss_index, id_map, faiss_docs
    faiss_index = faiss.IndexFlatIP(384)  # Reinicializar o índice
    id_map = {}
    faiss_docs = []

    # Construir índice FAISS com documento consolidado
    try:
        # Buscar o documento consolidado pelo nome do arquivo
        documento = await prisma.knowledgebase.find_first(
            where={"origem": "cemear_base_conhecimento_consolidada.txt"}
        )
        
        if not documento:
            logger.warning("⚠️ Documento consolidado não encontrado, usando todos os documentos.")
            documentos = await prisma.knowledgebase.find_many(where={"embedding": {"not": None}})
        else:
            documentos = [documento]
            logger.info(f"✅ Usando documento consolidado ID {documento.id}")
        
        for idx, doc in enumerate(documentos):
            try:
                vetor = np.array(json.loads(doc.embedding), dtype=np.float32).reshape(1, -1)
                vetor_normalizado = normalize(vetor, axis=1)
                faiss_index.add(vetor_normalizado)
                id_map[idx] = doc.id
                faiss_docs.append(doc)
            except Exception as e:
                logger.warning(f"⚠️ Erro ao adicionar embedding ID {doc.id} ao FAISS: {e}")
        logger.info(f"📌 FAISS carregado com {len(faiss_docs)} documentos normalizados.")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar FAISS no startup: {e}")

@app.on_event("shutdown")
async def shutdown():
    """Desconexão ao desligar a aplicação."""
    await prisma.disconnect()
    logger.info("🔕 Prisma desconectado.")

# Modelo para requisições de chat
class PromptRequest(BaseModel):
    question: str
    session_id: str
    user_id: str


@app.post("/chat-atendente")
async def chat_atendente(data: PromptRequest):
    """Endpoint principal para chat com o atendente virtual."""
    try:
        if not prisma.is_connected():
            await prisma.connect()

        pergunta = data.question.strip()
        if not pergunta:
            raise HTTPException(400, "Mensagem vazia.")

        logger.info(f"📥 Pergunta recebida: {pergunta}")

        # 👋 Saudações simples: responde direto sem IA
        if mensagem_inicial_simples(pergunta):
            resposta = "Olá! Tudo bem? Como posso te ajudar?"
            logger.info("🤖 Resposta rápida para saudação simples.")
            return {"answer": resposta}

        # 🔎 Etapa 1: Classificação de intenção
        intencao = classificar_intencao_mistral(pergunta)
        logger.info(f"🧠 Intenção classificada: {intencao}")

        if intencao == "SAUDACAO":
            return {"answer": "Olá! Como posso ajudar?"}
        elif intencao == "ELOGIO":
            return {"answer": "Obrigado pelo feedback! Estamos à disposição."}
        elif intencao == "RECLAMACAO":
            return {
                "answer": "Lamento pelo ocorrido. Vou transferir para um atendente especializado resolver sua questão.",
                "encaminhar_humano": True
            }
        elif intencao == "ORCAMENTO":
            return {
                "answer": "Certo! Encaminharei suas informações para um de nossos consultores. Aguarde o contato. 👍",
                "encaminhar_vendedor": True
            }

        # 🔍 Etapa 2: Verificação da base de conhecimento
        if faiss_index.ntotal == 0 or not faiss_docs:
            return {"answer": "Base de conhecimento indisponível no momento."}

        emb_pergunta = np.array(json.loads(gerar_embedding(pergunta)), dtype=np.float32).reshape(1, -1)
        emb_pergunta = normalize(emb_pergunta, axis=1)

        k = min(1, faiss_index.ntotal)
        D, I = faiss_index.search(emb_pergunta, k)

        resultados = []
        for i, score in zip(I[0], D[0]):
            if i == -1 or i >= len(id_map):
                continue
            doc_id = id_map[i]
            doc = next((d for d in faiss_docs if d.id == doc_id), None)
            if doc:
                resultados.append((doc, float(score)))

        resultados_filtrados = [(doc, score) for doc, score in resultados if score > 0.3]

        if precisa_atendimento_humano(pergunta, resultados_filtrados, intencao):
            return {
                "answer": "Sua pergunta requer análise especializada. Vou transferir para um de nossos técnicos.",
                "encaminhar_humano": True
            }

        if not resultados_filtrados:
            return {"answer": "Não encontrei essa informação nos documentos técnicos."}

        contexto = resultados_filtrados[0][0].conteudo

        prompt = f"""
Responda com base apenas no documento abaixo.
Use no máximo 3 frases curtas e diretas.
Se não encontrar a resposta, diga apenas: "Preciso consultar um especialista sobre isso."

DOCUMENTO:
{contexto}

PERGUNTA:
{pergunta}
"""
        resposta = chamar_mistral_api(prompt, temperature=0.3, max_tokens=300)
        return {"answer": resposta}

    except Exception:
        logger.error("❌ Erro no /chat-atendente:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno ao processar atendimento.")

@app.post("/chat-com-voz")
async def chat_com_voz(data: PromptRequest):
    try:
        texto = data.question.strip()
        session_id = data.session_id.strip()
        user_id = data.user_id.strip()

        if not texto or not session_id or not user_id:
            raise HTTPException(status_code=400, detail="Campos obrigatórios ausentes.")

        fluxo = await prisma.fluxoconversa.find_first(where={"sessionId": session_id})

        if not fluxo:
            fluxo = await prisma.fluxoconversa.create(
                data={
                    "telefone": user_id,
                    "sessionId": session_id,
                    "userId": user_id,
                    "etapaAtual": "inicio",
                    "dadosParciais": "{}",
                    "tipoFluxo": "voz_drive_thru"
                }
            )

        etapa = fluxo.etapaAtual
        dados_parciais = json.loads(fluxo.dadosParciais)

        if "pedido" not in dados_parciais:
            dados_parciais["pedido"] = []

        documentos = await prisma.knowledgebase.find_many()
        doc = next((d for d in documentos if "drive" in d.origem.lower()), None)
        contexto = doc.conteudo.strip() if doc else ""

        if etapa == "inicio":
            prompt = f"""
Você é um atendente de drive-thru por voz do restaurante Suny Burger.

Dê apenas uma saudação inicial amigável e clara, como:
"Olá, meu nome é Anderson. Seja bem-vindo ao Suny Burger! O que você gostaria de pedir hoje?"

Após a saudação, escute e registre com precisão o primeiro item que o cliente mencionar. Não se apresente novamente nas próximas interações.

Base oficial de atendimento:
{contexto}
"""
            proxima_etapa = "meio"

        elif etapa == "meio":
            prompt = f"""
Você é um atendente de drive-thru por voz do restaurante Suny Burger. O cliente está montando seu pedido por etapas.

Base oficial de atendimento:
{contexto}

Histórico do pedido até agora:
{json.dumps(dados_parciais["pedido"], ensure_ascii=False)}

Fala do cliente:
{texto}

Atualize o pedido com base na fala, acrescentando, corrigindo ou removendo itens se necessário.
Se ainda não houver batata, bebida ou sobremesa, sugira gentilmente.
Não se apresente novamente. Seja claro, simpático e direto.
"""
            proxima_etapa = "meio"
            dados_parciais["pedido"].append(texto)

        elif etapa == "fim":
            prompt = f"""
Você é um atendente do Suny Burger. O cliente está finalizando o pedido.

Pedido registrado:
{json.dumps(dados_parciais["pedido"], ensure_ascii=False)}

Fala do cliente:
{texto}

Confirme item por item e pergunte se está tudo certo ou se deseja alterar algo.
Finalize o atendimento com simpatia, sem se apresentar novamente.
"""
            proxima_etapa = "concluido"

        else:
            prompt = f"""
Fala final do cliente:
{texto}

Finalize o atendimento confirmando o pedido e agradecendo pela preferência.
Diga apenas: "Pedido finalizado. Obrigado e bom apetite!".
"""
            proxima_etapa = "concluido"

        resposta = chamar_mistral_api(
            prompt,
            temperature=0.5,
            max_tokens=300,
            system_override=(
                "Você é o atendente virtual por voz do Suny Burger. "
                "Nunca se apresente novamente após a primeira fala. "
                "Mantenha o pedido sempre atualizado com base nas falas do cliente. "
                "Seja simpático, confirme os itens, sugira batata, bebida ou sobremesa se ainda não foram pedidos, "
                "e nunca invente itens. Finalize o atendimento somente quando o cliente indicar."
            )
        )

        if not resposta or not resposta.strip():
            raise Exception("Resposta da IA está vazia.")

        logger.info(f"🧠 Resposta da IA: {resposta}")

        await prisma.fluxoconversa.update(
            where={"id": fluxo.id},
            data={
                "etapaAtual": proxima_etapa,
                "dadosParciais": json.dumps(dados_parciais, ensure_ascii=False),
                "pedido": json.dumps(dados_parciais["pedido"], ensure_ascii=False)
            }
        )

        api_key = os.getenv("ELEVEN_API_KEY")
        voice_id = os.getenv("ELEVEN_VOICE_ID", "TxGEqnHWrfWFTf9aZ8sM")
        audio_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        body = {
            "text": resposta,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.9
            }
        }

        res = requests.post(audio_url, headers=headers, json=body)

        if res.status_code != 200:
            logger.error(f"❌ Erro ElevenLabs: {res.status_code} - {res.text}")
            return {
                "resposta_texto": resposta,
                "resposta_audio": None
            }

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"resposta_{int(time.time())}.mp3"

        with open(output_path, "wb") as f:
            f.write(res.content)

        return {
            "resposta_texto": resposta,
            "resposta_audio": f"/outputs/{output_path.name}"
        }

    except Exception as e:
        logger.error(f"❌ Erro no /chat-com-voz: {e}")
        raise HTTPException(status_code=500, detail="Erro no chat com voz.")




@app.post("/upload")
async def upload_conhecimento(file: UploadFile = File(...)):
    """
    Endpoint para upload de documentos para a base de conhecimento.
    """
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

        logger.info(f"✅ Documento '{file.filename}' adicionado com ID {kb.id}")
        return {"status": "sucesso", "resumo": texto[:200], "id": kb.id}
    except Exception:
        logger.error("❌ Erro ao salvar conhecimento: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno ao processar o upload.")

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    try:
        dados = await request.json()
        telefone = dados.get("from")

        if not telefone or not dados.get("content"):
            return {"status": "ignorado"}

        # Remover sufixo "@c.us" se existir
        if telefone.endswith("@c.us"):
            telefone = telefone.replace("@c.us", "")

        # Adicionar a mensagem à fila
        await fila_usuarios[telefone].put(dados)
        logger.info(f"📥 Mensagem de {telefone} enfileirada.")

        # Iniciar processamento da fila se não estiver em execução
        asyncio.create_task(processar_fila_usuario(telefone))

        return {"status": "em_fila"}

    except Exception as e:
        logger.error(f"❌ Erro no webhook WhatsApp:\n{traceback.format_exc()}")
        return {"status": "erro", "mensagem": str(e)}
    
@app.post("/escutar")
async def escutar_audio(file: UploadFile = File(...)):
    """Transcreve áudio enviado (voz → texto) usando faster-whisper."""
    try:
        # Salvar áudio temporariamente
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(await file.read())
            caminho = temp_audio.name

        model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu")
        segments, _ = model.transcribe(caminho)

        transcricao = " ".join(segment.text for segment in segments)
        return {"transcricao": transcricao.strip()}

    except Exception as e:
        logger.error(f"❌ Erro ao transcrever áudio: {e}")
        raise HTTPException(status_code=500, detail="Erro ao transcrever áudio.")


@app.post("/falar")
async def falar_com_ia(prompt: PromptRequest):
    """Converte texto em voz realista (texto → fala) com ElevenLabs."""
    try:
        texto = prompt.question.strip()
        if not texto:
            raise HTTPException(status_code=400, detail="Texto vazio.")

        api_key = os.getenv("ELEVEN_API_KEY")
        voice_id = os.getenv("ELEVEN_VOICE_ID", "TxGEqnHWrfWFTf9aZ8sM")  # padrão

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        body = {
            "text": texto,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.9
            }
        }

        resposta = requests.post(url, headers=headers, json=body)
        if resposta.status_code != 200:
            raise Exception(f"Erro ElevenLabs: {resposta.status_code} - {resposta.text}")

        output_path = f"outputs/resposta_{int(time.time())}.mp3"
        Path("outputs").mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resposta.content)

        return {
            "audio_path": output_path,
            "mensagem_original": texto
        }

    except Exception as e:
        logger.error(f"❌ Erro no TTS: {e}")
        raise HTTPException(status_code=500, detail="Erro ao gerar áudio.")
    

@app.post("/registro")
async def registrar_usuario(nome: str = Form(...), email: str = Form(...), senha: str = Form(...)):
    try:
        usuario_existente = await prisma.usuario.find_unique(where={"email": email})
        if usuario_existente:
            raise HTTPException(status_code=400, detail="E-mail já cadastrado.")

        senha_hash = pwd_context.hash(senha)

        novo_usuario = await prisma.usuario.create(data={
            "nome": nome,
            "email": email,
            "senhaHash": senha_hash
        })

        return {
            "message": "✅ Registro realizado com sucesso.",
            "userId": novo_usuario.id
        }

    except Exception as e:
        logger.error(f"❌ Erro no registro: {e}")
        raise HTTPException(status_code=500, detail="Erro ao registrar.")

@app.post("/login")
async def login_usuario(email: str = Form(...), senha: str = Form(...)):
    try:
        usuario = await prisma.usuario.find_unique(where={"email": email})
        if not usuario or not pwd_context.verify(senha, usuario.senhaHash):
            raise HTTPException(status_code=401, detail="Credenciais inválidas.")

        token = secrets.token_hex(16)

        nova_sessao = await prisma.sessao.create(data={
            "token": token,
            "usuarioId": usuario.id
        })

        # Iniciar novo fluxo de conversa
        await prisma.fluxoconversa.create(data={
            "telefone": usuario.email,
            "sessionId": token,
            "userId": usuario.id,
            "etapaAtual": "inicio",
            "dadosParciais": "{}",
            "tipoFluxo": "voz_drive_thru"
        })

        return {
            "message": "✅ Login realizado com sucesso.",
            "userId": usuario.id,
            "sessionId": token
        }

    except Exception as e:
        logger.error(f"❌ Erro no login: {e}")
        raise HTTPException(status_code=500, detail="Erro ao realizar login.")
    
@app.get("/pedido")
async def obter_pedido(session_id: str):
    fluxo = await prisma.fluxoconversa.find_first(where={"sessionId": session_id})
    if not fluxo:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    
    try:
        pedido = json.loads(fluxo.pedido or "[]")
    except:
        pedido = []

    return {"pedido": pedido}

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
