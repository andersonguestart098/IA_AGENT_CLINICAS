import logging
import asyncio
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain_huggingface import HuggingFaceEmbeddings  # Nova implementação
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from prisma import Prisma

# Configurações iniciais
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vectorstore_path = Path("cemear_knowledge")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Lista de URLs para scraping
urls = [
    "https://www.tarkett.com.br/produtos/piso-vinilico",
    "https://www.eucafloor.com.br/linha/prime",
    "https://www.forbo.com/flooring/pt-br/",
    "https://www.cemear.com.br/",
    "https://www.cemear.com.br/divisorias/",
]

# Inicializa o Prisma
prisma = Prisma()

def scrape_website(url):
    try:
        logger.info(f"[+] Acessando: {url}")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.get(url)
        
        # Espera até que o conteúdo principal esteja carregado
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Tenta diferentes seletores para extrair texto
        texts = []
        for selector in ["p", "div", "span", "h1", "h2", "h3"]:  # Tenta diferentes tags
            elements = driver.find_elements(By.TAG_NAME, selector)  # Nova sintaxe
            for element in elements:
                text = element.text.strip()
                if text:
                    texts.append(text)
        
        driver.quit()
        return texts
    except Exception as e:
        logger.error(f"[x] Falha ao processar {url}: {str(e)}")
        return []

async def save_to_db_and_update_faiss(texts, url):
    try:
        # Divide os textos em blocos menores
        chunks = []
        for text in texts:
            chunks.extend(splitter.split_text(text))
        
        # Salva no banco de dados
        for chunk in chunks:
            await prisma.knowledgebase.create({
                "origem": url,
                "conteudo": chunk
            })
        
        # Carrega o FAISS existente ou cria um novo
        if vectorstore_path.exists():
            vectorstore = FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_texts(["Documento inicial vazio."], embeddings)
        
        # Adiciona os novos textos ao FAISS
        vectorstore.add_texts(chunks)
        vectorstore.save_local(str(vectorstore_path))
        
        logger.info(f"[✓] {len(chunks)} blocos adicionados da URL: {url}")
        return len(chunks)
    except Exception as e:
        logger.error(f"[x] Erro ao salvar no banco ou atualizar FAISS para {url}: {str(e)}")
        return 0

async def main():
    # Conecta ao Prisma
    await prisma.connect()
    logger.info("Conectado ao banco de dados Prisma para o scraper.")
    
    total_chunks = 0
    for url in urls:
        texts = scrape_website(url)
        if texts:
            chunks_added = await save_to_db_and_update_faiss(texts, url)
            total_chunks += chunks_added
    
    # Desconecta do Prisma
    await prisma.disconnect()
    logger.info(f"[✔] Scraper finalizado! Total de {total_chunks} blocos adicionados ao banco e FAISS.")

if __name__ == "__main__":
    asyncio.run(main())