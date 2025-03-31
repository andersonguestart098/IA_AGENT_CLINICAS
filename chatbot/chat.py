from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA

import os
import fitz  # PyMuPDF para leitura de PDFs
import tempfile
from transformers import pipeline

# Inicializando a API
app = FastAPI(title="Cemear AI Chatbot", description="API para chatbot inteligente da Cemear", version="1.0")

# Configuração do modelo Llama 3 (Hugging Face)
llm_pipeline = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B", max_length=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Carregar base de conhecimento (usando FAISS como banco vetorial)
vectorstore = FAISS.load_local("cemear_knowledge", HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Modelo de entrada do usuário
class UserQuery(BaseModel):
    question: str

# Rota para interação com o chatbot
@app.post("/chat")
def chat(query: UserQuery):
    try:
        response = qa_chain.run(query.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rota para upload de arquivos (PDFs, TXTs, etc.)
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        # Salvar arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_filepath = temp_file.name
        
        # Extrair texto do arquivo
        extracted_text = ""
        if file.filename.endswith(".pdf"):
            doc = fitz.open(temp_filepath)
            extracted_text = "\n".join([page.get_text("text") for page in doc])
        elif file.filename.endswith(".txt"):
            with open(temp_filepath, "r", encoding="utf-8") as txt_file:
                extracted_text = txt_file.read()
        else:
            return {"message": "Formato de arquivo não suportado"}
        
        # Criar embeddings e armazenar no FAISS
        texts = extracted_text.split(". ")  # Separação por frases para melhor indexação
        vectorstore.add_texts(texts)
        vectorstore.save_local("cemear_knowledge")
        
        return {"message": "Arquivo processado e adicionado à base de conhecimento."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rota de teste
@app.get("/")
def read_root():
    return {"message": "API do Chatbot da Cemear rodando!"}

# Executar a API com Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)