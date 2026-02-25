# Load Environment Variables
from dotenv import load_dotenv
import os
import tempfile
load_dotenv()  # Load .env variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Import Required Libraries for PDF Loading and Splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def ingest_uploaded_pdf(uploaded_file, store_path="vector_store/temp_index"):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Embed and index
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(store_path)

    # Clean up temporary file
    os.remove(temp_path)

    return store_path

# Load and Split the PDF
def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    splits = text_splitter.split_documents(docs)
    return splits

# Embed & Store in FAISS
def embed_and_store(chunks, store_path="vector_store/eso_faiss"):
    embeddings = OpenAIEmbeddings()
    
    faiss_store = FAISS.from_documents(chunks, embedding=embeddings)
    faiss_store.save_local(store_path)
    
    print(f"Saved FAISS index at: {store_path}")

# Script Entry Point
if __name__ == "__main__":
    pdf_file = "data/raw_pdfs/NESO_Annual_Report_2025.pdf"
    
    chunks = load_and_split(pdf_file)
    print(f"Total chunks created: {len(chunks)}")
    
    embed_and_store(chunks)
