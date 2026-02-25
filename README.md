# Domain-Adaptive Retrieval-Augmented Generation (RAG) System

An intent-aware Retrieval-Augmented Generation (RAG) system with live PDF upload and structured summarization, built using Streamlit and FAISS.

---

## 🚀 Project Overview

This application allows users to:

 - Upload a PDF document
 - Ask specific questions about its content
 - Generate structured executive summaries
 - Automatically detect user intent (QA vs Summary)
 - Retrieve relevant sections using semantic similarity
 - View grounded responses derived only from document context

Key capabilities:

🔎 Semantic Retrieval

- OpenAI embeddings
- FAISS vector index
- L2 distance similarity search
- Tuned threshold filtering
- Dynamic top-k adjustment

🧠 Intelligent Intent Routing

The system automatically detects:

 - Question-answering queries (e.g., “What risks were identified?”)
 - Summary requests (e.g., “Provide an overview”)

It dynamically switches retrieval and prompting strategy.

📊 Structured Executive Summary Mode

Generates summaries structured into:

 - Overall Purpose
 - Key Themes
 - Major Findings
 - Strategic Risks

💬 Chat-Based Interface

 - Scrollable conversation
 - Sidebar document upload
 - Session memory
 - Clean UI layout
 
---

## 🏗 Architecture

User Upload  
→ Temporary File Save
→ PDF Chunking
→ Embedding Generation
→ FAISS Index Creation
→ Query Embedding
→ Similarity Search
→ Intent Detection
→ Prompt Construction
→ LLM Response 

---
🛡 Guardrails

 - Refusal when no relevant chunks
 - Context-only answering
 - Threshold-based filtering
 - Page-level source traceability

---
🖥 Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Add your OpenAI key in a .env file:
```bash 
OPENAI_API_KEY=your_key_here
```
---
🌐 Deployment

Designed for deployment via:

 - Streamlit Community Cloud
