# BCA-Product-RAG-Conversational-AI

## View — BCA Product Knowledge Assistant
An Indonesian-language RAG chatbot that answers questions about BCA banking 
products using semantic search over a structured vector knowledge base.

## Architecture

User Query
    ↓
LLM Intent Analyzer (Groq / Llama 3.1 8B)
    → Detects product name from 75+ products
    → Triggers clarification if query is ambiguous
    ↓
Semantic Retrieval (Pinecone + BAAI/bge-m3)
    → similarity_search(k=20) — no Pinecone-side filter
    → Post-filter by product_name using fuzzy normalization
    ↓
Answer Generation (Groq / Llama 3.1 8B)
    → Strictly grounded in retrieved context
    → Conversation memory: last 8 turns

## Tech Stack

- **Embedding Model**: BAAI/bge-m3 (dim=1024)
- **Vector Database**: Pinecone
- **LLM**: Llama 3.1 8B (Groq API)
- **UI Framework**: Gradio (HuggingFace Spaces)
- **Knowledge Base**: 1,052 chunks, 75+ BCA products


## Product Coverage

- **Simpanan Individu** — Tahapan BCA, Deposito Berjangka, e-Deposito, etc.
- **Pinjaman Individu** — KPR, KTA, KKB, Secured Personal Loan, etc.
- **Wealth Management** — 40+ insurance products, Reksadana, Obligasi, RDN
- **Kartu Kredit BCA** — 14 card variants
- **Uang Elektronik & Reward** — Flazz, Sakuku, Reward BCA

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PINECONE_API_KEY=your_key
export GROQ_API_KEY=your_key

# Run
python app.py
```

## Environment Variables

- **PINECONE_API_KEY**: Pinecone API key
- **GROQ_API_KEY**: Groq API key 

---
Developed by **Fati Buulolo**
