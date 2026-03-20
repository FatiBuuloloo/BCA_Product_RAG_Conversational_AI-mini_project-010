# BCA-Product-RAG-Conversational-AI

## View — BCA Product Knowledge Assistant
An Indonesian-language RAG chatbot that answers questions about BCA banking 
products using semantic search over a structured vector knowledge base.

## Architecture

```mermaid
graph TD
    A[User Query] --> B[LLM Intent Analyzer <br/> Groq / Llama 3.1 8B]
    
    B --> B1{Product Detected?}
    B1 -- No/Ambiguous --> C[Trigger Clarification Context]
    B1 -- Yes --> D[Semantic Retrieval <br/> Pinecone + BAAI/bge-m3]
    
    C --> A
    
    D --> D1[Similarity Search k=20]
    D1 --> D2[Post-filter by product_name <br/> using Fuzzy Normalization]
    
    D2 --> E[Answer Generation <br/> Groq / Llama 3.1 8B]
    
    E --> E1[Strictly Grounded in Context]
    E --> E2[Conversation Memory: last 8 turns]
    
    E1 & E2 --> F[Final Response to User]
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bfb,stroke:#333,stroke-width:2px

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
