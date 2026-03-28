# NASA Systems Engineering Handbook — QA System

A locally-running RAG (Retrieval-Augmented Generation) system that answers
questions about the NASA Systems Engineering Handbook with page citations.

**100% free. No API keys. No internet needed after setup.**

---

## Tech Stack

| Component      | Tool                    | Why                              |
|----------------|-------------------------|----------------------------------|
| PDF parsing    | PyMuPDF (fitz)          | Fast, reliable text extraction   |
| Embeddings     | sentence-transformers   | Free, local, 80MB model          |
| Vector DB      | ChromaDB                | Local, no server needed          |
| LLM            | Ollama (llama3.2)       | Free, local, no API key          |
| UI             | Streamlit               | Python-only web UI               |

---

## Setup (one-time)

### 1. Install Ollama
Download from https://ollama.com and install it like a normal app.
Then pull the model:
```bash
ollama pull llama3.2
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Python packages
```bash
pip install -r requirements.txt
```

### 4. Download the PDF
Download the NASA Systems Engineering Handbook:
https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf

Save it as: `data/nasa_handbook.pdf`

### 5. Run ingestion (one-time)
```bash
python ingest.py
```
This takes 1-2 minutes. You only need to do this once.

### 6. Start the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## Architecture

```
User Question
     |
     v
query_processor.py   ← cleans text, expands acronyms (TRL → Technology Readiness Level)
     |
     v
retriever.py         ← embeds question, searches ChromaDB for top-5 similar chunks
     |          \
     |           chromadb (vector store)
     v
answer_generator.py  ← builds prompt with context + question, calls Ollama
     |
     v
app.py               ← displays answer + page citations in Streamlit UI
```

**Ingestion pipeline (run once):**
```
nasa_handbook.pdf
     |
     v
pdf_loader.py    ← extracts text page by page (keeps page numbers)
     |
     v
chunker.py       ← splits into 500-char chunks with 100-char overlap
     |
     v
embedder.py      ← converts each chunk to a 384-dim vector
     |
     v
vector_store.py  ← stores vectors + text + page numbers in ChromaDB
```

---

## Key Design Decisions

**Why chunk with overlap?**
Sentences that fall at a chunk boundary would be cut in half without overlap.
100-char overlap ensures every sentence appears complete in at least one chunk.

**Why all-MiniLM-L6-v2 for embeddings?**
Small (80MB), fast on CPU, and accurate enough for retrieval.
Larger models (e.g. text-embedding-ada-002) are more accurate but cost money.

**Why temperature=0.1 for the LLM?**
Technical document QA needs factual, consistent answers.
Low temperature = model sticks to the retrieved context instead of hallucinating.

**Why separate ingest.py from app.py?**
Ingestion takes 1-2 minutes. If it ran every time the app started,
the demo would be unusable. Separating it means the app starts instantly.

---

## Known Limitations

- Scanned pages (images) produce no text — only text-based PDFs work
- Diagrams and figures cannot be understood — text extraction only
- Very long tables that span multiple pages may be split incorrectly
- Answers are limited to what was retrieved — if the wrong chunks are
  retrieved, the answer will be wrong or incomplete
- Ollama's first response takes 10-20 seconds while the model loads into RAM

---

## Sample Questions to Demo

- "What is the Vee Model?"
- "What are the entry criteria for PDR?"
- "What does TRL mean?"
- "How does risk management work in systems engineering?"
- "What is a Work Breakdown Structure?"
- "What happens during a Critical Design Review?"
