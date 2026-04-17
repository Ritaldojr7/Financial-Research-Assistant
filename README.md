# Financial Research Assistant

RAG-powered financial research assistant that lets you query annual reports and earnings transcripts like a financial analyst.

Built with **FastAPI**, **OpenAI**, **Pinecone**, **LangChain**, and **pdfplumber**.

---

## Project Structure

```
app/
  main.py              # FastAPI application factory
  routes/
    query.py           # POST /query endpoint
    ingest.py          # POST /ingest endpoint
  services/
    retriever.py       # Pinecone vector search + caching
    llm.py             # OpenAI chat completion + prompt engineering
    analyzer.py        # Query routing, comparison detection, evaluation
  ingestion/
    pdf_loader.py      # PDF text extraction via pdfplumber
    chunker.py         # Token-aware overlapping chunker
    embedder.py        # OpenAI embeddings
  utils/
    config.py          # Pydantic settings from environment
```

---

## Quick Start (Local)

### 1. Clone & install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY and PINECONE_API_KEY
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open **[http://localhost:8000/docs](http://localhost:8000/docs)** for the interactive Swagger UI.

---

## Quick Start (Docker)

```bash
docker build -t financial-research-assistant .

docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e PINECONE_API_KEY=... \
  -e API_KEY=your-secret-key \
  financial-research-assistant
```

---

## Deploy to Render

[Deploy to Render](https://render.com/deploy?repo=https://github.com/Ritaldojr7/Financial-Research-Assistant)

Or manually:

1. Go to [render.com/new](https://render.com/new) and select **Web Service**
2. Connect your GitHub repo `Ritaldojr7/Financial-Research-Assistant`
3. Render auto-detects the Dockerfile — keep defaults
4. Add environment variables: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `API_KEY`
5. Click **Deploy**

Your app will be live at `https://financial-research-assistant-xxxx.onrender.com/docs`

---

## API Reference

### `POST /ingest`

Upload a PDF financial document to extract, chunk, embed, and store.

**Form fields:**


| Field     | Type   | Required | Description               |
| --------- | ------ | -------- | ------------------------- |
| `file`    | File   | Yes      | PDF file                  |
| `company` | string | No       | Company name              |
| `year`    | string | No       | Fiscal year (e.g. "2024") |


**Response:**

```json
{
  "message": "Successfully ingested 'apple-10k-2024.pdf'",
  "chunks_stored": 42,
  "total_characters": 125000
}
```

### `POST /query`

Query ingested documents with natural language.

**Body:**

```json
{
  "query": "What was Apple's revenue growth in 2024?",
  "company": "Apple",
  "year": "2024"
}
```

**Response:**

```json
{
  "answer": "- Apple reported total net revenue of $383.3B in FY2024...",
  "sources": [
    {
      "text": "...",
      "score": 0.91,
      "company": "Apple",
      "year": "2024"
    }
  ],
  "confidence": "high"
}
```

### Comparison Queries

The system auto-detects comparison queries (e.g. "Compare Apple and Microsoft revenue"). It retrieves documents for both companies and generates a structured side-by-side analysis.

### `GET /health`

Returns `{"status": "ok", "version": "1.0.0"}`.

---

## Features

- **Intelligent Chunking** — token-aware chunking (500–800 tokens with 100-token overlap) preserves context across splits
- **Metadata Filtering** — filter retrieved documents by company, year, and detected report section
- **Comparison Detection** — automatically detects "compare X and Y" queries and retrieves context for both entities
- **Confidence Scoring** — response includes a confidence note based on retrieval similarity scores
- **In-Memory Caching** — LRU + TTL cache avoids redundant embedding/retrieval calls
- **Financial Analyst Prompt** — system prompt engineered for structured, numbers-heavy financial analysis
- **Basic Evaluation** — built-in keyword recall evaluator for retrieval quality smoke-tests

