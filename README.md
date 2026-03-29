# Document Q&A System (Free Stack)

A retrieval-augmented generation (RAG) system for answering questions over a single long document (100+ pages).

| Component | Service | Model | Free tier |
|---|---|---|---|
| Embeddings | Jina AI | `jina-embeddings-v2-base-en` | 1M tokens/month |
| Re-ranking | Jina AI | `jina-reranker-v2-base-multilingual` | Free tier |
| Generation | Groq | `llama-3.1-8b-instant` | 30 req/min, 6k tokens/min |

---

## Problem Framing

I interpreted this task as designing a pipeline that can take in a long and dense document that is mostly text with some other elements such as tables and charts and answer questions regarding the contents. This will save the user the effort of having to read through pages of dry and likely useless information to find what they want. 

Although LLMs today are largely capable of such a task, the difficulty lies in the fact that very long documents may cause LLMs to experience many issues such as:
1. **"Lost in the middle" phenomenon**: Where the LLM prioritizes information at the start and end of a document while losing information from the middle.
2. **Context fragmentation**: Where the "attention" mechanism becomes spread thin and focuses too much on irrelevant words.
3. **Document faithfulness**: Where long documents increase the load on the model, causing it to fall back on prior training knowledge instead of the provided text.
4. **Token limitations**: Most models cannot digest extremely long documents in a single input.

This pipeline is designed to extract a select few chunks with the highest probability of answering a query, then using a standard LLM approach to generate a cited answer.

---

## Setup & Dockerization

Using Docker ensures that system-level dependencies for PDF parsing (like `poppler`) and vector processing are consistent and easy to set up.

### 1. Get your free API keys
* **Jina AI**: Sign up at [jina.ai](https://jina.ai) to get your API key.
* **Groq**: Sign up at [console.groq.com](https://console.groq.com) and create an API key.

### 2. Configuration
Create a `.env` file in the root directory(should be there already):
```text
JINA_API_KEY=jina_...
GROQ_API_KEY=gsk_...

```

### 3. Build the Container

```bash
docker-compose build

```

---

## Usage (Containerized)

### Ingest a document (run once)

Place your PDF in the `./data` folder on your host machine. For testing purposes, I uploaded it to src directly since I only dockerized it afterwards.

```bash
docker-compose run --rm rag-system python ingest.py Sapiens.pdf

```

This parses the PDF, builds a local FAISS index + BM25 index, and persists them to the `./index/` folder via Docker volumes.

### Ask questions

**Interactive mode:**

```bash
docker-compose run --rm rag-system python query.py

```

**Single question:**

```bash
docker-compose run --rm rag-system python query.py "What is the ultimate goal of humanity?"

```

### Development & Debugging

To enter the container shell once it is running to run manual tests:

```bash
docker exec -it document_qa /bin/bash

```

---

## Architecture

```
PDF
 └─ parser.py      Extract text + tables, build overlapping chunks
     └─ ingest.py  Embed via Jina AI → FAISS index + BM25 index (run once)
         └─ retrieval.py   BM25 + FAISS → RRF fusion → Jina reranker
             └─ generation.py   Prompt + chunks → Groq (Llama 3) → cited answer

```
## Execution Flow

### Phase 1 — Ingestion (run once per document)
```
1. parse_document()       [parser.py]
   └── pdfplumber extracts text and tables page by page
   └── text split into overlapping chunks (~512 tokens, 50 token overlap)
   └── paragraph boundaries respected, section headings detected
   └── each chunk stores: text, chunk_id, page_start, page_end, heading

2. _embed_texts()         [ingest.py]
   └── each chunk's text sent to Jina AI Embeddings API
   └── returns a 768-dimensional vector per chunk
   └── similar chunks end up geometrically close in vector space

3. _build_faiss_index()   [ingest.py]
   └── vectors L2-normalised (so inner product == cosine similarity)
   └── stored in a FAISS FlatIP index for exact nearest-neighbour search

4. _build_bm25_index()    [ingest.py]
   └── chunks tokenised and stored in a BM25Okapi index
   └── enables fast keyword frequency scoring at query time

5. save to ./index/       [ingest.py]
   └── faiss.index  — all chunk vectors
   └── bm25.pkl     — BM25 index
   └── chunks.json  — full text and metadata for every chunk
```

---

### Phase 2 — Querying (run per question)
```
1. Hybrid retrieval       [retrieval.py]
   ├── Semantic search
   │   └── query embedded via Jina AI (same model as ingestion)
   │   └── FAISS finds top 20 nearest chunk vectors by cosine similarity
   │   └── catches conceptual matches even when wording differs
   │
   └── BM25 keyword search
       └── query tokenised and scored against all chunks
       └── catches exact matches for specific terms, numbers, proper nouns

2. RRF fusion             [retrieval.py]
   └── two ranked lists combined via Reciprocal Rank Fusion
   └── score = sum of 1/(60 + rank) across both lists
   └── chunks ranking highly in both lists get boosted
   └── raw scores discarded — only rank position matters
   └── produces unified top 20 candidates

3. Jina reranker          [retrieval.py]
   └── all 20 (query, chunk) pairs sent to Jina Reranker API
   └── query and chunk processed jointly through a cross-encoder transformer
   └── every query token attends to every chunk token via self-attention
   └── produces a calibrated relevance score (0-1) per pair
   └── top 5 chunks returned ordered by relevance score

4. Abstention check       [generation.py]
   └── if best rerank score < 0.1, system abstains without calling LLM
   └── prevents hallucination when no relevant content was retrieved

5. Prompt construction    [generation.py]
   └── top 5 chunks formatted as numbered excerpts [1] through [5]
   └── each excerpt labelled with page number and section heading
   └── system prompt instructs LLM to cite sources and abstain if unsure

6. LLM generation         [generation.py]
   └── prompt sent to Groq API (Llama 3.1-8b-instant)
   └── answer generated with inline citations e.g. [1], [2][3]
   └── answer grounded in provided excerpts only
```

### Hybrid Retrieval Motivation

Dense embeddings (Jina) capture semantic similarity but miss exact terms—critical for financial documents where specific figures or "Note 14" need exact matching. BM25 handles these cases. Combining both with **Reciprocal Rank Fusion (RRF)** consistently outperforms either alone.

### Reranker (Cross-encoding) Motivation

The first-stage retrieval scores each chunk independently. The Jina reranker processes (query, chunk) jointly—equivalent to a cross-encoder—giving much more precise relevance scores. Since this is computationally expensive, it only re-scores the top 20 candidates.

---

## Test Results

Testing was conducted manually with a 300-page novel and a 250-page FinanceBench report.

* **Findings**: The pipeline excels at direct requests but has trouble with abstract inferences (e.g., realizing CAPEX reflects capital intensity).
* **Observation**: Performance was significantly better with general prose (the "Sapiens" novel) than with niche finance documents, likely due to the general-purpose nature of the 768-dimension embeddings.

---

## Assumptions and Limitations

* **English language**: BM25 tokenizer is whitespace-based and optimized for English.
* **Text-based PDFs**: Documents with heavy visual elements (images/charts) will not perform well.
* **Rate limits**: Free tiers are limited to 30 requests/minute and fixed credit quotas.
* **Context**: To answer questions that require information scattered throughout the text, this model will perform poorly because it is designed to extract information from specific chunks

## Future Improvements

* Custom-trained model for finance-specific document processing.
* Image/diagram processing step for visual charts.

### Disclaimer

AI was used to assist with code-formatting, syntax-correction, and debugging. All testing was done manually.
