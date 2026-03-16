# Document Q&A System (Free Stack)

A retrieval-augmented generation (RAG) system for answering questions over a single long document (100+ pages).

| Component | Service | Model | Free tier |
|---|---|---|---|
| Embeddings | Jina AI | `jina-embeddings-v2-base-en` | 1M tokens |
| Re-ranking | Jina AI | `jina-reranker-v2-base-multilingual` | Free tier |
| Generation | Groq | `llama-3.1-8b-instant` | 30 req/min, 6k tokens/min |

---
## Problem Framing

I interpreted this task as designing a pipeline that can take in a long and dense document that is mostly text with some 
other elements such as tables and charts and answer questions regarding the contents. This will save the user the effort 
of having to read through pages of dry and likely useless information to find what they want. Although LLMs today are largely
capable of such a task, the difficult lies in the fact that very long documents may cause LLMs to experience many issues such
as
1. "Lost in the middle" phenomenon, where the LLM prioritizes information at the start and end of a document while losing information from the middle
2. Context fragmentation, where the "attention" mechanism (the math that determines which words are relevant to others) becomes spread thin and so the model focuses too much on irrelevant words
3. Document faithfulness versus prior knowledge, where long documents tend to increase the load on the model so it will fall back on the information it learned during training instead of using the provided document
4. Token limitations, most models simply cannot digest extremely long documents as an input at one go

Thus this pipeline is designed to extract out a select few chunks from the document that that have the highest probability
answering a given query, then using a standard LLM approach to feed those chunks into a model and use generate an answer
## Setup

### 1. Get your free API keys

**Jina AI** (embeddings + reranking):
1. Go to https://jina.ai
2. Click "Get API Key"
3. Sign up with your email — no credit card required
4. Copy your API key from the dashboard

**Groq** (LLM generation):
1. Go to https://console.groq.com
2. Click "Sign Up"
3. Sign up with Google or email — no credit card required
4. Go to "API Keys" → "Create API Key"
5. Copy your API key

### 2. Install dependencies

```bash
git clone <repo>
cd rag_system
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export JINA_API_KEY=jina_...
export GROQ_API_KEY=gsk_...
```

Or create a `.env` file:
```
JINA_API_KEY=jina_...
GROQ_API_KEY=gsk_...
```

---

## Usage

### Ingest a document (run once)

```bash
python ingest.py path/to/document.pdf
```

This parses the PDF, sends chunks to Jina AI for embedding, builds a local FAISS index and BM25 index, and saves everything to `./index/`.

### Ask questions

**Interactive mode:**
```bash
python query.py
```

**Single question:**
```bash
python query.py "What was the total revenue in fiscal year 2023?"
```

**With retrieval scores:**
```bash
python query.py "What were the main risk factors? --verbose"
```

### Evaluate

Provide a `.jsonl` file where each line is:
```json
{"question": "What was net income?", "answer": "$2.1 billion", "evidence_page": 47}
```

```bash
python evaluate.py qa_pairs.jsonl
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

### Why hybrid retrieval?

Dense embeddings (Jina) capture semantic similarity but miss exact terms — critical for financial documents where "Note 14" or specific figures need exact matching. BM25 handles these cases. Combining both with Reciprocal Rank Fusion consistently outperforms either alone.

### Why a reranker?

The first-stage retrieval scores each chunk independently. The Jina reranker processes (query, chunk) jointly — equivalent to a cross-encoder — giving much more precise relevance scores. It's too slow to run over the whole index, so it only re-scores the top 20 candidates.

---

## Assumptions and Limitations

- **English language** — Jina's multilingual reranker handles other languages, but the BM25 tokeniser is whitespace-based and works best on English.
- **Text-based PDFs** — PDFS with a lot of visual elements like images or charts will not work well 
- **Groq and Jina rate limits** — the free tier allows 30 requests/minute. For bulk evaluation, add a small sleep between calls if you hit rate limits. Additionally, the free tier from Jina and Groq have a limited number of credits so it was not possible to simply feed in every single document in finance bench run a massive set of queries


## Test results
Testing was conducted manually with both a 300 page novel and a 250 page report from financebench. It was found
that while the pipeline can generally answer specific, relatively direct requests, it has trouble making more abstract inferrences
such as realising that CAPEX is a reflection of capital-intensity. This is likely due to the limitations of the Jina free model,
and the fact that it was trained on a more generic dataset, thus the 768 dimensions of it's embeddings cannot really capture the 
full variations in meaning in the niche area of finance. This was reflected in how the model seemed to perform better with 
the "Sapiens" novel.

## Future Improvements

- custom-trained model for finance document processing
- image/diagram processing step

### Disclaimer
AI was used to assist with code-formatting and syntax-correction, as well as to help debug some parts of the code. All testing
was done manually.
