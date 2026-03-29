# Chunking
CHUNK_SIZE = 512        # tokens per chunk (approximate, whitespace-based)
CHUNK_OVERLAP = 60      # token overlap between consecutive chunks

#  Embedding
# API docs: https://jina.ai/embeddings
JINA_EMBEDDING_MODEL = "jina-embeddings-v2-base-en"
EMBEDDING_DIMENSION = 768   # output dimension for jina-embeddings

#Retrieval
BM25_TOP_K = 20         # candidates from BM25 keyword search
SEMANTIC_TOP_K = 20     # candidates from semantic (vector) search
RETRIEVAL_TOP_K = 20    # candidates passed to the re-ranker after RRF fusion

#Re-ranking
# API docs: https://jina.ai/reranker
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
RERANK_TOP_K = 5        # chunks passed to the LLM after re-ranking

#Generation
# Groq free tier: https://console.groq.com
GROQ_MODEL = "llama-3.1-8b-instant"   # swap for "mixtral-8x7b-32768" for harder questions
MAX_ANSWER_TOKENS = 1024

#Paths
INDEX_DIR = "./index"
