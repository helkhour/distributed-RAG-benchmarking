MODEL_CONFIGS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_size": 384,
        "parameters": 22_700_000,
    },
    "BAAI/bge-base-en-v1.5": {
        "base_model": "BAAI/bge-base-en-v1.5",
        "embedding_size": 768,
        "parameters": 110_000_000,
        "normalize_embeddings": True,
    },
    "thenlper/gte-base": {
        "base_model": "thenlper/gte-base",
        "embedding_size": 768,
        "parameters": 110_000_000,
    },
}

DB_URI = "mongodb://localhost:32768/?directConnection=true"
DB_NAME = "rag_db"
COLLECTION_NAME = "kilt_docs"
K = 3
DATASET_NAME = "facebook/kilt_tasks"
SUBSET_NAME = "hotpotqa"
CORPUS_NAME = "corag/kilt-corpus"
CORPUS_LIMIT = 10000
limit=10000
VERBOSE = False


# Database indexing configuration
DB_CONFIG = {
    # Allow more time for the search index to be created on large corpora
    "index_timeout_s": 600
}

# Batch size for embedding operations
EMBED_BATCH_SIZE = 32
