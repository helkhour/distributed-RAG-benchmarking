MODEL_CONFIGS = {
    "mixedbread-ai/mxbai-embed-large-v1-256": {
        "base_model": "mixedbread-ai/mxbai-embed-large-v1",
        "embedding_size": 256,
        "parameters": 335_000_000
    },
    "mixedbread-ai/mxbai-embed-large-v1-512": {
        "base_model": "mixedbread-ai/mxbai-embed-large-v1",
        "embedding_size": 512,
        "parameters": 335_000_000
    },
    "mixedbread-ai/mxbai-embed-large-v1-1024": {
        "base_model": "mixedbread-ai/mxbai-embed-large-v1",
        "embedding_size": 1024,
        "parameters": 335_000_000
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_size": 384,
        "parameters": 22_700_000
    },
    "intfloat/e5-small-v2": {
        "base_model": "intfloat/e5-small-v2",
        "embedding_size": 384,
        "parameters": 33_000_000
    },
    "thenlper/gte-base-384": {
        "base_model": "thenlper/gte-base",
        "embedding_size": 384,
        "parameters": 110_000_000
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "base_model": "sentence-transformers/all-mpnet-base-v2",
        "embedding_size": 768,
        "parameters": 110_000_000
    },
    "BAAI/bge-base-en-v1.5": {
        "base_model": "BAAI/bge-base-en-v1.5",
        "embedding_size": 768,
        "parameters": 110_000_000
    },
    "thenlper/gte-base": {
        "base_model": "thenlper/gte-base",
        "embedding_size": 768,
        "parameters": 110_000_000
    }
}

DB_CONFIG = {
    "index_timeout_s": 300
}

DB_URI = "mongodb://localhost:32768/?directConnection=true"
DB_NAME = "rag_db"
COLLECTION_NAME = "kilt_docs"
K = 3
numCandidates = 100
DATASET_NAME = "facebook/kilt_tasks"
SUBSET_NAME = "hotpotqa"
CORPUS_NAME = "corag/kilt-corpus"
CORPUS_LIMIT = 100 #None for no limit
limit = 100
VERBOSE = False
BATCH_SIZE = 64
