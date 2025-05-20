from config import K, MODEL_CONFIGS, VERBOSE
from system_evaluation import SystemEvaluator
import time
import logging
import numpy as np

LOG_LIMIT = 4

def retrieve_top_k(query_embedding, collection, num_candidates=100, model_name=None):
    """Retrieve top K documents based on vector similarity with timing."""
    logger = logging.getLogger(__name__)
    evaluator = SystemEvaluator()

    if model_name:
        expected_dim = MODEL_CONFIGS[model_name]["embedding_size"]
        if len(query_embedding) != expected_dim:
            logger.error(f"Query embedding dimension mismatch: expected {expected_dim}, got {len(query_embedding)}")
            raise ValueError(f"Query embedding dimension mismatch: expected {expected_dim}, got {len(query_embedding)}")
        
        norm = np.linalg.norm(query_embedding)
        if norm < 1e-6:
            logger.warning(f"Query embedding has zero or near-zero norm: {norm:.4f}")

    
    start_time = time.time()
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "index": "vector_index",
                "limit": K,
                "numCandidates": num_candidates
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "question_ids": 1,
                "source": 1
            }
        }
    ])
    results_list = list(results)
    search_duration = time.time() - start_time
    

    if not results_list:
        logger.warning(f"No results retrieved for query embedding (norm: {norm:.4f})")

    if VERBOSE:
        logger.debug(f"Retrieved {len(results_list)} documents in {search_duration:.2f}s")
    
    return results_list, {"vector_search": search_duration}