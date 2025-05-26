from pymongo import MongoClient
import numpy as np
import logging
import time

def retrieve_top_k(query_embedding, collection, num_candidates):
    """Retrieve top-K documents from MongoDB using vector search."""
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    # Ensure query_embedding is a list
    query_embedding = np.array(query_embedding, dtype=np.float32).tolist()
    
    # MongoDB vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates * 10,
                "limit": num_candidates
            }
        },
        {
            "$project": {
                "text": 1,
                "embedding": 1,
                "wikipedia_id": 1,
                "wikipedia_title": 1,
                "source": 1,
                "_id": 0
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
        if not results:
            logger.warning("No documents retrieved from vector search.")
        else:
            logger.debug(f"Retrieved document fields: {list(results[0].keys())}")
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        results = []
    
    search_time = time.time() - start_time
    logger.info(f"Vector search retrieved {len(results)} documents in {search_time:.4f}s")
    
    return results, {"vector_search": search_time}