from pymongo import MongoClient
import numpy as np
import logging
import time
from config import BATCH_SIZE

logger = logging.getLogger(__name__)

def retrieve_top_k(query_embeddings, collection, num_candidates):
    """Retrieve top-K documents from MongoDB using vector search for a batch of query embeddings."""
    logger.info(f"Starting retrieval with num_candidates={num_candidates} for {len(query_embeddings)} queries.")
    logger.debug(f"Retrieval input: query_embeddings type={type(query_embeddings)}, len={len(query_embeddings)}, sample dims={len(query_embeddings[0]) if query_embeddings and query_embeddings[0] else 0}")
    
    start_time = time.time()
    results = []
    total_search_time = 0.0
    
    # Ensure query_embeddings is a list of lists
    if isinstance(query_embeddings, np.ndarray):
        query_embeddings = query_embeddings.tolist()
    elif isinstance(query_embeddings[0], np.ndarray):
        query_embeddings = [emb.tolist() for emb in query_embeddings]
    elif not isinstance(query_embeddings[0], list):
        query_embeddings = [query_embeddings]
    logger.debug(f"Processed query_embeddings: type={type(query_embeddings)}, len={len(query_embeddings)}, first query dims={len(query_embeddings[0]) if query_embeddings else 0}")
    
    # MongoDB vector search pipeline
    pipeline = lambda query_vector: [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
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
    
    # Log collection stats
    doc_count = collection.count_documents({})
    logger.debug(f"Collection contains {doc_count} documents")
    
    # Verify vector index exists
    indexes = list(collection.list_search_indexes())
    logger.debug(f"Available indexes: {indexes}")
    if not any(index["name"] == "vector_index" and index["status"] == "READY" for index in indexes):
        logger.error("Vector index 'vector_index' not found or not READY. Cannot perform vector search.")
        return [[] for _ in query_embeddings], {"vector_search": 0.0}
    
    for i in range(0, len(query_embeddings), BATCH_SIZE):
        batch = query_embeddings[i:i + BATCH_SIZE]
        batch_results = []
        batch_start = time.time()
        for j, query_embedding in enumerate(batch):
            query_idx = i + j + 1
            logger.debug(f"Processing query {query_idx} with vector (first 5 dims): type={type(query_embedding)}, value={query_embedding[:5]}")
            try:
                search_results = list(collection.aggregate(pipeline(query_embedding)))
                logger.debug(f"Query {query_idx} raw search results: type={type(search_results)}, len={len(search_results)}, first item={search_results[0] if search_results else 'No results'}")
                
                validated_results = []
                if not search_results:
                    logger.warning(f"Query {query_idx} returned no results.")
                else:
                    for res in search_results:
                        if not isinstance(res, dict):
                            logger.error(f"Query {query_idx} search result is not a dictionary: type={type(res)} - value={res}")
                            continue
                        if "wikipedia_id" not in res or "wikipedia_title" not in res:
                            logger.error(f"Query {query_idx} missing required fields in search result: {res.keys()}")
                            continue
                        validated_results.append(res)
                batch_results.append(validated_results)
            except Exception as e:
                logger.error(f"Vector search failed for query {query_idx}: {str(e)}")
                batch_results.append([])
        batch_search_time = time.time() - batch_start
        total_search_time += batch_search_time
        logger.debug(f"Batch {i//BATCH_SIZE + 1} retrieved {len(batch_results)} queries in {batch_search_time:.4f}s")
        results.extend(batch_results)
    
    # Log retrieved document fields for the first non-empty result
    for res in results:
        if res:
            logger.debug(f"Retrieved document fields: {list(res[0].keys())}")
            break
    else:
        logger.warning("No documents retrieved from vector search.")
    
    logger.info(f"Vector search completed, retrieved {sum(len(res) for res in results)} documents for {len(results)} queries in {total_search_time:.4f}s")
    logger.debug(f"Final results structure: type={type(results)}, len={len(results)}, first batch={results[0] if results else []}")
    return results, {"vector_search": total_search_time}