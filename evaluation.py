import time
from pymongo import MongoClient
from config import DB_URI, DB_NAME, VERBOSE
from retrieval import retrieve_top_k
import logging
import timeout_decorator

def process_query(entry, collection, embedding_generator, num_candidates):
    """Process a single query and return metrics with timing."""
    logger = logging.getLogger(__name__)
    
    query = entry["input"]
    logger.debug(f"Processing query: {query[:50]}...")
    
    start_time = time.time()
    try:
        query_embedding_list, embed_timings = embedding_generator.generate_embedding(query)
        query_embedding = query_embedding_list[0]
        results, search_timings = retrieve_top_k(query_embedding, collection, num_candidates)
    except Exception as e:
        logger.error(f"Error processing query '{query[:50]}...': {str(e)}")
        return {
            "latency": 0,
            "results": [],
            "retrieved_docs": [],
            "relevant_count": 0,
            "query": query,
            "has_results": False,
            "timings": {"query_preprocessing": 0, "query_encoding": 0, "vector_search": 0}
        }
    
    latency = time.time() - start_time

    # Extract ground-truth Wikipedia IDs/titles from provenance
    relevant_docs = []
    for output in entry["output"]:
        for prov in output.get("provenance", []):
            if prov.get("wikipedia_id") or prov.get("title"):
                relevant_docs.append({
                    "wikipedia_id": prov.get("wikipedia_id", ""),
                    "wikipedia_title": prov.get("title", "")
                })
    
    # Validate retrieved documents
    retrieved_docs = []
    for r in results:
        if "wikipedia_id" not in r or "wikipedia_title" not in r:
            logger.error(f"Missing 'wikipedia_id' or 'wikipedia_title' in retrieved document: {r.keys()}")
            continue
        retrieved_docs.append({
            "wikipedia_id": r["wikipedia_id"],
            "wikipedia_title": r["wikipedia_title"]
        })
    
    # Count relevant retrieved documents
    relevant_count = sum(
        any(
            doc["wikipedia_id"] == ref["wikipedia_id"] or
            doc["wikipedia_title"] == ref["wikipedia_title"]
            for ref in relevant_docs
        )
        for doc in retrieved_docs
    )
        
    timings = {
        "query_preprocessing": embed_timings["query_preprocessing"],
        "query_encoding": embed_timings["query_encoding"],
        "vector_search": search_timings["vector_search"]
    }
    
    if VERBOSE:
        logger.debug(f"Query: {query[:50]}... processed in {latency:.4f}s")
    
    return {
        "latency": latency,
        "results": results,
        "retrieved_docs": retrieved_docs,
        "relevant_count": relevant_count,
        "query": query,
        "has_results": bool(results),
        "timings": timings
    }

@timeout_decorator.timeout(60, timeout_exception=TimeoutError)  # 60s timeout per query
def evaluate_retrieval_performance(dataset, collection, embedding_generator, num_candidates=100):
    """Evaluate retrieval performance with sequential queries and timing."""
    logger = logging.getLogger(__name__)
    
    total_queries = len(dataset)
    logger.info(f"Total queries to process: {total_queries}")
    
    queries_with_results = 0
    total_relevant = 0
    total_retrieved = 0
    total_latency = 0
    query_timings = {
        "query_preprocessing": 0.0,
        "query_encoding": 0.0,
        "vector_search": 0.0,
        "evaluation_overhead": 0.0
    }

    logger.info(f"Starting sequential evaluation with {total_queries} queries, {num_candidates} candidates")
    start_total_time = time.time()
    
    start_overhead = time.time()
    results = []
    try:
        for i, entry in enumerate(dataset, 1):
            try:
                result = process_query(entry, collection, embedding_generator, num_candidates)
                results.append(result)
                if i % 10 == 0:  # Log progress every 10 queries
                    logger.info(f"Processed {i}/{total_queries} queries")
            except TimeoutError:
                logger.error(f"Timeout processing query {i}: {entry['input'][:50]}...")
                results.append({
                    "latency": 0,
                    "results": [],
                    "retrieved_docs": [],
                    "relevant_count": 0,
                    "query": entry["input"],
                    "has_results": False,
                    "timings": {"query_preprocessing": 0, "query_encoding": 0, "vector_search": 0}
                })
    except Exception as e:
        logger.error(f"Evaluation loop failed: {str(e)}")
        raise
    
    query_timings["evaluation_overhead"] = time.time() - start_overhead
    logger.info(f"Evaluation Overhead Duration: {query_timings['evaluation_overhead']:.2f}s")
    logger.info(f"Completed evaluation for {len(results)} queries")

    for res in results:
        total_latency += res["latency"]
        if res["has_results"]:
            queries_with_results += 1
        total_relevant += res["relevant_count"]
        total_retrieved += len(res["retrieved_docs"])
        for key in query_timings:
            if key in res["timings"]:
                query_timings[key] += res["timings"][key]

    total_time = time.time() - start_total_time

    retrieval_success = queries_with_results / total_queries if total_queries > 0 else 0
    avg_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
    avg_latency = total_latency / total_queries if total_queries > 0 else 0
    throughput = total_queries / total_time if total_time > 0 else 0

    client = MongoClient(DB_URI)
    db = client[DB_NAME]
    stats = db.command("dbStats")
    db_size_mb = stats["dataSize"] / (1024 * 1024)
    doc_count = collection.count_documents({})

    logger.info("\n=== Retrieval Performance Summary ===")
    logger.info(f"Database Size: {db_size_mb:.2f} MB ({doc_count} documents)")
    logger.info(f"Retrieval Success: {retrieval_success:.2%} ({queries_with_results}/{total_queries} queries)")
    logger.info(f"Average Precision: {avg_precision:.2%} ({total_relevant}/{total_retrieved} docs)")
    logger.info(f"Average Latency: {avg_latency:.4f} seconds/query")
    logger.info(f"Throughput: {throughput:.2f} queries/second (sequential processing, {num_candidates} candidates)")
    logger.info(f"Total Query Preprocessing: {query_timings['query_preprocessing']:.2f}s")
    logger.info(f"Total Query Encoding: {query_timings['query_encoding']:.2f}s")
    logger.info(f"Total Vector Search: {query_timings['vector_search']:.2f}s")

    return {
        "retrieval_success": retrieval_success,
        "avg_precision": avg_precision,
        "avg_latency": avg_latency,
        "throughput": throughput,
        "db_size_mb": db_size_mb,
        "doc_count": doc_count,
        "total_time": total_time,
        "query_timings": query_timings,
        "total_queries": total_queries
    }