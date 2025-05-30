import time
from pymongo import MongoClient
from config import DB_URI, DB_NAME, VERBOSE, BATCH_SIZE, COLLECTION_NAME, K
from retrieval import retrieve_top_k
import logging
import timeout_decorator
from tqdm import tqdm

logger = logging.getLogger(__name__)

def process_query(entries, collection, embedding_generator, num_candidates):
    """Process a batch of queries and return metrics with timing."""
    logger.info(f"Processing batch of {len(entries)} queries.")
    #logger.debug(f"Batch entries: {[entry['input'][:50] for entry in entries]}")
    
    queries = [entry["input"] for entry in entries]
    #logger.debug(f"Extracted {len(queries)} queries from batch.")
    
    start_time = time.time()
    try:
        # Generate embeddings for queries
        query_embeddings_list, embed_timings = embedding_generator.generate_embedding(queries)
        # logger.debug(f"Generated embeddings: type={type(query_embeddings_list)}, len={len(query_embeddings_list)}, sample dims={len(query_embeddings_list[0]) if query_embeddings_list else 0}")
        # logger.debug(f"Sample embedding (first 5 dims): {query_embeddings_list[0][:5] if query_embeddings_list and query_embeddings_list[0] else 'No embeddings'}")
        
        # Retrieve documents
        results, search_timings = retrieve_top_k(query_embeddings_list, collection, num_candidates)
        # logger.debug(f"Retrieval results: type={type(results)}, len={len(results) if isinstance(results, (list, tuple)) else 'N/A'}, first item={results[0] if isinstance(results, (list, tuple)) and results else 'N/A'}")
        if not isinstance(results, list):
            logger.error(f"retrieve_top_k returned non-list type: {type(results)} - Value: {results}")
            raise TypeError(f"Expected list from retrieve_top_k, got {type(results)}")
    except Exception as e:
        logger.error(f"Error processing batch of {len(queries)} queries: {str(e)}")
        return [{
            "latency": 0,
            "results": [],
            "retrieved_docs": [],
            "relevant_count": 0,
            "query": query,
            "has_results": False,
            "timings": {"query_preprocessing": 0, "query_encoding": 0, "vector_search": 0}
        } for query in queries]
    
    latency = time.time() - start_time
    avg_latency = latency / len(queries) if queries else 0
    # logger.debug(f"Batch processing latency: {latency:.4f}s, avg_latency={avg_latency:.4f}s")
    
    batch_results = []
    for i, (query, query_embedding, result) in enumerate(zip(queries, query_embeddings_list, results)):
        query_idx = i + 1
        # logger.debug(f"Processing query {query_idx} in batch: {query[:50]}...")
        # logger.debug(f"Query {query_idx} retrieved results: type={type(result)}, value={result}")
        
        # Extract ground-truth Wikipedia IDs/titles from provenance
        relevant_docs = []
        provenance = []
        for output in entries[i]["output"]:
            provenance.extend(output.get("provenance", []))
        for prov in provenance:
            if prov.get("wikipedia_id") or prov.get("title"):
                relevant_docs.append({
                    "wikipedia_id": prov.get("wikipedia_id", ""),
                    "wikipedia_title": prov.get("title", "")
                })
        #logger.debug(f"Query {query_idx} relevant docs: {len(relevant_docs)}")
        
        # Validate retrieved documents
        retrieved_docs = []
        if not isinstance(result, list):
            logger.error(f"Retrieved result is not a list for query {query_idx}: type={type(result)}, value={result}")
            result = []
        for r in result:
            if not isinstance(r, dict):
                logger.error(f"Retrieved document is not a dictionary for query {query_idx}: type={type(r)}, value={r}")
                continue
            if "wikipedia_id" not in r or "wikipedia_title" not in r:
                logger.error(f"Missing 'wikipedia_id' or 'wikipedia_title' in retrieved document for query {query_idx}: {r.keys()}")
                continue
            retrieved_docs.append({
                "wikipedia_id": r["wikipedia_id"],
                "wikipedia_title": r["wikipedia_title"]
            })
        # logger.debug(f"Query {query_idx} validated retrieved docs: {len(retrieved_docs)}")
        
        # Count relevant retrieved documents
        relevant_count = sum(
            any(
                doc["wikipedia_id"] == ref["wikipedia_id"] or
                doc["wikipedia_title"] == ref["wikipedia_title"]
                for ref in relevant_docs
            )
            for doc in retrieved_docs
        )
        # logger.debug(f"Query {query_idx} relevant count: {relevant_count}")

        timings = {
            "query_preprocessing": embed_timings["query_preprocessing"] / len(queries),
            "query_encoding": embed_timings["query_encoding"] / len(queries),
            "vector_search": search_timings["vector_search"] / len(queries)
        }
        # logger.debug(f"Query {query_idx} timings: {timings}")
        
        # if VERBOSE:
        #     logger.debug(f"Query: {query[:50]}... processed in {avg_latency:.4f}s")
        #     logger.debug(f"Retrieved {len(retrieved_docs)} docs, {relevant_count} relevant")
        
        batch_results.append({
            "latency": avg_latency,
            "results": result,
            "retrieved_docs": retrieved_docs,
            "relevant_count": relevant_count,
            "query": query,
            "has_results": bool(result),
            "timings": timings
        })

    # logger.debug(f"Batch processing completed with {len(batch_results)} results.")
    return batch_results

@timeout_decorator.timeout(600, timeout_exception=TimeoutError)
def evaluate_retrieval_performance(dataset, collection, embedding_generator, k=K, num_candidates=50):
    """Evaluate retrieval performance with batched queries and timing."""
    logger = logging.getLogger(__name__)
    
    total_queries = len(dataset)
    logger.info(f"Total queries to process: {total_queries}")
    logger.debug(f"Dataset sample: {dataset[0]['input'][:50]}...") if total_queries > 0 else logger.debug("Dataset is empty")
    
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

    logger.info(f"Starting batch evaluation with {total_queries} queries, {num_candidates} candidates")
    #logger.debug(f"Evaluation parameters: k={k}, batch_size={BATCH_SIZE}")
    start_total_time = time.time()
    
    start_overhead = time.time()
    results = []
    try:
        for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Processing query batches", disable=not VERBOSE):
            batch = dataset[i:i + BATCH_SIZE]
            # logger.debug(f"Processing batch {i//BATCH_SIZE + 1} with {len(batch)} queries")
            try:
                batch_results = process_query(batch, collection, embedding_generator, num_candidates)
                results.extend(batch_results)
                if (i // BATCH_SIZE + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch)}/{total_queries} queries")
            except TimeoutError:
                logger.warning(f"Timeout processing batch starting at query {i + 1}. Saving partial results.")
                for entry in batch:
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
                logger.error(f"Error in batch {i//BATCH_SIZE + 1}: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Evaluation loop failed: {e}")
        raise
    
    query_timings["evaluation_overhead"] = time.time() - start_overhead
    #logger.debug(f"Evaluation overhead: {query_timings['evaluation_overhead']:.2f}s")
    #logger.info(f"Completed evaluation for {len(results)} queries")
    #logger.debug(f"First result sample: {results[0] if results else 'No results'}")
    
    for res in results:
        total_latency += res["latency"]
        if res["has_results"]:
            queries_with_results += 1
        total_relevant += res["relevant_count"]
        total_retrieved += min(len(res["retrieved_docs"]), k)
        for key in query_timings:
            if key in res["timings"]:
                query_timings[key] += res["timings"][key]

    total_time = time.time() - start_total_time
    #logger.debug(f"Total evaluation time: {total_time:.2f}s")

    retrieval_success = queries_with_results / total_queries if total_queries > 0 else 0
    avg_precision = total_relevant / (k * total_queries) if total_retrieved > 0 else 0
    avg_latency = total_latency / total_queries if total_queries > 0 else 0
    throughput = total_queries / total_time if total_time > 0 else 0

    client = MongoClient(DB_URI)
    db = client[DB_NAME]
    stats = db.command("collStats", COLLECTION_NAME)
    db_size_mb = stats["size"] / (1024 * 1024)
    doc_count = collection.count_documents({})
    #logger.debug(f"Database stats: size={db_size_mb:.2f} MB, doc_count={doc_count}")

    logger.info("\n=== Retrieval Performance Summary ===")
    logger.info(f"Collection Size: {db_size_mb:.2f} MB ({doc_count} documents)")
    logger.info(f"Retrieval Success: {retrieval_success:.2%} ({queries_with_results}/{total_queries} queries)")
    logger.info(f"Average Precision@{k}: {avg_precision:.2%} ({total_relevant}/{k * total_queries} docs)")
    logger.info(f"Average Latency: {avg_latency:.4f} seconds/query")
    logger.info(f"Throughput: {throughput:.2f} queries/second")
    logger.info(f"Total Query Preprocessing: {query_timings['query_preprocessing']:.2f}s")
    logger.info(f"Total Query Encoding: {query_timings['query_encoding']:.2f}s")
    logger.info(f"Total Vector Search: {query_timings['vector_search']:.2f}s")

    if avg_precision == 0 and VERBOSE:
        logger.info("Note: Zero precision is expected with a small corpus size (e.g., 100 documents). Consider increasing CORPUS_LIMIT for better evaluation.")
        logger.debug(f"Debug: total_relevant={total_relevant}, k={k}, total_queries={total_queries}")

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