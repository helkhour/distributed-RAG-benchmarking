import logging
from transformers.utils import logging as transformers_logging
from sentence_transformers import SentenceTransformer
from data_loader import load_and_store_data
from evaluation import evaluate_retrieval_performance
from embedding_utils import EmbeddingGenerator
from config import MODEL_CONFIGS, VERBOSE, limit, K, numCandidates
from system_evaluation import SystemEvaluator

def run_study(model_name, embedding_size):
    logger = logging.getLogger(__name__)
    logger.info(f"\n=== Evaluating Model: {model_name} ===")
    
    evaluator = SystemEvaluator()
    results = {}
    logger.info("Running batch embedding mode")
    embedding_generator = EmbeddingGenerator(model_name, embedding_size, quantize=False)
    
    evaluator.start_monitoring()
    collection, dataset_raw, data_timings, data_stats = load_and_store_data(
        limit=limit, embedding_generator=embedding_generator, embedding_size=embedding_size
    )
    data_duration, data_cpu_delta = evaluator.end_monitoring("Data Load (batch)")

    # Ensure dataset is a list of dictionaries
    dataset = [dict(item) for item in dataset_raw]

    logger.info(f"\nRunning evaluation (top-{K}, numCandidates={numCandidates}, batch)...")
    evaluator.start_monitoring()
    metrics_k = evaluate_retrieval_performance(
        dataset, collection, embedding_generator, k=K, num_candidates=numCandidates
    )
    eval_duration_k, eval_cpu_delta_k = evaluator.end_monitoring(f"Evaluation (top-{K}, batch)")
    
    if metrics_k["avg_precision"] == 0:
        logger.warning(f"Zero precision detected for {model_name} (batch). Check embedding dimensions and vector index.")
    
    results["batch"] = {
        "data_timings": data_timings,
        "data_stats": data_stats,
        "metrics_k": metrics_k,
        "eval_duration_k": eval_duration_k
    }
    
    return results

def summarize_results(model_name, results):
    logger = logging.getLogger(__name__)
    config = MODEL_CONFIGS[model_name]
    embedding_size = config["embedding_size"]
    parameters = config.get("parameters", "Unknown")

    timings = results["batch"]
    data_timings = timings["data_timings"]
    data_stats = timings["data_stats"]
    metrics_k = timings["metrics_k"]

    pipeline_timings = {
        "Database Connection": data_timings["db_connection"],
        "Dataset Loading": data_timings["dataset_load"],
        "Document Collection": data_timings["doc_collection"],
        "Database Clearing": data_timings["db_clearing"],
        "Document Embedding": data_timings["document_embedding"],
        "Document Storage": data_timings["sequential_storage"],
        "Document Indexing": data_timings["document_indexing"]
    }

    total_pipeline_time = sum(pipeline_timings.values())
    proportions = {key: (value / total_pipeline_time * 100) if total_pipeline_time > 0 else 0.0 for key, value in pipeline_timings.items()}

    logger.info(f"\n=== Summary for Model: {model_name} (Batch Embedding) ===")
    logger.debug(f"Summary data: embedding_size={embedding_size}, parameters={parameters}")
    logger.info(f"Model Specs:")
    if parameters != "Unknown":
        logger.info(f"  Parameters: {parameters:,} (~{parameters // 1_000_000}M)")
    else:
        logger.info(f"  Parameters: {parameters}")
    logger.info(f"  Embedding Size: {embedding_size}")
    
    logger.info(f"\nDataset and Database Statistics:")
    logger.info(f"{'Metric':<40} {'Value':<20}")
    logger.info("-" * 60)
    logger.info(f"{'KILT Corpus Entries':<40} {data_stats['corpus_entries']:<20}")
    logger.info(f"{'HotpotQA Queries':<40} {data_stats['hotpotqa_entries']:<20}")
    logger.info(f"{'Total Query Entries':<40} {data_stats['total_entries']:<20}")
    logger.info(f"{'KILT Corpus Documents':<40} {data_stats['corpus_docs']:<20}")
    logger.info(f"{'HotpotQA Provenance Docs':<40} {data_stats['hotpotqa_docs']:<20}")
    logger.info(f"{'Total Documents':<40} {data_stats['total_docs']:<20}")
    logger.info(f"{'KILT Corpus Size (MB)':<40} {data_stats['corpus_size_mb']:<20.2f}")
    logger.info(f"{'HotpotQA Size (MB)':<40} {data_stats['hotpotqa_size_mb']:<20.2f}")
    logger.info(f"{'Total Dataset Size (MB)':<40} {data_stats['total_size_mb']:<20.2f}")
    logger.info(f"{'Documents Stored (Batch)':<40} {data_stats['docs_stored_batch']:<20}")
    logger.info(f"{'Database Entries':<40} {data_stats['db_entries']:<20}")
    logger.info(f"{'Database Size (MB)':<40} {metrics_k['db_size_mb']:<20.2f}")
    logger.info(f"{'Total Queries Evaluated':<40} {metrics_k['total_queries']:<20}")
    logger.info(f"{'Embedding Mode':<40} {data_stats['embedding_mode']:<20}")
    
    logger.info(f"\nTiming Breakdown (RAG Pipeline, Total Time: {total_pipeline_time:.2f}s):")
    logger.info(f"{'Process':<30} {'Time (s)':<12} {'Proportion (%)':<15}")
    logger.info("-" * 57)
    for key, duration in pipeline_timings.items():
        logger.info(f"{key:<30} {duration:<12.2f} {proportions[key]:<15.2f}")
    
    logger.info(f"\nEvaluation Metrics (top-{K}, numCandidates={numCandidates}):")
    logger.info(f"  Latency (s/query): {metrics_k['avg_latency']:.4f}")
    logger.info(f"  Throughput (q/s): {metrics_k['throughput']:.2f}")
    logger.info(f"  Precision@{K} (%): {metrics_k['avg_precision'] * 100:.2f}")

def main():
    transformers_logging.set_verbosity_error()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("rag_evaluation.log"),
            logging.StreamHandler()
        ]
    )   
    # Suppress INFO logs from specific modules
    for logger_name in [
        'pymongo', 'pymongo.topology', 'pymongo.serverSelection', 'pymongo.connection', 'pymongo.command',
        'urllib3', 'urllib3.connectionpool',
        'filelock',
        'fsspec', 'fsspec.local',
        'sentence_transformers.SentenceTransformer',
        'system_evaluation',
        'embedding_utils',  # Suppress embedding_utils INFO logs
        'evaluation',      # Suppress evaluation INFO logs
        'retrieval'        # Suppress retrieval INFO logs
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Starting RAG retrieval evaluation...")
    logger.debug(f"Configured models: {list(MODEL_CONFIGS.keys())}")
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        # "mixedbread-ai/mxbai-embed-large-v1-512"
    ]

    for model_name in models:
        embedding_size = MODEL_CONFIGS[model_name]["embedding_size"]
        logger.debug(f"Running study for model: {model_name}, embedding_size: {embedding_size}")
        results = run_study(model_name, embedding_size)
        summarize_results(model_name, results)

if __name__ == "__main__":
    main()