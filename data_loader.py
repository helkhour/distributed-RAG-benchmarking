from datasets import load_dataset
from config import (
    DATASET_NAME,
    SUBSET_NAME,
    CORPUS_NAME,
    CORPUS_LIMIT,
    VERBOSE,
    EMBED_BATCH_SIZE,
)
from db_utils import get_db_connection, setup_vector_index
from system_evaluation import SystemEvaluator
from tqdm import tqdm
import numpy as np
import logging
import time

def compute_doc_stats(dataset, is_corpus=False):
    """Compute document count and size in MB for a dataset or corpus."""
    logger = logging.getLogger(__name__)
    doc_count = 0
    total_bytes = 0
    if is_corpus:
        for entry in dataset:
            if "contents" not in entry:
                logger.error(f"Missing 'contents' field in corpus entry: {entry.keys()}")
                raise KeyError("Expected 'contents' field in KILT corpus entry")
            text = entry["contents"]
            doc_count += 1
            total_bytes += len(text.encode("utf-8"))
    else:
        for entry in dataset:
            documents = entry["output"][0]["provenance"]
            doc_count += len(documents)
            total_bytes += sum(len(doc["title"].encode("utf-8")) for doc in documents)
    return doc_count, total_bytes / (1024 * 1024)

def load_and_store_data(limit=None, embedding_generator=None, embedding_size=None):
    """Load KILT corpus and hotpotqa queries, store in MongoDB with embeddings sequentially."""
    logger = logging.getLogger(__name__)
    
    evaluator = SystemEvaluator()
    if embedding_generator is None or embedding_size is None:
        raise ValueError("Error: Embedding generator and size are required.")
    
    # Initialize all timing keys
    timings = {
        "db_connection": 0.0,
        "dataset_load": 0.0,
        "doc_collection": 0.0,
        "db_clearing": 0.0,
        "document_embedding": 0.0,
        "embedding_preprocessing": 0.0,
        "embedding_encoding": 0.0,
        "sequential_storage": 0.0,
        "document_indexing": 0.0
    }
    stats = {}
        
    # Load KILT corpus (documents) using streaming to avoid large memory usage
    start_time = time.time()
    logger.warning(f"Loading KILT corpus: {CORPUS_NAME} (streaming mode)")
    kilt_corpus = load_dataset(CORPUS_NAME, split="train", streaming=True)
    if CORPUS_LIMIT:
        kilt_corpus = kilt_corpus.take(CORPUS_LIMIT)
    timings["dataset_load"] = time.time() - start_time
    logger.warning(f"Dataset Loading Duration: {timings['dataset_load']:.2f}s")
    
    # Load KILT hotpotqa queries
    logger.warning(f"Loading KILT queries: {DATASET_NAME} with subset: {SUBSET_NAME}")
    # The KILT test split includes the ground-truth provenance so it can be used
    # for evaluation similarly to the train split.
    hotpotqa_dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split="test")
    if limit:
        hotpotqa_dataset = hotpotqa_dataset.select(range(min(limit, len(hotpotqa_dataset))))
    
    if VERBOSE:
        logger.debug("Streaming KILT corpus - sample unavailable")
        logger.debug(f"Sample HotpotQA Entry: {hotpotqa_dataset[0]}")
    
    evaluator.start_monitoring()

    query_doc_count, query_mb = compute_doc_stats(hotpotqa_dataset)

    stats["hotpotqa_entries"] = len(hotpotqa_dataset)
    stats["total_entries"] = len(hotpotqa_dataset)
    stats["hotpotqa_docs"] = query_doc_count
    stats["hotpotqa_size_mb"] = query_mb
    stats["embedding_mode"] = "batch"

    if stats["hotpotqa_entries"] < 100:
        logger.warning(f"Query dataset size is small ({stats['hotpotqa_entries']} entries). Ensure 'limit' is appropriate.")

    # Database operations - connect and clear collection before processing stream
    evaluator.start_monitoring()

    start_time = time.time()
    collection = get_db_connection()
    timings["db_connection"] = time.time() - start_time
    logger.warning(f"Database Connection Duration: {timings['db_connection']:.2f}s")

    start_time = time.time()
    collection.delete_many({})
    timings["db_clearing"] = time.time() - start_time
    logger.warning(f"Database Clearing Duration: {timings['db_clearing']:.2f}s")

    logger.warning("Using batch embedding mode with streaming dataset")

    # Streaming embedding generation and incremental storage
    embedding_norms = []
    corpus_entries = 0
    corpus_doc_count = 0
    corpus_bytes = 0
    batch_docs = []
    batch_texts = []
    meta_entries = []
    batch_size = EMBED_BATCH_SIZE

    start_embed_time = time.time()
    for entry in tqdm(kilt_corpus, desc="Processing corpus", disable=True):  # always disable tqdm
        if "contents" not in entry or "wikipedia_id" not in entry or "title" not in entry:
            logger.error(f"Missing required fields in corpus entry: {entry.keys()}")
            raise KeyError("Expected 'contents', 'wikipedia_id', and 'title' in KILT corpus entry")
        text = entry["contents"]
        corpus_entries += 1
        corpus_doc_count += 1
        corpus_bytes += len(text.encode("utf-8"))

        batch_texts.append(text)
        meta_entries.append(entry)

        if len(batch_texts) >= batch_size:
            embeddings, seq_timings = embedding_generator.generate_embedding(batch_texts)
            timings["embedding_preprocessing"] += seq_timings["query_preprocessing"]
            timings["embedding_encoding"] += seq_timings["query_encoding"]

            for text_item, embedding, meta in zip(batch_texts, embeddings, meta_entries):
                norm = np.linalg.norm(embedding)
                embedding_norms.append(norm)
                if norm < 1e-6:
                    logger.warning(f"Zero or near-zero embedding norm for text: {text_item[:50]}...")
                batch_docs.append({
                    "text": text_item,
                    "embedding": embedding,
                    "wikipedia_id": meta["wikipedia_id"],
                    "wikipedia_title": meta["title"],
                    "source": "kilt_corpus"
                })

            start_store = time.time()
            collection.insert_many(batch_docs)
            timings["sequential_storage"] += time.time() - start_store
            batch_docs = []
            batch_texts = []
            meta_entries = []

    # insert any remaining docs
    if batch_texts:
        embeddings, seq_timings = embedding_generator.generate_embedding(batch_texts)
        timings["embedding_preprocessing"] += seq_timings["query_preprocessing"]
        timings["embedding_encoding"] += seq_timings["query_encoding"]

        for text_item, embedding, meta in zip(batch_texts, embeddings, meta_entries):
            norm = np.linalg.norm(embedding)
            embedding_norms.append(norm)
            if norm < 1e-6:
                logger.warning(f"Zero or near-zero embedding norm for text: {text_item[:50]}...")
            batch_docs.append({
                "text": text_item,
                "embedding": embedding,
                "wikipedia_id": meta["wikipedia_id"],
                "wikipedia_title": meta["title"],
                "source": "kilt_corpus"
            })

    if batch_docs:
        start_store = time.time()
        collection.insert_many(batch_docs)
        timings["sequential_storage"] += time.time() - start_store

    timings["document_embedding"] = time.time() - start_embed_time
    logger.warning(f"Document Embedding Duration: {timings['document_embedding']:.2f}s")
    
    if VERBOSE:
        logger.debug(f"Embedding Preprocessing: {timings['embedding_preprocessing']:.2f}s")
        logger.debug(f"Embedding Encoding: {timings['embedding_encoding']:.2f}s")
        logger.debug(f"Average Embedding Norm: {np.mean(embedding_norms):.4f} ± {np.std(embedding_norms):.4f}")

    logger.warning("Setting up vector index")
    start_time = time.time()
    timings["document_indexing"] = setup_vector_index(collection, embedding_size)
    logger.warning(f"Document Indexing Duration: {timings['document_indexing']:.2f}s")

    evaluator.log_resources("After Index Setup")
    total_docs = collection.count_documents({})

    stats["corpus_entries"] = corpus_entries
    stats["corpus_docs"] = corpus_doc_count
    stats["corpus_size_mb"] = corpus_bytes / (1024 * 1024)
    stats["total_docs"] = corpus_doc_count
    stats["total_size_mb"] = stats["corpus_size_mb"]
    stats["docs_stored"] = corpus_doc_count
    stats["db_entries"] = total_docs

    logger.warning(f"KILT Corpus: {stats['corpus_entries']} entries, {stats['corpus_docs']} docs, {stats['corpus_size_mb']:.2f} MB")
    logger.warning(f"HotpotQA Queries: {stats['hotpotqa_entries']} entries, {stats['hotpotqa_docs']} docs, {stats['hotpotqa_size_mb']:.2f} MB")
    logger.warning(f"Total Documents: {stats['total_docs']} docs, {stats['total_size_mb']:.2f} MB")
    logger.warning(f"Stored {stats['docs_stored']} documents in batches, {stats['db_entries']} entries in MongoDB.")

    return collection, hotpotqa_dataset, timings, stats