from datasets import load_dataset
from config import DATASET_NAME, SUBSET_NAME, CORPUS_NAME, CORPUS_LIMIT, VERBOSE
from db_utils import get_db_connection, setup_vector_index
from system_evaluation import SystemEvaluator
from tqdm import tqdm
import torch
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
        
    # Load KILT corpus (documents)
    start_time = time.time()
    split = "train" if limit is None else f"train[:{limit}]"
    logger.warning(f"Loading KILT corpus: {CORPUS_NAME} with split: {split}")
    kilt_corpus = load_dataset(CORPUS_NAME, split=split)
    if CORPUS_LIMIT:
        kilt_corpus = kilt_corpus.select(range(min(CORPUS_LIMIT, len(kilt_corpus))))
    timings["dataset_load"] = time.time() - start_time
    logger.warning(f"Dataset Loading Duration: {timings['dataset_load']:.2f}s")
    
    # Load KILT hotpotqa queries
    logger.warning(f"Loading KILT queries: {DATASET_NAME} with subset: {SUBSET_NAME}")
    hotpotqa_dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split="train")
    if limit:
        hotpotqa_dataset = hotpotqa_dataset.select(range(min(limit, len(hotpotqa_dataset))))
    
    if VERBOSE:
        logger.debug(f"Sample KILT Corpus Entry: {kilt_corpus[0]}")
        logger.debug(f"Sample HotpotQA Entry: {hotpotqa_dataset[0]}")
    
    evaluator.start_monitoring()
    
    corpus_doc_count, corpus_mb = compute_doc_stats(kilt_corpus, is_corpus=True)
    query_doc_count, query_mb = compute_doc_stats(hotpotqa_dataset)
    
    stats["corpus_entries"] = len(kilt_corpus)
    stats["hotpotqa_entries"] = len(hotpotqa_dataset)
    stats["total_entries"] = len(hotpotqa_dataset)
    stats["corpus_docs"] = corpus_doc_count
    stats["hotpotqa_docs"] = query_doc_count
    stats["total_docs"] = corpus_doc_count
    stats["corpus_size_mb"] = corpus_mb
    stats["hotpotqa_size_mb"] = query_mb
    stats["total_size_mb"] = corpus_mb
    stats["embedding_mode"] = "sequential"
    
    logger.warning(f"KILT Corpus: {stats['corpus_entries']} entries, {corpus_doc_count} docs, {corpus_mb:.2f} MB")
    logger.warning(f"HotpotQA Queries: {stats['hotpotqa_entries']} entries, {query_doc_count} docs, {query_mb:.2f} MB")
    logger.warning(f"Total Documents: {stats['total_docs']} docs, {stats['total_size_mb']:.2f} MB")
    
    if stats["hotpotqa_entries"] < 100:
        logger.warning(f"Query dataset size is small ({stats['hotpotqa_entries']} entries). Ensure 'limit' is appropriate.")
    
    # Collect all documents from KILT corpus
    start_time = time.time()
    all_docs = []
    for entry in kilt_corpus:
        if "contents" not in entry or "wikipedia_id" not in entry or "title" not in entry:
            logger.error(f"Missing required fields in corpus entry: {entry.keys()}")
            raise KeyError("Expected 'contents', 'wikipedia_id', and 'title' in KILT corpus entry")
        text = entry["contents"]
        all_docs.append({
            "text": text,
            "wikipedia_id": entry["wikipedia_id"],
            "wikipedia_title": entry["title"],
            "source": "kilt_corpus"
        })
    timings["doc_collection"] = time.time() - start_time
    logger.warning(f"Document Collection Duration: {timings['doc_collection']:.2f}s")
    logger.warning(f"Total Documents to Embed: {len(all_docs)}")
    
    # Sequential embedding generation
    evaluator.start_monitoring()
    start_time = time.time()
    docs = []
    embedding_norms = []
    
    logger.warning("Using sequential embedding mode")
    for item in tqdm(all_docs, desc="Generating sequential embeddings", disable=True):  # Always disable tqdm
        embedding, seq_timings = embedding_generator.generate_embedding(item["text"])
        embedding = embedding[0]
        norm = np.linalg.norm(embedding)
        embedding_norms.append(norm)
        if norm < 1e-6:
            logger.warning(f"Zero or near-zero embedding norm for text: {item['text'][:50]}...")
        timings["embedding_preprocessing"] += seq_timings["query_preprocessing"]
        timings["embedding_encoding"] += seq_timings["query_encoding"]
        docs.append({
            "text": item["text"],
            "embedding": embedding,
            "wikipedia_id": item["wikipedia_id"],
            "wikipedia_title": item["wikipedia_title"],
            "source": item["source"]
        })
    timings["document_embedding"] = time.time() - start_time
    logger.warning(f"Document Embedding Duration: {timings['document_embedding']:.2f}s")
    
    if VERBOSE:
        logger.debug(f"Embedding Preprocessing: {timings['embedding_preprocessing']:.2f}s")
        logger.debug(f"Embedding Encoding: {timings['embedding_encoding']:.2f}s")
        logger.debug(f"Average Embedding Norm: {np.mean(embedding_norms):.4f} ± {np.std(embedding_norms):.4f}")
    
    # Database operations
    evaluator.start_monitoring()
    
    start_time = time.time()
    collection = get_db_connection()
    timings["db_connection"] = time.time() - start_time
    logger.warning(f"Database Connection Duration: {timings['db_connection']:.2f}s")
    
    start_time = time.time()
    collection.delete_many({})
    timings["db_clearing"] = time.time() - start_time
    logger.warning(f"Database Clearing Duration: {timings['db_clearing']:.2f}s")
    
    start_time = time.time()
    logger.warning("Inserting documents sequentially into MongoDB")
    for doc in docs:
        collection.insert_one(doc)
    timings["sequential_storage"] = time.time() - start_time
    logger.warning(f"Sequential Document Storage Duration: {timings['sequential_storage']:.2f}s")
    
    logger.warning("Setting up vector index")
    start_time = time.time()
    timings["document_indexing"] = setup_vector_index(collection, embedding_size)
    logger.warning(f"Document Indexing Duration: {timings['document_indexing']:.2f}s")
    
    evaluator.log_resources("After Index Setup")
    total_docs = collection.count_documents({})
    stats["docs_stored_sequential"] = len(docs)
    stats["db_entries"] = total_docs
    
    logger.warning(f"Stored {stats['docs_stored_sequential']} documents sequentially, {stats['db_entries']} entries in MongoDB.")
    
    return collection, hotpotqa_dataset, timings, stats