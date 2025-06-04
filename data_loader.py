from datasets import load_dataset, disable_progress_bars
from config import DATASET_NAME, SUBSET_NAME, CORPUS_NAME, CORPUS_LIMIT, VERBOSE, BATCH_SIZE
from db_utils import get_db_connection, setup_vector_index
from system_evaluation import SystemEvaluator
from tqdm import tqdm
import torch
import numpy as np
import logging
import time
import os

# Suppress progress bars for dataset loading
try:
    disable_progress_bars()
except AttributeError:
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
logger = logging.getLogger(__name__)

def compute_doc_stats(dataset, is_corpus=False):
    """Compute document count and size in MB for a dataset or corpus."""
    logger.info(f"Computing stats for {'corpus' if is_corpus else 'dataset'} with {len(dataset)} entries")
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
            outputs = entry.get("output", [])
            query_text = entry["input"]
            doc_count += sum(len(output.get("provenance", [])) for output in outputs)
            total_bytes += len(query_text.encode("utf-8"))
            for output in outputs:
                for prov in output.get("provenance", []):
                    total_bytes += len(prov.get("title", "").encode("utf-8"))
    logger.debug(f"Computed stats: doc_count={doc_count}, size_mb={total_bytes / (1024 * 1024):.2f}")
    return doc_count, total_bytes / (1024 * 1024)

def load_and_store_data(limit=None, embedding_generator=None, embedding_size=None):
    """Load KILT corpus and HotpotQA queries, store in MongoDB with embeddings."""
    logger.info(f"Starting data load and storage with limit={limit}, embedding_size={embedding_size}")
    if embedding_generator is None or embedding_size is None:
        logger.error("Embedding generator and size are required.")
        raise ValueError("Embedding generator and size are required.")
    
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
    
    # Load KILT corpus
    start_time = time.time()
    split = "train" if CORPUS_LIMIT is None else f"train[:{CORPUS_LIMIT}]"
    logger.info(f"Loading KILT corpus: {CORPUS_NAME} with split: {split}")
    try:
        kilt_corpus = load_dataset(CORPUS_NAME, split=split)
    except Exception as e:
        logger.error(f"Failed to load KILT corpus: {str(e)}")
        raise
    timings["dataset_load"] = time.time() - start_time
    # logger.debug(f"Dataset Loading Duration: {timings['dataset_load']:.2f}s, corpus size={len(kilt_corpus)}")
    
    # Load KILT HotpotQA queries
    logger.info(f"Loading KILT queries: {DATASET_NAME} with subset: {SUBSET_NAME}")
    try:
        hotpotqa_dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split="train")
    except Exception as e:
        logger.error(f"Failed to load HotpotQA dataset: {str(e)}")
        raise
    if limit:
        hotpotqa_dataset = hotpotqa_dataset.select(range(min(limit, len(hotpotqa_dataset))))
    logger.debug(f"HotpotQA dataset loaded, size={len(hotpotqa_dataset)}")
    
    if VERBOSE:
        logger.debug(f"Sample KILT Corpus Entry: {kilt_corpus[0]}")
        logger.debug(f"Sample HotpotQA Entry: {hotpotqa_dataset[0]}")
    
    evaluator = SystemEvaluator()
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
    stats["total_size_mb"] = corpus_mb + query_mb
    stats["embedding_mode"] = "batch"
    
    logger.info(f"KILT Corpus: {stats['corpus_entries']} entries, {corpus_doc_count} docs, {corpus_mb:.2f} MB")
    logger.info(f"HotpotQA Queries: {stats['hotpotqa_entries']} entries, {query_doc_count} docs, {query_mb:.2f} MB")
    logger.info(f"Total Documents: {stats['total_docs']} docs, {stats['total_size_mb']:.2f} MB")
    
    if stats["hotpotqa_entries"] < 100:
        logger.warning(f"Query dataset size is small ({stats['hotpotqa_entries']} entries). Consider increasing 'limit'.")
    
    # Collect all documents from KILT corpus
    start_time = time.time()
    all_docs = []
    for entry in kilt_corpus:
        if "contents" not in entry or "wikipedia_id" not in entry or "title" not in entry:
            logger.error(f"Missing required fields in corpus entry: {entry.keys()}")
            raise KeyError("Expected 'contents', 'wikipedia_id', and 'title' in KILT corpus entry")
        all_docs.append({
            "text": entry["contents"],
            "wikipedia_id": entry["wikipedia_id"],
            "wikipedia_title": entry["title"],
            "source": "kilt_corpus"
        })
    timings["doc_collection"] = time.time() - start_time
    # logger.debug(f"Document Collection Duration: {timings['doc_collection']:.2f}s, docs collected={len(all_docs)}")
    
    # Batch embedding generation
    start_time = time.time()
    docs = []
    embedding_norms = []
    
    logger.info("Using batch embedding mode")
    for i in tqdm(range(0, len(all_docs), BATCH_SIZE), desc="Generating batch embeddings", disable=not VERBOSE):
        batch = all_docs[i:i + BATCH_SIZE]
        batch_texts = [item["text"] for item in batch]
        # logger.debug(f"Embedding batch {i//BATCH_SIZE + 1}, size={len(batch_texts)}")
        try:
            embeddings, seq_timings = embedding_generator.generate_embedding(batch_texts)
            for j, embedding in enumerate(embeddings):
                norm = np.linalg.norm(embedding)
                embedding_norms.append(norm)
                if norm < 1e-6:
                    logger.warning(f"Zero or near-zero embedding norm for text: {batch[j]['text'][:50]}...")
                docs.append({
                    "text": batch[j]["text"],
                    "embedding": embedding,
                    "wikipedia_id": batch[j]["wikipedia_id"],
                    "wikipedia_title": batch[j]["wikipedia_title"],
                    "source": batch[j]["source"]
                })
            timings["embedding_preprocessing"] += seq_timings["query_preprocessing"]
            timings["embedding_encoding"] += seq_timings["query_encoding"]
        except Exception as e:
            logger.error(f"Error embedding batch {i//BATCH_SIZE + 1}: {str(e)}")
            continue
    timings["document_embedding"] = time.time() - start_time
   # logger.debug(f"Document Embedding Duration: {timings['document_embedding']:.2f}s, docs embedded={len(docs)}")
    
    # if VERBOSE:
    #     logger.debug(f"Embedding Preprocessing: {timings['embedding_preprocessing']:.2f}s")
    #     logger.debug(f"Embedding Encoding: {timings['embedding_encoding']:.2f}s")
    #     logger.debug(f"Average Embedding Norm: {np.mean(embedding_norms):.4f} ± {np.std(embedding_norms):.4f}")
    
    # Database operations
    start_time = time.time()
    collection = get_db_connection()
    timings["db_connection"] = time.time() - start_time
    # logger.debug(f"Database Connection Duration: {timings['db_connection']:.2f}s")
    
    start_time = time.time()
    collection.delete_many({})
    timings["db_clearing"] = time.time() - start_time
   #  logger.debug(f"Database Clearing Duration: {timings['db_clearing']:.2f}s")
    
    start_time = time.time()
    logger.info("Inserting documents in batches into MongoDB")
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
      #   logger.debug(f"Inserting batch {i//BATCH_SIZE + 1}, size={len(batch)}")
        try:
            collection.insert_many(batch)
        except Exception as e:
            logger.error(f"Error inserting batch {i//BATCH_SIZE + 1}: {str(e)}")
            continue
    timings["sequential_storage"] = time.time() - start_time
    # logger.debug(f"Batch Document Storage Duration: {timings['sequential_storage']:.2f}s")
    
    logger.info("Setting up vector index")
    start_time = time.time()
    timings["document_indexing"] = setup_vector_index(collection, embedding_size)
    # logger.debug(f"Document Indexing Duration: {timings['document_indexing']:.2f}s")
    
    evaluator.log_resources("After Index Setup")
    total_docs = collection.count_documents({})
    stats["docs_stored_batch"] = len(docs)
    stats["db_entries"] = total_docs
    # logger.debug(f"Stored {stats['docs_stored_batch']} documents, db_entries={stats['db_entries']}")
    
    logger.info(f"Stored {stats['docs_stored_batch']} documents in batches, {stats['db_entries']} entries in MongoDB.")
    return collection, hotpotqa_dataset, timings, stats