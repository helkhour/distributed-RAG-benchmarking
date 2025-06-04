import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from config import MODEL_CONFIGS, VERBOSE, EMBED_BATCH_SIZE
import logging
import time

logger = logging.getLogger(__name__)

class EmbeddingGenerator(nn.Module):
    """Generate embeddings for text using specified model."""
    
    def __init__(self, model_name, embedding_size, quantize=False):
        super().__init__()
        logger.info(f"Initializing EmbeddingGenerator with model={model_name}, embedding_size={embedding_size}, quantize={quantize}")
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = MODEL_CONFIGS[model_name].get("base_model", model_name)
        self.normalize_embeddings = MODEL_CONFIGS[model_name].get(
            "normalize_embeddings", False
        )
        
        self.model = SentenceTransformer(self.base_model, device=self.device)
        self.model.eval()
        hidden_size = self.model.get_sentence_embedding_dimension()
        # logger.debug(f"Model loaded, hidden_size={hidden_size}")
        
        if hidden_size != embedding_size:
            self.projection = nn.Linear(hidden_size, embedding_size, dtype=torch.float16).to(self.device)
            self.projection.eval()
            # logger.debug(f"Projection layer added: {hidden_size} -> {embedding_size}")
        else:
            self.projection = None
            # logger.debug("No projection layer needed, sizes match")
    
    def generate_embedding(self, texts):
        """Generate embeddings for a list of texts with timing."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        # logger.debug(f"Sample text: {texts[0][:50]}..." if texts else "No texts provided")
        timings = {"query_preprocessing": 0.0, "query_encoding": 0.0}

        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]
            # logger.debug("Converted single string to list")

        if "Llama-3.1" in self.model_name:
            try:
                # Time preprocessing (tokenization)
                start_time = time.time()
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                preprocess_duration = time.time() - start_time
                timings["query_preprocessing"] = preprocess_duration
                logger.debug(f"Query Preprocessing Duration: {preprocess_duration:.4f}s")

                # Time encoding
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
                num_tokens = torch.sum(attention_mask, dim=1)
                embeddings = sum_embeddings / num_tokens
                embeddings = self.projection(embeddings)
                encoding_duration = time.time() - start_time
                timings["query_encoding"] = encoding_duration
                logger.debug(f"Query Encoding Duration: {encoding_duration:.4f}s")

                embeddings = embeddings.cpu().tolist()  # always return list of vectors    
                return embeddings, timings

            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                raise

        else:
            # Time encoding (includes preprocessing)
            start_time = time.time()
            embedding = self.model.encode(
                texts,
                convert_to_tensor=True,
                batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
            )
            encoding_duration = time.time() - start_time
            timings["query_encoding"] = encoding_duration
            logger.debug(f"Query Encoding Duration: {encoding_duration:.4f}s")

            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            if hasattr(self, "output_size"):
                embedding = embedding[:, :self.output_size]
            elif hasattr(self, "projection"):
                if embedding.dtype != self.projection.weight.dtype:
                    embedding = embedding.to(dtype=self.projection.weight.dtype)
                embedding = self.projection(embedding)

            embeddings = embedding.cpu().tolist()

            return embeddings, timings  # always return list of vectors
