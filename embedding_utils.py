import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from config import MODEL_CONFIGS, VERBOSE
import logging
import time

class EmbeddingGenerator(nn.Module):
    """Generate embeddings for text using specified model."""
    
    def __init__(self, model_name, embedding_size, quantize=False):
        super().__init__()
        logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.device = "cuda"
        self.base_model = MODEL_CONFIGS[model_name].get("base_model", model_name)
        
        if "Llama-3.1" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                cache_dir="/home/ubuntu/rag_project/llama-3.1-8b",
                token=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            ) if quantize else None
            
            self.model = AutoModel.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                cache_dir="/home/ubuntu/rag_project/llama-3.1-8b",
                torch_dtype=torch.float16,
                device_map="auto",
                token=True
            )
            self.model.eval()
            hidden_size = self.model.config.hidden_size
            self.projection = nn.Linear(hidden_size, embedding_size, dtype=torch.float16).to(self.device)
            self.projection.eval()
        else:
            self.model = SentenceTransformer(self.base_model).to(self.device)
            self.model.eval()
            hidden_size = self.model.get_sentence_embedding_dimension()
            if "mxbai-embed-large-v1" in model_name:
                self.output_size = embedding_size
            elif "gte-base-384" in model_name and embedding_size == 384:
                self.projection = nn.Linear(hidden_size, embedding_size, dtype=torch.float16).to(self.device)
                self.projection.eval()
    
    def generate_embedding(self, texts):
        """Generate embeddings for a list of texts with timing."""
        logger = logging.getLogger(__name__)
        timings = {"query_preprocessing": 0.0, "query_encoding": 0.0}

        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]

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
            embedding = self.model.encode(texts, convert_to_tensor=True, batch_size=1)  # Sequential processing
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