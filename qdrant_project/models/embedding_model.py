"""
Embedding model management for text vectorization.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
from config.settings import settings
from utils.logger import get_logger, log_performance
from pydantic import BaseModel, Field, field_validator

# Initialize logger for this module
logger = get_logger("embedding_model")


class ModelParameters(BaseModel):
    """Validated embedding model parameters."""
    
    model_name: str = Field(default=settings.model.embedding_model_name, min_length=1, description="HuggingFace model name")
    device: Optional[str] = Field(default=None, description="Device to use (cpu, cuda, etc.)")
    cache_folder: Optional[str] = Field(default=None, description="Cache folder for models")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        if not v.strip():
            raise ValueError("model_name cannot be empty or whitespace only")
        # Basic HuggingFace model name validation
        if not v.startswith(('sentence-transformers/', 'all-MiniLM', 'all-mpnet')):
            logger.warning(f"Model name '{v}' may not be a standard sentence transformer model")
        return v.strip()


class EncodingParameters(BaseModel):
    """Validated text encoding parameters."""
    
    texts: List[str] = Field(..., min_items=1, description="List of texts to encode")
    batch_size: int = Field(default=32, gt=0, le=512, description="Batch size for encoding")
    normalize_embeddings: bool = Field(default=True, description="Normalize embeddings to unit length")
    convert_to_numpy: bool = Field(default=True, description="Convert output to numpy arrays")
    show_progress_bar: bool = Field(default=False, description="Show progress bar during encoding")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if not all(isinstance(text, str) and text.strip() for text in v):
            raise ValueError("All texts must be non-empty strings")
        return [text.strip() for text in v]
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be positive")
        if v > 512:
            logger.warning(f"Large batch_size {v} may cause memory issues")
        return v


class EmbeddingModel:
    """Manages sentence transformer embedding model for text vectorization."""
    
    def __init__(self, model_name: str = None, device: Optional[str] = None):
        """
        Initialize embedding model.
        
        Args:
            model_name (str): Name of the HuggingFace model to use.
            device (Optional[str]): Device to use for model inference.
        """
        # Validate parameters
        model_params = ModelParameters(
            model_name=model_name,
            device=device
        )
        
        self.model_name = model_params.model_name
        self.device = model_params.device
        
        logger.info(f"Initializing embedding model: {self.model_name}")
        
        try:
            # Initialize SentenceTransformer model
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=model_params.cache_folder
            )
            
            # Get model info
            self.vector_size = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Successfully loaded model: {self.model_name}")
            logger.info(f"Model vector size: {self.vector_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {e}")
            raise
    
    @log_performance("Text Encoding")
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False
    ) -> Union[List[np.ndarray], np.ndarray]:
        """
        Encode a list of texts to embeddings.
        
        Args:
            texts (List[str]): List of texts to encode.
            batch_size (int): Batch size for processing.
            normalize_embeddings (bool): Whether to normalize embeddings.
            convert_to_numpy (bool): Whether to convert output to numpy arrays.
            show_progress_bar (bool): Whether to show progress bar.
            
        Returns:
            Union[List[np.ndarray], np.ndarray]: Encoded embeddings.
        """
        # Validate parameters
        encoding_params = EncodingParameters(
            texts=texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar
        )
        
        logger.info(f"Encoding {len(encoding_params.texts)} texts to embeddings")
        
        try:
            # Encode texts using the model
            embeddings = self.model.encode(
                sentences=encoding_params.texts,
                batch_size=encoding_params.batch_size,
                normalize_embeddings=encoding_params.normalize_embeddings,
                convert_to_numpy=encoding_params.convert_to_numpy,
                show_progress_bar=encoding_params.show_progress_bar
            )
            
            logger.info(f"Successfully encoded {len(encoding_params.texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise
    
    def get_model_name(self) -> str:
        """
        Get the name of the loaded model.
        
        Returns:
            str: Model name.
        """
        return self.model_name
    
    def get_vector_size(self) -> int:
        """
        Get the vector dimension size of the model.
        
        Returns:
            int: Vector dimension size.
        """
        return self.vector_size
    
    def get_model_info(self) -> dict:
        """
        Get comprehensive information about the loaded model.
        
        Returns:
            dict: Model information including name, vector size, and device.
        """
        return {
            "model_name": self.model_name,
            "vector_size": self.vector_size,
            "device": str(self.model.device),
            "max_seq_length": self.model.max_seq_length
        }
