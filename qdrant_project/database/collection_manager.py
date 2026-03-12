"""
Qdrant collection management functionality.
"""

from typing import List, Dict, Any, Optional, Generator, Tuple, Union
from qdrant_client.http import models
from qdrant_client import QdrantClient
from config.settings import settings
from utils.logger import get_logger
from utils.parquet_service import ParquetService
from utils.train_options import TrainOptions
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field, field_validator

# Initialize logger for this module
logger = get_logger("collection_manager")


class CollectionParameters(BaseModel):
    """Validated collection creation parameters."""
    
    collection_name: str = Field(..., min_length=1, description="Name of the collection")
    vector_size: int = Field(default=settings.model.vector_size, gt=0, le=10000, description="Vector dimension size")
    
    @field_validator('collection_name')
    @classmethod
    def validate_collection_name(cls, v):
        if not v.strip():
            raise ValueError("collection_name cannot be empty or whitespace only")
        # Qdrant collection name restrictions
        if any(char in v for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            raise ValueError("collection_name contains invalid characters")
        return v.strip()


class TrainingParameters(BaseModel):
    """Validated training parameters."""
    
    batch_size: int = Field(default=128, gt=0, le=1000, description="Training batch size")
    embedding_model_name: str = Field(default=settings.model.embedding_model_name, description="Embedding model name")
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be positive")
        if v > 1000:
            raise ValueError("batch_size too large, may cause memory issues")
        return v


class UpsertParameters(BaseModel):
    """Validated upsert parameters."""
    
    payloads: List[Dict[str, Any]] = Field(..., min_items=1, description="Document payloads")
    ids: List[Union[int, str]] = Field(..., min_items=1, description="Document IDs")
    vectors: Dict[str, List[List[float]]] = Field(..., description="Document vectors")
    collection_name: str = Field(..., min_length=1, description="Collection name")
    
    @field_validator('payloads', 'ids')
    @classmethod
    def validate_lists_have_same_length(cls, v, info):
        # Pydantic V2: info.data ile erişim
        if 'payloads' in info.data and 'ids' in info.data:
            if len(info.data['payloads']) != len(info.data['ids']):
                raise ValueError("payloads and ids must have the same length")
        return v
    
    @field_validator('vectors')
    @classmethod
    def validate_vectors_structure(cls, v):
        if not v:
            raise ValueError("vectors cannot be empty")
        for vector_name, vector_list in v.items():
            if not isinstance(vector_list, list):
                raise ValueError(f"vector {vector_name} must be a list")
        return v


class CollectionManager:
    """Handles Qdrant collection operations including vector operations."""
    
    def __init__(self, client: QdrantClient):
        """
        Initialize collection manager.
        
        Args:
            client (QdrantClient): Qdrant client instance.
        """
        self.client = client
        self.id_counter = 0
        logger.info("CollectionManager initialized")
    
    def create_collection(self, collection_name: str, vector_size: int = settings.model.vector_size) -> None:
        """
        Create a collection with proper checks and cleanup.
        
        Args:
            collection_name (str): Name of the collection to create.
            vector_size (int): Size of the vectors.
        """
        # Validate parameters
        collection_params = CollectionParameters(
            collection_name=collection_name,
            vector_size=vector_size
        )
        
        logger.info(f"Creating collection: {collection_params.collection_name}")
        
        # Check if collection exists and delete if it does
        try:
            if self.client.collection_exists(collection_params.collection_name):
                logger.info(f"Collection '{collection_params.collection_name}' exists, deleting it first")
                self.client.delete_collection(collection_name=collection_params.collection_name)
                logger.info(f"Deleted existing collection '{collection_params.collection_name}'")
            
            # Create new collection with named vector
            vector_field_name = self.get_vector_field_name()
            self.client.create_collection(
                collection_name=collection_params.collection_name,
                vectors_config={
                    vector_field_name: models.VectorParams(
                        size=collection_params.vector_size,
                        distance=models.Distance.COSINE
                    )
                }
            )
            logger.info(f"Successfully created collection '{collection_params.collection_name}' with named vector '{vector_field_name}' and size {collection_params.vector_size}")
            print(f"✓ Created collection '{collection_params.collection_name}' with named vector '{vector_field_name}' and size {collection_params.vector_size}")
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_params.collection_name}': {e}")
            raise
    
    def train(self, train_options: "TrainOptions") -> "CollectionManager":
        """
        Train the collection using TrainOptions and ParquetService.
        
        Args:
            train_options (TrainOptions): Training options containing data path and column configurations.
            
        Returns:
            CollectionManager: Self for method chaining.
        """
        logger.info(f"Starting training with TrainOptions: {train_options.data_path}")
        
        # Reset ID counter for new training session
        self.id_counter = 0
        
        # Initialize SentenceTransformer model once (cache it)
        logger.info("Initializing SentenceTransformer model...")
        embedding_model = SentenceTransformer(settings.model.embedding_model_name)
        logger.info("SentenceTransformer model initialized and cached")
        
        try:
            for batch in ParquetService.read_parquet_file_batch(train_options.data_path):
                batch_size = len(batch)
                
                # Handle ID generation - use custom ID column if provided, otherwise generate sequential IDs
                if train_options.id_column_name and len(train_options.id_column_name) > 0:
                    # Use custom ID column(s)
                    if len(train_options.id_column_name) == 1:
                        # Single ID column
                        ids = batch[train_options.id_column_name[0]].astype(str).to_list()
                    else:
                        # Multiple ID columns - combine them
                        ids = batch[train_options.id_column_name].apply(
                            lambda x: '_'.join(x.astype(str)), axis=1
                        ).to_list()
                else:
                    # Generate sequential IDs for this batch
                    ids = self.get_next_ids(batch_size)
                
                # Handle columns_for_embed - can be single column or list
                if isinstance(train_options.columns_for_embed, list):
                    if len(train_options.columns_for_embed) == 1:
                        # Single column
                        docs_for_embed = batch[train_options.columns_for_embed[0]].to_list()
                    else:
                        # Multiple columns - combine them
                        docs_for_embed = batch[train_options.columns_for_embed].apply(
                            lambda x: ' '.join(x.astype(str)), axis=1
                        ).to_list()
                else:
                    # Single column name as string
                    docs_for_embed = batch[train_options.columns_for_embed].to_list()
                
                # Handle payloads columns
                payloads_columns = train_options.payloads_columns or []
                payloads = []
                
                for index, row in batch.iterrows():
                    payload = {}
                    for column in payloads_columns:
                        value = row[column]
                        # Convert numpy types to Python native types for Qdrant compatibility
                        if hasattr(value, 'item'):
                            payload[column] = value.item()  # Convert numpy.int64, numpy.float64, etc.
                        else:
                            payload[column] = value
                    payloads.append(payload)
                
                # Add shard keys to payloads if specified
                if train_options.shard_keys_columns and len(train_options.shard_keys_columns) > 0:
                    for i, payload in enumerate(payloads):
                        if len(train_options.shard_keys_columns) == 1:
                            # Single shard key column
                            shard_value = batch.iloc[i][train_options.shard_keys_columns[0]]
                            if hasattr(shard_value, 'item'):
                                payload['_shard_key'] = shard_value.item()
                            else:
                                payload['_shard_key'] = shard_value
                        else:
                            # Multiple shard key columns - combine them
                            shard_keys = []
                            for col in train_options.shard_keys_columns:
                                value = batch.iloc[i][col]
                                if hasattr(value, 'item'):
                                    shard_keys.append(str(value.item()))
                                else:
                                    shard_keys.append(str(value))
                            payload['_shard_key'] = '_'.join(shard_keys)
                
                # Use cached model instead of creating new one
                vectors = {
                    self.get_vector_field_name(): [
                        arr.tolist()
                        for arr in embedding_model.encode(
                            sentences=docs_for_embed,
                            batch_size=256, 
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                    ]
                }
                
                doc_len = len(docs_for_embed)
                self.upsert_collection(payloads, ids, vectors, settings.collection.bbc_news_collection_name, doc_len)
                
                logger.info(f"Processed batch with {doc_len} documents")
            
            logger.info("Training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def upsert_collection(self, payloads, ids, vectors, collection_name: str = settings.collection.bbc_news_collection_name, doc_len: int = None) -> None:
        """
        Upsert documents to an existing collection.
        
        Args:
            payloads: Document payloads to upsert.
            ids: Document IDs.
            vectors: Document vectors in named format: {"vector": [vector_list]}.
            collection_name (str): Name of the collection. Defaults to BBC_NEWS_COLLECTION_NAME.
            doc_len (int): Number of documents being upserted.
        """
        # Validate parameters
        upsert_params = UpsertParameters(
            payloads=payloads,
            ids=ids,
            vectors=vectors,
            collection_name=collection_name
        )
        
        if self.client.collection_exists(upsert_params.collection_name):
            try:
                # vectors should already be in the correct named format from train function
                # Format: {"vector": [vector_list]}
                self.client.upsert(
                    collection_name=upsert_params.collection_name,
                    points=models.Batch(ids=upsert_params.ids, vectors=upsert_params.vectors, payloads=upsert_params.payloads),
                )
                logger.info(
                    f"Uploaded collection '{upsert_params.collection_name}' with {doc_len} documents."
                )
            except Exception as e:
                logger.error(f"Failed to upsert documents: {e}")
                raise
        else:
            logger.error(f"Collection '{upsert_params.collection_name}' does not exist.")
    
    def get_next_ids(self, count: int) -> List[int]:
        """
        Get the next batch of IDs.
        
        Args:
            count (int): Number of IDs needed.
            
        Returns:
            List[int]: List of IDs.
        """
        ids = [self.id_counter + i for i in range(count)]
        self.id_counter += count
        logger.debug(f"Generated {count} IDs: {ids[0]}-{ids[-1]}")
        return ids
    
    def reset_id_counter(self) -> None:
        """Reset the ID counter to 0."""
        old_counter = self.id_counter
        self.id_counter = 0
        logger.info(f"ID counter reset from {old_counter} to 0")

    def get_vector_field_name(self) -> str:
        """
        Get the vector field name for the collection.
        
        Returns:
            str: Vector field name.
        """
        return "vector"
