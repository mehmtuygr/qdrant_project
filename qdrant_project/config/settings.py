"""
Configuration settings for the Qdrant project.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Set


class QdrantSettings(BaseModel):
    """Qdrant connection settings with validation."""
    
    host: str = Field(default="localhost", description="Qdrant server host")
    port: int = Field(default=6333, ge=1, le=65535, description="Qdrant server port")
    prefer_grpc: bool = Field(default=False, description="Prefer gRPC over HTTP")


class CollectionSettings(BaseModel):
    """Collection configuration settings with validation."""
    
    bbc_news_collection_name: str = Field(default="bbc_news_collection", description="BBC News collection name")


class ModelSettings(BaseModel):
    """Embedding model settings with validation."""
    
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model name")
    vector_size: int = Field(default=384, gt=0, description="Vector dimension size")


class DataProcessingSettings(BaseModel):
    """Data processing settings with validation."""
    
    batch_size: int = Field(default=128, gt=0, le=1000, description="Batch size for processing")
    dataset_name: str = Field(default="SetFit/bbc-news", description="HuggingFace dataset name")
    dataset_split: str = Field(default="train", description="Dataset split to use")
    parquet_file_path: str = Field(default="./bbc_news.parquet", description="Parquet file path")


class StopwordsSettings(BaseModel):
    """Stopwords configuration with validation."""
    
    stopwords: Set[str] = Field(
        default={
            "the", "and", "is", "in", "to", "of", "a", "for", "on", "with", 
            "as", "by", "at", "an", "be", "this", "that", "it"
        },
        description="Stopwords for text preprocessing"
    )


class LoggingSettings(BaseModel):
    """Logging configuration with validation."""
    
    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="logs/qdrant_project.log", description="Log file path")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v


class ProjectSettings(BaseModel):
    """Main project settings combining all configurations."""
    
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    collection: CollectionSettings = Field(default_factory=CollectionSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    data_processing: DataProcessingSettings = Field(default_factory=DataProcessingSettings)
    stopwords: StopwordsSettings = Field(default_factory=StopwordsSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    class Config:
        validate_assignment = True
        extra = "forbid"


# Create global settings instance
settings = ProjectSettings()