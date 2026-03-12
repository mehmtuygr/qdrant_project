"""
Training options configuration for collection training.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union


class TrainOptions(BaseModel):
    """Training options for collection training."""
    
    data_path: str = Field(..., description="Path to the parquet file containing training data.")
    
    id_column_name: Optional[Union[str, List[str]]] = Field(
        default=None, 
        description="Column name(s) to use as document IDs. Can be a single column name or list of column names."
    )
    
    columns_for_embed: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Column(s) to use for generating embeddings. Can be a single column name or list of column names."
    )
    
    payloads_columns: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Column(s) to include in the payload. Can be a single column name or list of column names."
    )
    
    shard_keys_columns: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Column(s) to use as shard keys for distributed collections. Can be a single column name or list of column names."
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
    
    def model_post_init(self, __context) -> None:
        """Validate and normalize the training options after initialization."""
        if not self.data_path:
            raise ValueError("data_path cannot be empty")
        
        # Convert single column names to lists for consistency
        if isinstance(self.id_column_name, str):
            self.id_column_name = [self.id_column_name]
        
        if isinstance(self.columns_for_embed, str):
            self.columns_for_embed = [self.columns_for_embed]
        
        if isinstance(self.payloads_columns, str):
            self.payloads_columns = [self.payloads_columns]
        
        if isinstance(self.shard_keys_columns, str):
            self.shard_keys_columns = [self.shard_keys_columns]
        
        # Validate that required fields are provided
        if not self.columns_for_embed:
            raise ValueError("columns_for_embed cannot be empty")
        
        if not self.payloads_columns:
            raise ValueError("payloads_columns cannot be empty")
    
    def get_id_column(self) -> Optional[str]:
        """Get the primary ID column name."""
        if self.id_column_name and len(self.id_column_name) > 0:
            return self.id_column_name[0]
        return None
    
    def get_shard_key_column(self) -> Optional[str]:
        """Get the primary shard key column name."""
        if self.shard_keys_columns and len(self.shard_keys_columns) > 0:
            return self.shard_keys_columns[0]
        return None
