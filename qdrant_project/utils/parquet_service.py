"""
Parquet service for reading parquet files in batches.
"""

import pandas as pd
from typing import Generator
from config.settings import settings
from tqdm import tqdm


class ParquetService:
    """Service for reading parquet files in batches."""
    
    @staticmethod
    def read_parquet_file_batch(file_path: str, batch_size: int = None) -> Generator[pd.DataFrame, None, None]:
        """
        Read parquet file in batches.
        
        Args:
            file_path (str): Path to the parquet file.
            batch_size (int): Size of each batch.
            
        Yields:
            pd.DataFrame: DataFrame batch.
        """
        batch_size = batch_size or settings.data_processing.batch_size
        try:
            # Read parquet file in chunks
            parquet_file = pd.read_parquet(file_path, engine='pyarrow')
            
            # Calculate total batches
            total_rows = len(parquet_file)
            total_batches = (total_rows + batch_size - 1) // batch_size
            
            # Use tqdm for progress tracking
            for i in tqdm(range(total_batches), desc="Processing batches", unit="batch"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_rows)
                
                batch = parquet_file.iloc[start_idx:end_idx].copy()
                yield batch
                
        except Exception as e:
            raise Exception(f"Failed to read parquet file {file_path}: {e}")
    
    @staticmethod
    def read_parquet_file(file_path: str) -> pd.DataFrame:
        """
        Read entire parquet file.
        
        Args:
            file_path (str): Path to save the parquet file.
            
        Returns:
            pd.DataFrame: Complete DataFrame.
        """
        try:
            return pd.read_parquet(file_path, engine='pyarrow')
        except Exception as e:
            raise Exception(f"Failed to read parquet file {file_path}: {e}")
