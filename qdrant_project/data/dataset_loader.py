"""
Dataset loading and processing functionality for BBC News.
"""

import pandas as pd
from datasets import load_dataset
from config.settings import settings
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger("dataset_loader")


class DatasetLoader:
    """Handles BBC News dataset loading and processing."""
    
    def __init__(self, dataset_name: str = None, split: str = None):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name (str): Name of the dataset to load (SetFit/bbc-news).
            split (str): Dataset split to use.
        """
        self.dataset_name = dataset_name or settings.data_processing.dataset_name
        self.split = split or settings.data_processing.dataset_split
        self.df = None
        logger.info(f"DatasetLoader initialized for {self.dataset_name} with split {self.split}")
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the BBC News dataset and convert to pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded dataset as DataFrame.
        """
        logger.info(f"Loading BBC News dataset: {self.dataset_name} (split: {self.split})")
        
        # Load BBC News dataset
        dataset = load_dataset(self.dataset_name, split=self.split)
        self.df = dataset.to_pandas()
        
        # BBC News dataset has 'text' and 'label' columns
        logger.info(f"BBC News dataset size: {len(self.df)} records")
        logger.info(f"Columns: {list(self.df.columns)}")
            
        return self.df
    
    def save_to_parquet(self, file_path: str = None) -> pd.DataFrame:
        """
        Save the dataset to parquet format.
        
        Args:
            file_path (str): Path to save the parquet file.
            
        Returns:
            pd.DataFrame: The saved DataFrame for further processing.
        """
        if self.df is not None:
            file_path = file_path or settings.data_processing.parquet_file_path
            try:
                self.df.to_parquet(file_path, index=False)
                logger.info(f"Dataset saved to parquet: {file_path}")
                return self.df
            except Exception as e:
                logger.error(f"Failed to save dataset to parquet: {e}")
                raise
        else:
            logger.warning("No dataset loaded to save")
            return None
