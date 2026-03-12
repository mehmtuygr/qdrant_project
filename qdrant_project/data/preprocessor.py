"""
Text preprocessing functionality.
"""

import string
from config.settings import settings
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger("text_preprocessor")


class TextPreprocessor:
    """Handles text preprocessing operations."""
    
    def __init__(self, stopwords: set = None):
        """
        Initialize the text preprocessor.
        
        Args:
            stopwords (set): Set of stopwords to remove.
        """
        self.stopwords = stopwords or settings.stopwords.stopwords
        logger.info(f"TextPreprocessor initialized with {len(self.stopwords)} stopwords")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text by lowercasing, removing punctuation, and removing stopwords.
        
        Args:
            text (str): Input text to preprocess.
            
        Returns:
            str: Preprocessed text.
        """
        if not isinstance(text, str):
            logger.warning(f"Input text is not a string: {type(text)}")
            return str(text) if text is not None else ""
        
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        
        # Join back to string
        result = ' '.join(words)
        
        return result
    
    def preprocess_dataframe_column(self, df, column_name: str) -> None:
        """
        Preprocess a specific column in a DataFrame.
        
        Args:
            df: Pandas DataFrame.
            column_name (str): Name of the column to preprocess.
        """
        if column_name not in df.columns:
            logger.error(f"Column '{column_name}' not found in DataFrame")
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        logger.info(f"Preprocessing DataFrame column '{column_name}' with {len(df)} rows")
        
        try:
            df[column_name] = df[column_name].apply(self.preprocess_text)
            logger.info(f"Successfully preprocessed column '{column_name}'")
        except Exception as e:
            logger.error(f"Failed to preprocess column '{column_name}': {e}")
            raise
