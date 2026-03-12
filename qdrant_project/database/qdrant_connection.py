"""
Qdrant client connection management.
"""

from qdrant_client import QdrantClient
from config.settings import settings
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger("qdrant_connection")


class QdrantConnection:
    """Handles Qdrant database connection."""
    
    def __init__(self, host: str = None, 
                 port: int = None, 
                 prefer_grpc: bool = None):
        """
        Initialize Qdrant connection.
        
        Args:
            host (str): Qdrant host address.
            port (int): Qdrant port number.
            prefer_grpc (bool): Whether to prefer gRPC over HTTP.
        """
        self.host = host or settings.qdrant.host
        self.port = port or settings.qdrant.port
        self.prefer_grpc = prefer_grpc if prefer_grpc is not None else settings.qdrant.prefer_grpc
        self.client = None
        logger.info(f"QdrantConnection initialized for {self.host}:{self.port} (gRPC: {self.prefer_grpc})")
    
    def connect(self) -> QdrantClient:
        """
        Establish connection to Qdrant.
        
        Returns:
            QdrantClient: Connected Qdrant client.
        """
        logger.info(f"Establishing connection to Qdrant at {self.host}:{self.port}")
        
        try:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                prefer_grpc=self.prefer_grpc
            )
            logger.info("Successfully connected to Qdrant")
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test the connection to Qdrant.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        logger.info("Testing Qdrant connection")
        
        try:
            if self.client is None:
                self.connect()
            collections = self.client.get_collections()
            logger.info("Connection test successful")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
