"""
Logging configuration and utilities using Loguru.
"""

import sys
from pathlib import Path
from loguru import logger
from config.settings import settings


class LoggerConfig:
    """Configures and manages logging for the Qdrant project."""
    
    def __init__(self):
        """Initialize logger configuration."""
        self.logger = logger
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Remove default logger
        logger.remove()
        
        # Create logs directory if it doesn't exist
        log_dir = Path(settings.logging.log_file).parent
        log_dir.mkdir(exist_ok=True)
        
        # Console logging
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.logging.log_level,
            colorize=True
        )
        
        # File logging
        logger.add(
            settings.logging.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.logging.log_level,
            rotation="5 MB",
            retention="7 days"
        )
    
    def get_logger(self, name: str = None):
        """
        Get a logger instance with the specified name.
        
        Args:
            name (str): Name for the logger (usually module name).
            
        Returns:
            Logger: Configured logger instance.
        """
        if name:
            return logger.bind(name=name)
        return logger
    
    def log_performance(self, operation_name: str):
        """
        Decorator to log operation performance.
        
        Args:
            operation_name (str): Name of the operation being performed.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                import time
                start_time = time.time()
                
                logger.info(f"Starting {operation_name}")
                
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    logger.info(f"Completed {operation_name} in {duration:.2f} seconds")
                    return result
                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    logger.error(f"Failed {operation_name} after {duration:.2f} seconds: {str(e)}")
                    raise
            return wrapper
        return decorator


# Global logger instance
logger_config = LoggerConfig()


def get_logger(name: str = None):
    """
    Get a logger instance.
    
    Args:
        name (str): Name for the logger.
        
    Returns:
        Logger: Configured logger instance.
    """
    return logger_config.get_logger(name)


def log_performance(operation_name: str):
    """
    Decorator to log operation performance.
    
    Args:
        operation_name (str): Name of the operation.
    """
    return logger_config.log_performance(operation_name)
