from config.settings import settings
from data.dataset_loader import DatasetLoader
from data.preprocessor import TextPreprocessor
from models.embedding_model import EmbeddingModel
from database.qdrant_connection import QdrantConnection
from database.collection_manager import CollectionManager
from services.search_service import SearchService
from utils.logger import get_logger, log_performance
from utils.train_options import TrainOptions
from qdrant_client.http import models

# Initialize logger for main module
logger = get_logger("main")

@log_performance("Main Qdrant Project Execution")
def main():
    
    logger.info("Starting Qdrant Vector Database Project")
    
    try:
        # 1. Load and preprocess dataset
        logger.info("Loading and preprocessing dataset...")
        dataset_loader = DatasetLoader(dataset_name=settings.data_processing.dataset_name, split=settings.data_processing.dataset_split)
        df = dataset_loader.load_dataset()
        
        preprocessor = TextPreprocessor(stopwords=settings.stopwords.stopwords)
        preprocessor.preprocess_dataframe_column(df, 'text')
        
        # Save dataset to parquet
        try:
            dataset_loader.save_to_parquet(file_path=settings.data_processing.parquet_file_path)
            logger.info(f"Dataset preprocessed and saved ({len(df)} records)")
        except Exception as e:
            logger.error(f"Failed to save dataset to parquet: {e}")
            return
        
        # 2. Initialize embedding model
        logger.info("Initializing embedding model...")
        embedding_model = EmbeddingModel(model_name=settings.model.embedding_model_name)
        vector_size = settings.model.vector_size
        logger.info(f"Model loaded with vector size: {vector_size}")
        
        # 3. Connect to Qdrant
        logger.info("Connecting to Qdrant...")
        qdrant_connection = QdrantConnection(host=settings.qdrant.host, port=settings.qdrant.port, prefer_grpc=settings.qdrant.prefer_grpc)
        client = qdrant_connection.connect()
        
        if not qdrant_connection.test_connection():
            logger.error("Failed to connect to Qdrant. Please ensure Qdrant is running.")
            return
        
        logger.info("Connected to Qdrant")
        
        # 4. Initialize services
        collection_manager = CollectionManager(client)
        search_service = SearchService(client, embedding_model)
        logger.info("All components initialized")
        
        # 5. Process BBC News collection
        logger.info("Processing BBC News collection...")
        collection_manager.create_collection(settings.collection.bbc_news_collection_name, vector_size)
        
        # Create TrainOptions for the new training structure
        train_options = TrainOptions(
            data_path=settings.data_processing.parquet_file_path,
            columns_for_embed="text",
            payloads_columns=["text", "label", "label_text"],
            shard_keys_columns="label"
        )
        
        collection_manager.train(train_options)
        logger.info("BBC News collection processed")
        
        # 6. Perform search operations
        logger.info("Performing search operations...")
        query_text = "UK economy shows signs of recovery as inflation falls "
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="label_text",
                    match=models.MatchValue(value="business")
                )
            ]
        )
        
        results = search_service.search_similar_texts(
            query_text, 
            settings.collection.bbc_news_collection_name,
            limit=5,
            score_threshold=0.5,
            use_named_vector=True,
            query_filter=query_filter 
        )
        search_service.print_search_results(results, "BBC News Search Results")
        
        # 7. Perform recommendation operations
        logger.info("Performing recommendation operations...")
        results = search_service.recommend_similar_texts(
            positive_ids=[1,3,9],
            negative_ids=[5],
            collection_name=settings.collection.bbc_news_collection_name,
            limit=3,
            score_threshold=0.25,
            use_named_vector=True,
            query_filter=query_filter 
        )
        search_service.print_search_results(results, "Basic Recommendation Results")
        
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.exception(f"Critical error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

