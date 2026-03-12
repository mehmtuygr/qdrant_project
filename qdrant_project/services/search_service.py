"""
Search and recommendation service for Qdrant collections.
"""

from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from models.embedding_model import EmbeddingModel
from utils.logger import get_logger, log_performance
from pydantic import BaseModel, Field, field_validator

# Initialize logger for this module
logger = get_logger("search_service")


class SearchParameters(BaseModel):
    """Validated search parameters."""
    
    query_text: str = Field(..., min_length=1, description="Text to search for")
    collection_name: str = Field(..., min_length=1, description="Collection name to search in")
    limit: int = Field(default=5, gt=0, le=100, description="Maximum number of results")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    use_named_vector: bool = Field(default=True, description="Use named vector for search")
    query_filter: Optional[Union[Dict[str, Any], models.Filter]] = Field(
        default=None,
        description="Optional Qdrant filter to apply during search"
    )
    
    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v):
        if not v.strip():
            raise ValueError("query_text cannot be empty or whitespace only")
        return v.strip()


class RecommendationParameters(BaseModel):
    """Validated recommendation parameters."""
    
    positive_ids: List[int] = Field(..., min_items=1, description="Positive example IDs")
    negative_ids: Optional[List[int]] = Field(
        default=None,
        description="Negative example IDs"
    )
    collection_name: str = Field(..., min_length=1, description="Collection name for recommendations")
    limit: int = Field(default=3, gt=0, le=100, description="Maximum number of recommendations")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    use_named_vector: bool = Field(default=True, description="Use named vector for recommendations")
    query_filter: Optional[Union[Dict[str, Any], models.Filter]] = Field(
        default=None,
        description="Optional Qdrant filter to apply during recommendation"
    )
    
    @field_validator('positive_ids')
    @classmethod
    def validate_positive_ids(cls, v):
        if not all(isinstance(id_val, int) and id_val >= 0 for id_val in v):
            raise ValueError("All positive_ids must be non-negative integers")
        return v

    @field_validator('negative_ids')
    @classmethod
    def validate_negative_ids(cls, v):
        if v is None:
            return v
        if not all(isinstance(id_val, int) and id_val >= 0 for id_val in v):
            raise ValueError("All negative_ids must be non-negative integers")
        return v


class SearchResult(BaseModel):
    """Validated search result structure."""
    
    id: int = Field(..., description="Document ID")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    payload: Dict[str, Any] = Field(..., description="Document payload")


class SearchService:
    """Service for performing similarity searches and recommendations."""
    
    def __init__(self, client: QdrantClient, embedding_model: EmbeddingModel):
        """
        Initialize search service.
        
        Args:
            client (QdrantClient): Qdrant client instance.
            embedding_model (EmbeddingModel): Embedding model instance.
        """
        self.client = client
        self.embedding_model = embedding_model
        logger.info("SearchService initialized")
    
    @log_performance("Similarity Search")
    def search_similar_texts(
        self, 
        query_text: str, 
        collection_name: str,
        limit: int = 5,
        score_threshold: float = 0.5,
        use_named_vector: bool = True,
        query_filter: Optional[Union[Dict[str, Any], models.Filter]] = None
    ) -> List[SearchResult]:
        """
        Search for similar texts in a collection.
        
        Args:
            query_text (str): Text to search for.
            collection_name (str): Name of the collection to search in.
            limit (int): Maximum number of results to return.
            score_threshold (float): Minimum similarity score threshold.
            use_named_vector (bool): Whether to use named vector for search.
            
        Returns:
            List[SearchResult]: List of search results with scores and payloads.
        """
        # Validate parameters
        search_params = SearchParameters(
            query_text=query_text,
            collection_name=collection_name,
            limit=limit,
            score_threshold=score_threshold,
            use_named_vector=use_named_vector,
            query_filter=query_filter
        )
        
        logger.info(f"Searching for similar texts in {search_params.collection_name}")
        
        try:
            # Encode query text to vector
            query_vector = self.embedding_model.encode_texts([search_params.query_text])
            
            # Prepare search parameters
            search_params_dict = {
                "collection_name": search_params.collection_name,
                "query_vector": query_vector[0],
                "limit": search_params.limit,
                "score_threshold": search_params.score_threshold,
                "with_payload": True,
                "with_vectors": False
            }
            
            # Use named vector if specified
            if search_params.use_named_vector:
                # Qdrant expects NamedVector format
                from qdrant_client.http.models import NamedVector
                search_params_dict["query_vector"] = NamedVector(
                    name="vector",
                    vector=query_vector[0].tolist()
                )
            else:
                search_params_dict["query_vector"] = query_vector[0].tolist()

            if search_params.query_filter is not None:
                filter_param = search_params.query_filter
                if isinstance(filter_param, dict):
                    filter_param = models.Filter(**filter_param)
                search_params_dict["query_filter"] = filter_param
            
            # Perform search
            search_results = self.client.search(**search_params_dict)
            
            # Convert to validated SearchResult objects
            validated_results = []
            for result in search_results:
                validated_result = SearchResult(
                    id=result.id,
                    score=result.score,
                    payload=result.payload
                )
                validated_results.append(validated_result)
            
            logger.info(f"Search completed: {len(validated_results)} results found")
            return validated_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    @log_performance("Recommendation Search")
    def recommend_similar_texts(
        self,
        positive_ids: List[int],
        collection_name: str,
        negative_ids: Optional[List[int]] = None,
        limit: int = 3,
        score_threshold: float = 0.25,
        use_named_vector: bool = True,
        query_filter: Optional[Union[Dict[str, Any], models.Filter]] = None
    ) -> List[SearchResult]:
        """
        Generate recommendations based on positive examples.
        
        Args:
            positive_ids (List[int]): List of positive example IDs.
            collection_name (str): Name of the collection for recommendations.
            limit (int): Maximum number of recommendations to return.
            score_threshold (float): Minimum similarity score threshold.
            use_named_vector (bool): Whether to use named vector for recommendations.
            
        Returns:
            List[SearchResult]: List of recommended results with scores and payloads.
        """
        # Validate parameters
        rec_params = RecommendationParameters(
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            collection_name=collection_name,
            limit=limit,
            score_threshold=score_threshold,
            use_named_vector=use_named_vector,
            query_filter=query_filter
        )
        
        logger.info(f"Generating recommendations for {rec_params.collection_name}")
        
        try:
            # Prepare recommendation parameters
            rec_params_dict = {
                "collection_name": rec_params.collection_name,
                "positive": rec_params.positive_ids,
                "limit": rec_params.limit,
                "score_threshold": rec_params.score_threshold,
                "with_payload": True,
                "with_vectors": False
            }
            
            if rec_params.negative_ids:
                rec_params_dict["negative"] = rec_params.negative_ids

            # Use named vector if specified
            if rec_params.use_named_vector:
                # Qdrant expects proper format for recommendations
                rec_params_dict["positive"] = rec_params.positive_ids
                rec_params_dict["using"] = "vector"
            else:
                rec_params_dict["positive"] = rec_params.positive_ids

            if rec_params.query_filter is not None:
                filter_param = rec_params.query_filter
                if isinstance(filter_param, dict):
                    filter_param = models.Filter(**filter_param)
                rec_params_dict["query_filter"] = filter_param
            
            # Generate recommendations
            recommendations = self.client.recommend(**rec_params_dict)
            
            # Convert to validated SearchResult objects
            validated_results = []
            for rec in recommendations:
                validated_result = SearchResult(
                    id=rec.id,
                    score=rec.score,
                    payload=rec.payload
                )
                validated_results.append(validated_result)
            
            logger.info(f"Recommendation completed: {len(validated_results)} results found")
            return validated_results
            
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            raise
    
    def print_search_results(self, results: List[SearchResult], title: str = "Search Results") -> None:
        """
        Print search results in a formatted way.
        
        Args:
            results (List[SearchResult]): List of search results to print.
            title (str): Title for the results section.
        """
        if not results:
            print(f"{title}: No results found")
            return
        
        print(f"\n{title}:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   ID: {result.id}")
            print(f"   Payload: {result.payload}")
            print()
