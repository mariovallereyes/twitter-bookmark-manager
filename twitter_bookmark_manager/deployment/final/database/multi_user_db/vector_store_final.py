"""
Vector store implementation for PythonAnywhere using Qdrant.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import logging
import os
import time
import glob
from pathlib import Path
import uuid
import hashlib
import random
import string

# Set up logging
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = None, user_id: str = None):
        """Initialize Qdrant client in memory mode to avoid lock issues
        
        Args:
            persist_directory: Directory where vector store data will be persisted
            user_id: User ID for multi-user support to create user-specific collections
        """
        try:
            # Define config locally to avoid dependency on pythonanywhere directory
            VECTOR_STORE_CONFIG = {
                'persist_directory': os.environ.get('VECTOR_STORE_PATH', './vector_db'),
                'collection_name': 'bookmarks',
                'embedding_dimension': 384
            }
            
            # Store path for later use (may be used for backup/restore)
            self.vector_db_path = VECTOR_STORE_CONFIG['persist_directory']
            
            # Store user_id for filtering operations
            self.user_id = user_id
            
            # Initialize with retry logic
            retry_count = 0
            max_retries = 3
            last_error = None
            
            # Generate a unique instance ID to prevent collisions
            instance_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            logger.info(f"Creating Qdrant in-memory instance with ID: {instance_id}")
            
            while retry_count < max_retries:
                try:
                    # Initialize Qdrant client in MEMORY mode to avoid lock issues
                    self.client = QdrantClient(
                        location=":memory:",  # Use in-memory storage
                        timeout=10.0,  # Add timeout for operations
                        prefer_grpc=False  # Use HTTP instead of gRPC for better compatibility
                    )
                    
                    # Initialize sentence transformer
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    
                    # Create collection with a unique name since we're in memory mode
                    # If user_id is provided, include it in the collection name
                    if user_id:
                        self.collection_name = f"bookmarks_user_{user_id}_{instance_id}"
                        logger.info(f"Creating user-specific collection for user_id: {user_id}")
                    else:
                        self.collection_name = f"bookmarks_{instance_id}"
                    
                    self.vector_size = self.model.get_sentence_embedding_dimension()
                    
                    # Always create a new collection since we're in memory mode
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    
                    logger.info(f"✅ VectorStore initialized with Qdrant in-memory backend (collection: {self.collection_name})")
                    return  # Success, exit the retry loop
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Attempt {retry_count+1}/{max_retries} to initialize Qdrant failed: {last_error}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2)  # Wait before retrying
            
            # If we get here, all retries failed
            error_msg = f"Error initializing VectorStore: {last_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {e}")
            raise

    def _generate_uuid(self, bookmark_id: str) -> str:
        """Generate a deterministic UUID from bookmark ID"""
        # Create a namespace UUID (using URL namespace)
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        # Create a UUID using the namespace and bookmark ID
        return str(uuid.uuid5(namespace, str(bookmark_id)))

    def add_bookmark(self, 
                    bookmark_id: str,
                    text: str,
                    metadata: Dict[str, Any] = None):
        """Add a single bookmark to the vector store"""
        try:
            # Generate embedding
            embedding = self.model.encode(text).tolist()
            
            # Generate UUID from bookmark ID
            point_id = self._generate_uuid(bookmark_id)
            
            # Store original ID in metadata
            if metadata is None:
                metadata = {}
            metadata['original_id'] = bookmark_id
            
            # Add to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={"text": text, **(metadata or {})}
                    )
                ]
            )
            logger.debug(f"Added bookmark {bookmark_id} to vector store with UUID {point_id}")
        except Exception as e:
            logger.error(f"Error adding bookmark to vector store: {e}")
            raise

    def search(self, 
               query_text: str, 
               limit: int = 10, 
               exclude_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Search for bookmarks similar to the query text
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            exclude_ids: List of bookmark IDs to exclude from results
            
        Returns:
            List of dictionaries containing bookmarks and their similarity scores
        """
        try:
            # Embed the query
            query_embedding = self.model.encode(query_text).tolist()
            
            # Convert exclude_ids to UUIDs
            exclude_uuids = [self._generate_uuid(id) for id in exclude_ids] if exclude_ids else []
            
            # Prepare filter for excluding specific IDs
            id_filter = None
            if exclude_uuids:
                id_filter = models.Filter(
                    should_not=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchAny(any=exclude_uuids)
                        )
                    ]
                )
            
            # Add user_id filter if provided during initialization
            if self.user_id:
                user_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(value=self.user_id)
                        )
                    ]
                )
                
                # Combine filters if we have both
                if id_filter:
                    # Combine user_filter and id_filter
                    combined_filter = models.Filter(
                        must=[
                            models.NestedCondition(filter=user_filter.must[0])
                        ],
                        should_not=[
                            models.NestedCondition(filter=id_filter.should_not[0])
                        ]
                    )
                    search_filter = combined_filter
                else:
                    search_filter = user_filter
            else:
                search_filter = id_filter
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                filter=search_filter
            )
            
            # Process and return results
            results = []
            for scored_point in search_result:
                result = {
                    'id': scored_point.payload.get('original_id', ''),
                    'score': scored_point.score,
                    'distance': 1.0 - scored_point.score,  # Convert cosine similarity to distance
                    'metadata': scored_point.payload.get('metadata', {})
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching in vector store: {e}")
            return []

    def delete_bookmark(self, bookmark_id: str):
        """Delete a bookmark from the vector store"""
        try:
            point_id = self._generate_uuid(bookmark_id)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point_id])
            )
            logger.debug(f"Deleted bookmark {bookmark_id} from vector store")
        except Exception as e:
            logger.error(f"Error deleting bookmark from vector store: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed information about the collection"""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': collection.vectors_count,
                'status': collection.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {'error': str(e)}

    def query_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar bookmarks"""
        return self.search(query=query, n_results=n_results)

    def search_with_exclusions(self, 
                              query_embedding: List[float] = None,
                              query: str = None,
                              limit: int = 100,
                              excluded_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search that supports excluding specific bookmark IDs from results.
        
        Args:
            query_embedding: Optional pre-computed embedding
            query: Text query (will be encoded if query_embedding not provided)
            limit: Maximum number of results to return
            excluded_ids: List of bookmark IDs to exclude from results
            
        Returns:
            List of processed search results with excluded IDs filtered out
        """
        try:
            # Convert excluded_ids to a set for faster lookups
            excluded_set = set(excluded_ids or [])
            
            # Handle both embedding and text-based search
            if query_embedding is None and query is not None:
                query_embedding = self.model.encode(query).tolist()
            elif query_embedding is None and query is None:
                raise ValueError("Either query_embedding or query must be provided")
            
            # Request more results than needed to account for exclusions
            # For Qdrant, we'll use a multiplication factor to ensure enough results
            buffer_size = min(limit * 3, 1000)  # Increased cap from 300 to 1000 to allow more search results
            
            logger.info(f"Performing Qdrant search with exclusions: limit={limit}, buffer={buffer_size}, excluded={len(excluded_set)}")
            
            # Generate UUIDs for all excluded IDs
            excluded_uuids = [self._generate_uuid(bookmark_id) for bookmark_id in excluded_set]
            
            # Perform the search with the buffer size
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=buffer_size,
                # Qdrant doesn't support direct filtering by excluded IDs, so we'll filter after
            )
            
            # Process results, excluding specified IDs
            processed_results = []
            for result in results:
                try:
                    # Get the original bookmark ID from payload
                    bookmark_id = result.payload.get('original_id')
                    
                    # Skip excluded IDs
                    if bookmark_id in excluded_set:
                        continue
                        
                    processed_results.append({
                        'bookmark_id': bookmark_id,
                        'score': result.score,
                        'distance': 1.0 - result.score,  # Convert score to distance for compatibility
                        'metadata': {k: v for k, v in result.payload.items() if k not in ['text', 'original_id']},
                        'text': result.payload.get('text')
                    })
                    
                    # Stop if we have enough results after filtering
                    if len(processed_results) >= limit:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing result: {e}")
                    continue
            
            logger.info(f"Qdrant search with exclusions found {len(processed_results)} results after filtering")
            return processed_results
            
        except Exception as e:
            logger.error(f"Search with exclusions error in VectorStore: {e}")
            return [] 