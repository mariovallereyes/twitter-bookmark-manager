"""
Enhanced vector store implementation that provides persistence
and caching while maintaining isolation from existing implementation.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import os
import threading

# Set up logging
logger = logging.getLogger(__name__)

class SearchCache:
    """
    Cache for search results to improve performance.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """Initialize the cache"""
        self._cache = {}
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()
        
        logger.info(f"Initialized SearchCache with max_size={max_size}, ttl={ttl}")
    
    def _check_expiration(self) -> None:
        """Check for and remove expired cache entries"""
        with self._lock:
            now = time.time()
            
            # Find expired keys
            expired_keys = []
            for key, entry in self._cache.items():
                if now > entry['expires']:
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del self._cache[key]
                
            # Trim cache if over max size
            if len(self._cache) > self._max_size:
                # Sort by last_accessed
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: x[1]['last_accessed']
                )
                
                # Keep only max_size newest items
                to_remove = sorted_items[:len(self._cache) - self._max_size]
                
                for key, _ in to_remove:
                    del self._cache[key]
    
    def generate_key(self, query: str, **kwargs) -> str:
        """Generate a cache key for the query and additional parameters"""
        import hashlib
        
        # Create a string representation of the kwargs
        kwargs_str = json.dumps(sorted(kwargs.items()), sort_keys=True)
        
        # Create a hash of the query and kwargs
        key_string = f"{query}|{kwargs_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set a cache entry"""
        with self._lock:
            expiration = time.time() + (ttl or self._ttl)
            
            self._cache[key] = {
                'data': data,
                'expires': expiration,
                'last_accessed': time.time()
            }
            
            logger.debug(f"Added entry to cache with key={key}")
            
            # Check for expired entries
            self._check_expiration()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cache entry"""
        with self._lock:
            if key not in self._cache:
                return None
                
            entry = self._cache[key]
            
            # Check if expired
            if time.time() > entry['expires']:
                del self._cache[key]
                return None
                
            # Update last accessed time
            entry['last_accessed'] = time.time()
            
            logger.debug(f"Cache hit for key={key}")
            return entry['data']
    
    def clear(self) -> None:
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared search cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'ttl': self._ttl
            }


class PersistentVectorStore:
    """
    Enhanced vector store implementation that provides persistence and caching
    while maintaining compatibility with the existing system.
    """
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 1.0, 
                 cache_ttl: int = 3600):
        """Initialize the vector store"""
        self._qdrant_client = None
        self._collection_name = "bookmarks"
        self._initialized = False
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._last_error = None
        self._lock = threading.Lock()
        
        # Initialize cache
        self._cache = SearchCache(ttl=cache_ttl)
        
        logger.info(f"Initialized PersistentVectorStore with max_retries={max_retries}")
    
    def ensure_initialized(self) -> None:
        """Ensure the vector store is initialized"""
        with self._lock:
            if self._initialized:
                return
            
            try:
                # Add retry logic for robustness
                retry_count = 0
                last_error = None
                
                while retry_count < self._max_retries:
                    try:
                        self._initialize_store()
                        self._initialized = True
                        logger.info("Vector store initialized successfully")
                        return
                    except Exception as e:
                        retry_count += 1
                        last_error = e
                        logger.warning(f"Initialization attempt {retry_count} failed: {e}")
                        
                        if retry_count < self._max_retries:
                            delay = self._retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                            logger.info(f"Retrying in {delay:.2f}s...")
                            time.sleep(delay)
                
                # If we got here, all retries failed
                error_msg = f"Failed to initialize vector store after {self._max_retries} attempts: {last_error}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            except Exception as e:
                logger.error(f"Initialization error: {e}")
                raise
    
    async def initialize_async(self) -> None:
        """Initialize any asynchronous components (placeholder for now)"""
        # Most initialization is done synchronously already
        # This method exists for API compatibility with async initialization pattern
        if not self._initialized:
            self.ensure_initialized()
        
        logger.info("Async initialization of vector store completed")
    
    def _initialize_store(self) -> None:
        """Initialize the vector store with proper error handling"""
        try:
            # Import Qdrant client
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            # Try to use local file system first
            data_path = os.environ.get('QDRANT_PATH', './qdrant_data')
            
            # Ensure the data directory exists
            if not os.path.exists(data_path):
                os.makedirs(data_path, exist_ok=True)
                logger.info(f"Created Qdrant data directory: {data_path}")
            
            # Initialize Qdrant client with persistent storage
            self._qdrant_client = QdrantClient(path=data_path)
            
            # Check if the collection exists
            collections = self._qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self._collection_name not in collection_names:
                logger.warning(f"Collection '{self._collection_name}' not found, creating it")
                
                # Create the collection
                self._qdrant_client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                
                logger.info(f"Created collection '{self._collection_name}'")
            
            # Check if we have any vectors loaded
            collection_info = self._qdrant_client.get_collection(self._collection_name)
            points_count = collection_info.vectors_count
            
            logger.info(f"Vector store initialized with {points_count} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def search(self, 
              query: str, 
              limit: int = 10,
              **kwargs) -> List[Dict[str, Any]]:
        """
        Search for vectors similar to the query
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        # Ensure the store is initialized
        self.ensure_initialized()
        
        # Check cache first
        cache_key = self._cache.generate_key(query, limit=limit, **kwargs)
        cached_results = self._cache.get(cache_key)
        if cached_results is not None:
            logger.info(f"Returning cached vector search results for query='{query[:30]}'")
            return cached_results
            
        # Execute the search
        retry_count = 0
        while retry_count < self._max_retries:
            try:
                results = self._execute_search(query, limit, **kwargs)
                
                # Cache the results
                self._cache.set(cache_key, results)
                
                return results
                
            except Exception as e:
                retry_count += 1
                self._last_error = str(e)
                
                if retry_count < self._max_retries:
                    wait_time = self._retry_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Vector search attempt {retry_count} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Vector search failed after {retry_count} attempts: {e}")
                    return []
    
    def _execute_search(self, 
                       query: str, 
                       limit: int = 10,
                       **kwargs) -> List[Dict[str, Any]]:
        """Execute the vector search"""
        try:
            # Import necessary modules
            import openai
            from openai import OpenAI
            
            # Create OpenAI client
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            # Get embedding for the query
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            
            # Get the embedding
            embedding = response.data[0].embedding
            
            # Search Qdrant
            search_results = self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=embedding,
                limit=limit
            )
            
            # Process results
            results = []
            for result in search_results:
                # Extract payload and score
                payload = result.payload
                score = result.score
                
                # Create result dictionary
                result_dict = {
                    'id': payload.get('id'),
                    'bookmark_id': payload.get('id'),  # For backward compatibility
                    'url': payload.get('url'),
                    'title': payload.get('title'),
                    'description': payload.get('description'),
                    'image_url': payload.get('image_url'),
                    'created_at': payload.get('created_at'),
                    'user_id': payload.get('user_id'),
                    'username': payload.get('username'),
                    'score': score,
                    'source': 'vector'
                }
                
                results.append(result_dict)
            
            logger.info(f"Vector search returned {len(results)} results for query='{query[:30]}'")
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            raise
    
    def add_bookmark(self, bookmark_data: Dict[str, Any]) -> bool:
        """
        Add a bookmark to the vector store
        
        Args:
            bookmark_data: Bookmark data
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure the store is initialized
        self.ensure_initialized()
        
        try:
            # Import necessary modules
            import openai
            from openai import OpenAI
            from qdrant_client.http import models
            
            # Extract required fields
            bookmark_id = bookmark_data.get('id')
            title = bookmark_data.get('title', '')
            description = bookmark_data.get('description', '')
            url = bookmark_data.get('url', '')
            
            # Skip if missing required data
            if not bookmark_id or not title:
                logger.warning(f"Skipping bookmark with id={bookmark_id} due to missing data")
                return False
            
            # Create text to embed
            text_to_embed = f"{title} {description}"
            
            # Create OpenAI client
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            # Get embedding
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_to_embed
            )
            
            # Get the embedding
            embedding = response.data[0].embedding
            
            # Add to Qdrant
            self._qdrant_client.upsert(
                collection_name=self._collection_name,
                points=[
                    models.PointStruct(
                        id=str(bookmark_id),
                        vector=embedding,
                        payload=bookmark_data
                    )
                ]
            )
            
            logger.info(f"Added bookmark with id={bookmark_id} to vector store")
            
            # Clear cache since we've modified the vector store
            self.clear_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding bookmark: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the search cache"""
        self._cache.clear()
        logger.info("Cleared vector store cache")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the vector store"""
        status = {
            'initialized': self._initialized,
            'last_error': self._last_error,
            'cache': self._cache.get_stats()
        }
        
        if self._initialized and self._qdrant_client:
            try:
                # Get collection info
                collection_info = self._qdrant_client.get_collection(self._collection_name)
                
                # Add collection information
                status['collection'] = {
                    'name': self._collection_name,
                    'vectors_count': collection_info.vectors_count,
                    'status': 'available'
                }
            except Exception as e:
                status['collection'] = {
                    'name': self._collection_name,
                    'status': 'error',
                    'error': str(e)
                }
        
        return status 