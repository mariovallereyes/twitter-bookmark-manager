"""
Vector store implementation for final environment.
Uses Qdrant for vector storage and SentenceTransformer for embeddings.
"""

import os
import sys
import logging
import json
import time
import traceback
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import psutil  # Add psutil for memory tracking
import uuid
import gc  # Add import for garbage collection
import tempfile
import shutil
import random
import string
import threading
import filelock
import numpy as np

# Configure logging
logger = logging.getLogger('vector_store_final')

# Import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("Cannot import SentenceTransformer - embedding functionality will not be available")
    SentenceTransformer = None

# Import Qdrant for vector storage with proper error handling
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http import exceptions as qdrant_exceptions
    QDRANT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Cannot import Qdrant client: {e}")
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

# Import database utilities
from sqlalchemy import text as sql_text
from database.multi_user_db.db_final import get_db_connection
from sqlalchemy.sql import bindparam
from sqlalchemy import Integer

# Constants
BASE_DIR = os.environ.get('APP_BASE_DIR', '/app')
DATABASE_DIR = os.environ.get('DATABASE_DIR', os.path.join(BASE_DIR, 'database'))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "twitter_bookmark_manager", "data", "vector_store")
QDRANT_HOST = os.environ.get('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.environ.get('QDRANT_PORT', '6333'))

# Check if vector store is disabled via environment variable
DISABLE_VECTOR_STORE = os.environ.get('DISABLE_VECTOR_STORE', 'false').lower() == 'true'
PREFER_LOCAL_VECTOR = os.environ.get('PREFER_LOCAL_VECTOR', 'true').lower() == 'true'
SHORT_TIMEOUT = float(os.environ.get('QDRANT_TIMEOUT', '2.0'))  # Short timeout to prevent hanging

# Log vector store configuration
logger.info(f"Vector store configuration: DISABLE_VECTOR_STORE={DISABLE_VECTOR_STORE}, "
            f"PREFER_LOCAL_VECTOR={PREFER_LOCAL_VECTOR}, "
            f"QDRANT_HOST={QDRANT_HOST}, QDRANT_PORT={QDRANT_PORT}, "
            f"TIMEOUT={SHORT_TIMEOUT}s")

# Ensure vector store directory exists
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Flag to track if we should stop trying server mode
_disable_server_mode = False

class VectorStoreMultiUser:
    """
    Vector store implementation for multi-user support.
    Uses a singleton pattern and file locking to prevent concurrent access issues.
    """
    _instances = {}  # Class-level dictionary to store instances
    _lock = threading.Lock()  # Class-level lock for thread safety
    
    def __new__(cls, user_id):
        """
        Implement singleton pattern per user_id to prevent multiple instances
        trying to access the same storage location.
        """
        with cls._lock:
            if user_id not in cls._instances:
                instance = super(VectorStoreMultiUser, cls).__new__(cls)
                cls._instances[user_id] = instance
            return cls._instances[user_id]
    
    def __init__(self, user_id):
        """Initialize the vector store for a specific user."""
        # Only initialize once per instance
        if hasattr(self, 'initialized'):
            return
            
        self.user_id = user_id
        self.initialized = False
        
        try:
            # Set up paths
            self.base_dir = os.environ.get('APP_BASE_DIR', '/app')
            self.data_dir = os.path.join(self.base_dir, 'twitter_bookmark_manager', 'data')
            self.vector_store_dir = os.path.join(self.data_dir, 'vector_store')
            self.lock_file = os.path.join(self.data_dir, f'vector_store_{user_id}.lock')
            
            # Ensure directories exist
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.vector_store_dir, exist_ok=True)
            
            # Initialize with file lock to prevent concurrent access
            lock = filelock.FileLock(self.lock_file, timeout=60)  # 60 second timeout
            
            with lock:
                self._initialize_vector_store()
                
            self.initialized = True
            logger.info(f"✅ Vector store initialized for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _initialize_vector_store(self):
        """Initialize the vector store components with proper error handling."""
        try:
            # Initialize sentence transformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize Qdrant client with proper settings for single-instance use
            self.client = QdrantClient(
                path=self.vector_store_dir,
                force_disable_multiple_clients_check=True  # Allow multiple workers to access
            )
            
            # Create collection if it doesn't exist
            collection_name = "bookmarks"
            try:
                collections = self.client.get_collections()
                collection_names = [c.name for c in collections.collections]
                
                if collection_name not in collection_names:
                    logger.info(f"Creating vector store instance with collection {collection_name}")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=384,  # Dimension for all-MiniLM-L6-v2
                            distance=models.Distance.COSINE
                        )
                    )
            except Exception as e:
                logger.error(f"Error checking/creating collection: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error in _initialize_vector_store: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup when the instance is destroyed."""
        try:
            # Remove instance from _instances
            with self._lock:
                if self.user_id in self._instances:
                    del self._instances[self.user_id]
            
            # Close client connection
            if hasattr(self, 'client'):
                self.client.close()
        except:
            pass  # Ignore cleanup errors

class VectorStore:
    """
    Vector Store implementation for searching similar bookmarks.
    Uses SentenceTransformer for embeddings and Qdrant for vector storage.
    """
    
    def __init__(self, collection_name="bookmarks", server_mode=False):
        """Initialize the vector store."""
        self.collection_name = collection_name
        self.client = None
        self.model = None
        self.vector_size = 384  # for all-MiniLM-L6-v2
        self.server_mode = server_mode
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client with proper error handling."""
        try:
            # Clean up any existing connections
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
                self.client = None

            if self.server_mode:
                # Server mode - use REST API
                logger.info(f"Initializing Qdrant client in server mode with timeout={SHORT_TIMEOUT}s")
                self.client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    timeout=SHORT_TIMEOUT  # Short timeout to fail fast
                )
            else:
                # Local mode - use file system
                logger.info(f"Initializing Qdrant client in local mode")
                self.client = QdrantClient(
                    path=VECTOR_STORE_PATH,
                    prefer_grpc=False,
                    force_disable_multiple_clients_check=True  # Allow multiple workers
                )

            # Ensure collection exists
            try:
                collections = self.client.get_collections()
                collection_names = [c.name for c in collections.collections]
                
                if self.collection_name not in collection_names:
                    logger.info(f"Creating collection {self.collection_name}")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        )
                    )
            except Exception as e:
                logger.error(f"Error checking/creating collection: {str(e)}")
                raise

            logger.info("Successfully initialized vector store client")

        except Exception as e:
            logger.error(f"Error initializing vector store client: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("Successfully loaded sentence transformer model")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        try:
            if self.model is None:
                self._load_model()
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def add_bookmark(self, bookmark_id: int, text: str, user_id: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a bookmark to the vector store.
        
        Args:
            bookmark_id: ID of the bookmark
            text: Text content of the bookmark
            user_id: User ID who owns the bookmark
            metadata: Optional metadata for the bookmark
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("Vector store not properly initialized")
            return False
            
        try:
            # Ensure consistent ID format - convert to string
            bookmark_id_str = str(bookmark_id)
            
            # Skip if text is empty
            if not text or not text.strip():
                logger.warning(f"Skipping bookmark {bookmark_id} due to empty text")
                return False
                
            # Limit text size to avoid processing extremely large texts
            text = text[:10000] if len(text) > 10000 else text
            
            # Load model, generate embedding, then unload immediately
            memory_before = self.get_memory_usage()
            logger.info(f"Memory before embedding (bookmark {bookmark_id}): {memory_before}")
            
            # Ensure model is loaded
            self._load_model()
            
            # Generate embedding - simple approach like PythonAnywhere
            embedding = self._get_embedding(text)
            
            # Immediately unload model to save memory
            self._unload_model()
            
            memory_after = self.get_memory_usage()
            logger.info(f"Memory after unloading model: {memory_after}")
            
            # Create a deterministic UUID from user_id and bookmark_id
            # Use uuid5 with a namespace to create a deterministic UUID
            namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
            seed = f"{user_id}_{bookmark_id_str}"
            point_id = str(uuid.uuid5(namespace, seed))
            
            # Create payload with metadata
            payload = {
                "bookmark_id": bookmark_id_str,
                "text": text[:1000] if text else "",  # Limit text size in payload
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add metadata if provided
            if metadata:
                payload.update(metadata)
            
            # Create point with metadata
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            
            # Upsert point to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            # Clean up to reduce memory usage
            del embedding
            self.clean_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding bookmark {bookmark_id} to vector store: {str(e)}")
            logger.error(traceback.format_exc())
            self._unload_model()  # Make sure to unload model even on error
            return False
            
    def find_similar(self, query: str, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find bookmarks similar to the query text.
        
        Args:
            query: Query text to find similar bookmarks
            user_id: User ID to filter results by
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing bookmark IDs and scores
        """
        if not self.client:
            logger.error("Vector store not properly initialized")
            return []
            
        try:
            # Ensure model is loaded
            self._load_model()
            
            # Generate embedding for query
            query_embedding = self._get_embedding(query)
            
            # Search for similar vectors
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="user_id",
                            match=rest.MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=limit
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "bookmark_id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", "")[:100] + "..." if hit.payload.get("text") else ""
                })
                
            logger.info(f"Found {len(results)} similar bookmarks for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar bookmarks: {str(e)}")
            logger.error(traceback.format_exc())
            return []
            
    def _add_vector(self, bookmark_id: int, embedding: List[float], user_id: int) -> bool:
        """Add a vector to the store with its metadata"""
        try:
            # Create a deterministic UUID from user_id and bookmark_id
            bookmark_id_str = str(bookmark_id)
            namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
            seed = f"{user_id}_{bookmark_id_str}"
            point_id = str(uuid.uuid5(namespace, seed))
            
            # Create point with metadata
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "bookmark_id": bookmark_id_str,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Upsert point to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
        except Exception as e:
            logger.error(f"Error adding vector for bookmark {bookmark_id}: {str(e)}")
            return False

    def rebuild_user_vectors(self, user_id, bookmarks):
        """Rebuild vectors for a user's bookmarks in batches"""
        try:
            # Re-initialize client to ensure clean state
            self._initialize_client()
            
            collection_name = f"{self.collection_name}_{user_id}"
            logging.info(f"Starting vector rebuild for user {user_id} in collection {collection_name}")
            
            # Create or recreate collection
            self._recreate_collection(collection_name)
            
            # Process in smaller batches
            batch_size = 3
            total_batches = (len(bookmarks) + batch_size - 1) // batch_size
            
            for i in range(0, len(bookmarks), batch_size):
                batch = bookmarks[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logging.info(f"Processing batch {batch_num}/{total_batches} for user {user_id}")
                
                # Process batch
                points = []
                for bookmark in batch:
                    try:
                        vector = self._get_embedding(bookmark['text'])
                        points.append(PointStruct(
                            id=bookmark['bookmark_id'],
                            vector=vector.tolist(),
                            payload={"text": bookmark['text']}
                        ))
                    except Exception as e:
                        logging.error(f"Error processing bookmark {bookmark['bookmark_id']}: {e}")
                
                # Upload batch
                if points:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                
                # Clean up after batch
                gc.collect()
                time.sleep(1)  # Small delay between batches
            
            logging.info(f"Successfully rebuilt vectors for user {user_id}")
            return True, "Vector store rebuilt successfully"
            
        except Exception as e:
            error_msg = f"Error rebuilding vector store: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
        finally:
            # Ensure client is properly closed
            if self.client is not None:
                try:
                    self.client.close()
                except Exception as e:
                    logging.error(f"Error closing client: {e}")
            gc.collect()

    def _recreate_collection(self, collection_name):
        """Recreate the collection with proper error handling"""
        try:
            # Try to delete if exists
            try:
                self.client.delete_collection(collection_name)
                logging.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                logging.info(f"Collection didn't exist or couldn't be deleted: {e}")

            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logging.info(f"Created new collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error recreating collection: {e}")
            raise

    def _unload_model(self):
        """Unload model to free memory"""
        try:
            logger.info("Unloading model to free memory")
            if hasattr(self, 'model'):
                import torch
                
                # Delete model and clear CUDA cache
                del self.model
                self.model = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Force garbage collection
                gc.collect()
                
                memory_after = self.get_memory_usage()
                logger.info(f"Model unloaded, memory usage: {memory_after:.2f}MB")
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
        
    def _delete_vectors_for_user(self, user_id: int) -> bool:
        """
        Delete all vectors for a specific user.
        
        Args:
            user_id: User ID whose vectors to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("Vector store client not initialized")
            return False
            
        try:
            # Delete points with filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="user_id",
                            match=rest.MatchValue(value=user_id)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted vectors for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors for user {user_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def delete_bookmark(self, bookmark_id: int, user_id: int = None) -> bool:
        """
        Delete a specific bookmark from the vector store.
        
        Args:
            bookmark_id: ID of the bookmark to delete
            user_id: Optional user ID for the bookmark owner
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("Vector store client not initialized")
            return False
            
        try:
            bookmark_id_str = str(bookmark_id)
            
            if user_id is not None:
                # If user_id is provided, we can use the same ID generation as add_bookmark
                namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
                seed = f"{user_id}_{bookmark_id_str}"
                point_id = str(uuid.uuid5(namespace, seed))
                
                # Delete point by ID
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=rest.PointIdsList(
                        points=[point_id]
                    )
                )
            else:
                # If no user_id, we need to use payload filtering
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="bookmark_id", 
                                match=rest.MatchValue(value=bookmark_id_str)
                            )
                        ]
                    )
                )
            
            logger.info(f"Deleted bookmark {bookmark_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting bookmark {bookmark_id} from vector store: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def delete_bookmarks(self, bookmark_ids: List[str]) -> bool:
        """
        Delete multiple bookmarks from the vector store.
        
        Args:
            bookmark_ids: List of bookmark IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client or not bookmark_ids:
            logger.error("Vector store client not initialized or empty bookmark_ids list")
            return False
            
        try:
            # Delete points in batches to avoid overwhelming the client
            logger.info(f"Deleting {len(bookmark_ids)} bookmarks from vector store")
            
            # Convert all IDs to strings for consistency
            ids_as_strings = [str(id) for id in bookmark_ids]
            
            # Use batches of 100 IDs at a time
            BATCH_SIZE = 100
            for i in range(0, len(ids_as_strings), BATCH_SIZE):
                batch = ids_as_strings[i:i+BATCH_SIZE]
                
                # Delete using payload filtering by bookmark_id
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="bookmark_id",
                                match=rest.OneOf(
                                    one_of=batch
                                )
                            )
                        ]
                    )
                )
                logger.info(f"Deleted batch of {len(batch)} bookmarks from vector store")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting multiple bookmarks from vector store: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector collection.
        
        Returns:
            Dictionary with collection information
        """
        if not self.client:
            logger.error("Vector store client not initialized")
            return {"error": "Vector store not initialized"}
            
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Format response
            info = {
                "name": self.collection_name,
                "vector_size": collection_info.config.params.vector_size,
                "distance": collection_info.config.params.distance,
                "points_count": collection_info.vectors_count,
                "status": "ready"
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def get_memory_usage(self):
        """Get the current memory usage of the process in MB as a float"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            return memory_mb
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return 0.0

    def clean_memory(self):
        """
        Perform memory cleanup to avoid out-of-memory errors.
        Call this after processing bookmarks.
        """
        # Force garbage collection
        gc.collect()
        
        # Try to clean PyTorch cache if it's available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared PyTorch CUDA cache")
        except:
            pass
        
        # Track memory usage before and after
        memory_before = self.get_memory_usage()
        
        # Call Python's built-in memory management
        import sys
        if hasattr(sys, 'intern'):
            sys.intern('')
        
        # Additional garbage collection for all generations
        gc.collect(0)  # Generation 0
        gc.collect(1)  # Generation 1
        gc.collect(2)  # Generation 2
        
        # Log memory usage after cleanup
        memory_after = self.get_memory_usage()
        logger.debug(f"Memory cleanup: {memory_before:.2f}MB → {memory_after:.2f}MB")

# Singleton instance with lock
_vector_store_instance = None
_vector_store_lock = threading.Lock()

def get_vector_store(collection_name="bookmarks"):
    """
    Get a singleton instance of the vector store.
    Uses double-checked locking pattern for thread safety.
    If server mode fails, falls back to local mode.
    If all modes fail, returns a DummyVectorStore.
    """
    global _vector_store_instance, _disable_server_mode
    
    # First check if vector store is disabled by environment variable
    if DISABLE_VECTOR_STORE:
        logger.warning("Vector store is disabled by environment variable")
        return DummyVectorStore("Vector store disabled by environment variable")
    
    # First check without lock
    if _vector_store_instance is None:
        with _vector_store_lock:
            # Second check with lock
            if _vector_store_instance is None:
                try:
                    logger.info(f"Initializing vector store with collection {collection_name}")
                    
                    # Try server mode first unless local mode is preferred
                    server_error = None
                    if not PREFER_LOCAL_VECTOR and not _disable_server_mode:
                        try:
                            logger.info(f"Attempting to initialize in server mode with host={QDRANT_HOST}, port={QDRANT_PORT}")
                            _vector_store_instance = VectorStore(
                                collection_name=collection_name,
                                server_mode=True
                            )
                            logger.info("Successfully initialized vector store in server mode")
                            return _vector_store_instance
                        except Exception as e:
                            server_error = e
                            logger.warning(f"Server mode failed: {str(e)}, falling back to local mode")
                            # Mark server mode as disabled for future calls
                            _disable_server_mode = True
                    elif PREFER_LOCAL_VECTOR:
                        logger.info("Local vector mode preferred by configuration")
                    else:
                        logger.info("Server mode is disabled from previous failures, trying local mode directly")
                        
                    # Try local mode
                    try:
                        logger.info("Initializing vector store in local mode")
                        _vector_store_instance = VectorStore(
                            collection_name=collection_name,
                            server_mode=False
                        )
                        logger.info("Successfully initialized vector store in local mode")
                        return _vector_store_instance
                    except Exception as local_error:
                        logger.error(f"Local mode failed: {str(local_error)}")
                        
                        # Create detailed error message
                        error_details = f"Server error: {str(server_error) if server_error else 'Not attempted'}\n" + \
                                       f"Local error: {str(local_error)}"
                        
                        # Return dummy store when all modes fail
                        _vector_store_instance = DummyVectorStore(error_details)
                
                except Exception as e:
                    logger.error(f"Unexpected error initializing vector store: {str(e)}")
                    _vector_store_instance = DummyVectorStore(str(e))
    
    return _vector_store_instance

def cleanup_vector_store():
    """Clean up the vector store instance."""
    global _vector_store_instance
    with _vector_store_lock:
        if _vector_store_instance is not None:
            try:
                if hasattr(_vector_store_instance, 'client') and _vector_store_instance.client is not None:
                    _vector_store_instance.client.close()
            except:
                pass
            _vector_store_instance = None
            logger.info("Vector store instance cleaned up")

class DummyVectorStore:
    """A dummy vector store that logs errors but doesn't fail."""
    
    def __init__(self, error_message):
        self.error_message = error_message
        self.vector_size = 384  # Same as the real store
        self.collection_name = "bookmarks"  # Default collection name
        self.client = None
        self.model = None
        self.is_dummy = True  # Flag to identify this as a dummy store
        logger.warning(f"Using dummy vector store due to initialization error: {error_message}")
    
    def __getattr__(self, name):
        def dummy_method(*args, **kwargs):
            logger.warning(f"Vector store operation '{name}' called in dummy mode: {self.error_message}")
            
            # Return appropriate values for specific methods
            if name == 'find_similar':
                return []  # Return empty list for search results
            elif name == 'add_bookmark' or name == 'add_text':
                return {"success": False, "message": "Using dummy vector store"}  # Return failure dict
            elif name == 'delete_bookmark':
                return {"success": False, "message": "Using dummy vector store"}  # Return failure dict
            elif name == 'get_collection_info':
                return {"status": "dummy", "message": self.error_message}  # Return status info
            elif name == 'rebuild_user_vectors':
                return {"success": False, "message": f"Vector store unavailable: {self.error_message}"}
            
            return None
        return dummy_method
        
    # Explicitly implement critical methods
    def add_text(self, text, metadata=None):
        """Add text to the vector store (dummy implementation)."""
        logger.warning(f"Cannot add text to vector store - using dummy mode: {self.error_message}")
        return {"success": False, "message": "Using dummy vector store"}
        
    def add_bookmark(self, bookmark_id, text=None, user_id=None, metadata=None):
        """Add bookmark to the vector store (dummy implementation)."""
        logger.warning(f"Cannot add bookmark to vector store - using dummy mode: {self.error_message}")
        return {"success": False, "message": "Using dummy vector store"}
        
    def find_similar(self, query, user_id=None, limit=5):
        """Find similar bookmarks (dummy implementation)."""
        logger.warning(f"Cannot search vector store - using dummy mode: {self.error_message}")
        return []
        
    def delete_bookmark(self, bookmark_id, user_id=None):
        """Delete bookmark from vector store (dummy implementation)."""
        logger.warning(f"Cannot delete from vector store - using dummy mode: {self.error_message}")
        return {"success": False, "message": "Using dummy vector store"}
        
    def rebuild_user_vectors(self, user_id, batch_size=20, session_id=None):
        """Rebuild user vectors (dummy implementation)."""
        logger.warning(f"Cannot rebuild vectors - using dummy mode: {self.error_message}")
        return {"success": False, "message": f"Vector store unavailable: {self.error_message}"}
        
    def clear(self):
        """Clear vector store (dummy implementation)."""
        logger.warning(f"Cannot clear vector store - using dummy mode: {self.error_message}")
        return {"success": False, "message": "Using dummy vector store"}

def get_multi_user_vector_store(collection_name="bookmarks"):
    """
    Get a vector store instance with multi-user support.
    This is a wrapper for compatibility with the multi-user environment.
    """
    # Check if vector store is disabled
    if DISABLE_VECTOR_STORE:
        logger.warning("Multi-user vector store is disabled by environment variable")
        return DummyVectorStore("Vector store disabled by environment variable")
    
    # Check if dependencies are available
    if not QDRANT_AVAILABLE or not SentenceTransformer:
        logger.error("Required dependencies for vector store are not available")
        return DummyVectorStore("Required dependencies not available")
    
    try:
        # Get user ID from context if available
        try:
            from auth.user_context_final import UserContext
            user = UserContext.get_current_user()
            user_id = user.id if user else None
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not get user context: {e}, using default user ID")
            user_id = None
        
        # If no user is found, return dummy store
        if not user_id:
            logger.warning("No user ID available for vector store, using dummy store")
            return DummyVectorStore("No user ID available")
        
        # Try to get a VectorStoreMultiUser instance
        try:
            vector_store = VectorStoreMultiUser(user_id=user_id)
            logger.info(f"Successfully initialized multi-user vector store for user {user_id}")
            return vector_store
        except Exception as e:
            logger.error(f"Error initializing multi-user vector store: {e}")
            logger.error(traceback.format_exc())
            
            # Fall back to the singleton vector store
            try:
                logger.info("Falling back to singleton vector store")
                return get_vector_store(collection_name=collection_name)
            except Exception as fallback_error:
                logger.error(f"Fallback to singleton vector store failed: {fallback_error}")
                return DummyVectorStore(f"Multi-user error: {e}, Fallback error: {fallback_error}")
    except Exception as e:
        logger.error(f"Unexpected error in get_multi_user_vector_store: {e}")
        logger.error(traceback.format_exc())
        return DummyVectorStore(str(e))

def get_memory_usage():
    """Get current memory usage in MB as a float"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    return memory_mb 