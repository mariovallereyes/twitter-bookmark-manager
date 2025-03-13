"""
Vector store implementation for Railway based on PythonAnywhere's implementation.
This version uses Qdrant for vector storage and SentenceTransformer for embeddings.
"""

import os
import sys
import logging
import json
import time
import traceback
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import psutil  # Add psutil for memory tracking
import uuid
import gc  # Add import for garbage collection
import tempfile
import shutil

# Import sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# Import Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vector_store_final')

class VectorStore:
    """
    Vector Store implementation for searching similar bookmarks.
    Uses SentenceTransformer for embeddings and Qdrant for vector storage.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store with a specified persist directory.
        
        Args:
            persist_directory: Directory to persist the vector store. If None,
                              a default location in the user's directory is used.
        """
        self.model = None  # Will be loaded lazily when needed
        self.client = None
        self.collection_name = "bookmark_embeddings"
        self.vector_size = 384  # BERT embedding size
        
        # Set default persist directory for Railway if not specified
        if persist_directory is None:
            # Check environment variables first
            base_dir = os.environ.get('VECTOR_STORE_DIR', '/app/vector_store')
            self.persist_directory = base_dir
        else:
            self.persist_directory = persist_directory
            
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize Qdrant client in memory mode to avoid file locking issues
        try:
            logger.info(f"Initializing Qdrant client in memory mode to avoid file locking issues")
            self.client = QdrantClient(
                location=":memory:",  # Always use in-memory mode like PythonAnywhere
                timeout=10.0  # Add timeout for operations
            )
            logger.info(f"Qdrant client initialized successfully in memory mode")
            
            # Create collection
            try:
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"Found existing collection: {self.collection_name}")
            except Exception as e:
                logger.info(f"Collection {self.collection_name} does not exist, creating...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _ensure_model_loaded(self):
        """Ensure the model is loaded before using it"""
        if self.model is None:
            self._initialize_model()
            
    def _initialize_model(self):
        """Initialize the model with a simple, reliable approach like PythonAnywhere"""
        # Log memory before loading
        memory_before = self.get_memory_usage()
        logger.info(f"Memory before model loading: {memory_before}")
        
        # Basic environment settings to reduce memory usage
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers parallelism
        
        # Initialize model - use a smaller model consistently like PA does
        try:
            logger.info(f"Initializing SentenceTransformer model")
            # Use paraphrase-MiniLM-L3-v2 which is smaller but effective
            self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
            logger.info(f"Model initialized successfully")
            
            # Set model to evaluation mode to reduce memory usage
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        # Log memory after loading
        memory_after = self.get_memory_usage()
        logger.info(f"Memory after model loading: {memory_after}")
        
        # Force garbage collection
        gc.collect()
            
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
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Ensure consistent ID format - convert to string
            bookmark_id_str = str(bookmark_id)
            
            # Skip if text is empty
            if not text or not text.strip():
                logger.warning(f"Skipping bookmark {bookmark_id} due to empty text")
                return False
                
            # Limit text size to avoid processing extremely large texts
            text = text[:10000] if len(text) > 10000 else text
            
            # Generate embedding - simple approach like PythonAnywhere
            embedding = self.model.encode(text)
            
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding bookmark {bookmark_id} to vector store: {str(e)}")
            logger.error(traceback.format_exc())
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
            self._ensure_model_loaded()
            
            # Generate embedding for query
            query_embedding = self.model.encode(query)
            
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
            
    def rebuild_user_vectors(self, user_id, batch_size=10):
        """Rebuild vector store for a specific user's bookmarks in memory to avoid locking issues"""
        from .db_final import get_bookmarks_for_user
        
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"🔄 [REBUILD-{session_id}] Starting vector store rebuild for user {user_id}")
        
        # Force garbage collection before starting
        gc.collect()
        logger.info(f"Initial memory usage: {self.get_memory_usage()}")
        
        try:
            # Get all bookmarks for the user
            bookmarks = get_bookmarks_for_user(user_id)
            total = len(bookmarks)
            logger.info(f"Found {total} bookmarks for user {user_id}")
            logger.info(f"Memory after fetching bookmarks: {self.get_memory_usage()}")
            
            if total == 0:
                logger.info(f"No bookmarks found for user {user_id}, nothing to rebuild")
                return True
            
            # Clear existing vectors first
            logger.info(f"Memory before deleting old vectors: {self.get_memory_usage()}")
            self._delete_vectors_for_user(user_id)
            logger.info(f"Memory after deleting old vectors: {self.get_memory_usage()}")
            
            # Ensure model is loaded (lazy loading)
            self._ensure_model_loaded()
            logger.info(f"Memory after loading model: {self.get_memory_usage()}")
            
            # Process in smaller batches
            success_count = 0
            error_count = 0
            
            # Pre-filter empty bookmarks to save processing time
            valid_bookmarks = []
            for bookmark in bookmarks:
                if not bookmark.text or not bookmark.text.strip():
                    logger.info(f"⚠️ [REBUILD-{session_id}] Pre-filtering bookmark {bookmark.id} due to empty text")
                    continue
                valid_bookmarks.append(bookmark)
            
            logger.info(f"📊 [REBUILD-{session_id}] Found {len(valid_bookmarks)} bookmarks to process after filtering")
            
            # Use the filtered bookmarks list
            total = len(valid_bookmarks)
            if total == 0:
                logger.info(f"No valid bookmarks found for user {user_id} after filtering, nothing to rebuild")
                return True
            
            for i in range(0, total, batch_size):
                batch = valid_bookmarks[i:i+batch_size]
                current_batch = i//batch_size + 1
                total_batches = (total+batch_size-1)//batch_size
                
                logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch)} bookmarks)")
                
                # Process each bookmark in the batch with individual error handling
                for j, bookmark in enumerate(batch):
                    try:
                        self.add_bookmark(
                            bookmark_id=bookmark.id,
                            user_id=user_id,
                            text=bookmark.text or "",
                            metadata={
                                'author': bookmark.author_username if hasattr(bookmark, 'author_username') else None,
                                'created_at': str(bookmark.created_at) if hasattr(bookmark, 'created_at') and bookmark.created_at else None
                            }
                        )
                        success_count += 1
                        
                        # Clean memory every few bookmarks even within a batch
                        if j % 3 == 0:  # Every 3 bookmarks
                            self.clean_memory()
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error adding bookmark {bookmark.id} to vector store: {str(e)}")
                
                # Free up resources - force Python to garbage collect after each batch
                del batch
                gc.collect()
                
                # More aggressive memory cleanup between batches
                self.clean_memory()
                logger.info(f"Memory after batch {current_batch}/{total_batches}: {self.get_memory_usage()}")
                
                # Log progress
                progress = ((i + len(batch)) / total) * 100
                logger.info(f"Progress: {progress:.1f}% ({i + len(batch)}/{total})")
                
                # Give the system a moment to free up memory
                time.sleep(0.5)
            
            # Unload model to free memory when done
            self._unload_model()
            logger.info(f"Final memory usage: {self.get_memory_usage()}")
            logger.info(f"Vector rebuild completed: {success_count} successes, {error_count} errors")
            return True
            
        except Exception as e:
            logger.error(f"Error during vector rebuild for user {user_id}: {str(e)}")
            logger.error(traceback.format_exc())
            # Try to unload model even after error
            self._unload_model()
            return False
            
    def _unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            logger.info(f"Unloading model to free memory")
            self.model = None
            # Force garbage collection
            gc.collect()
            logger.info(f"Model unloaded, memory usage: {self.get_memory_usage()}")
        
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
        """Get the current memory usage of the process"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return f"{memory_mb:.2f}MB"

    def clean_memory(self):
        """
        Perform aggressive memory cleanup to avoid out-of-memory errors.
        Call this after processing each batch of bookmarks.
        """
        # Force garbage collection multiple times
        gc.collect()
        gc.collect(2)  # Generation 2 garbage collection
        
        # Try to clean PyTorch cache if it's available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared PyTorch CUDA cache")
        except:
            pass
        
        # Log current memory usage
        memory_usage = self.get_memory_usage()
        logger.info(f"After cleanup, memory usage: {memory_usage}")

# Create a singleton instance
_vector_store_instance = None

def get_vector_store(persist_directory=None):
    """
    Get a singleton instance of the vector store.
    
    Args:
        persist_directory: Ignored - using in-memory mode to avoid locking issues
        
    Returns:
        VectorStore instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        try:
            # Force in-memory mode to avoid file locking regardless of passed parameter
            logger.info("Creating vector store instance in memory mode")
            _vector_store_instance = VectorStore(persist_directory=None)
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    return _vector_store_instance 

# Add this function to match what's called in api_server_multi_user.py
def get_multi_user_vector_store(persist_directory=None):
    """
    Get a vector store instance for the multi-user environment.
    This is a wrapper around get_vector_store for compatibility.
    Always uses in-memory mode to avoid file locking issues.
    
    Args:
        persist_directory: Ignored - using in-memory mode to avoid locking issues
        
    Returns:
        VectorStore instance
    """
    return get_vector_store(None)  # Force in-memory mode

def get_memory_usage():
    """Get current memory usage as a formatted string"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    return f"{memory_mb:.1f}MB" 