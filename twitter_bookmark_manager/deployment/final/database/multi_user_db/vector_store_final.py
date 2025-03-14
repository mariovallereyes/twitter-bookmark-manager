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
import random
import string

# Import sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# Import Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Import database utilities
from sqlalchemy import text as sql_text
from database.multi_user_db.db_final import get_db_connection
from sqlalchemy.sql import bindparam
from sqlalchemy import Integer

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
    
    def __init__(self, collection_name="bookmarks"):
        self.collection_name = collection_name
        self.vector_size = 384  # for all-MiniLM-L6-v2
        self.model = None
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        if self.client is not None:
            try:
                self.client.close()
            except:
                pass
        
        # Use Railway's volume mount path for persistent storage
        storage_path = "/app/twitter_bookmark_manager/data/vector_store"
        
        try:
            # First try to connect to existing client
            self.client = QdrantClient(
                path=storage_path,
                force_new_instance=True  # Force new instance to avoid conflicts
            )
            logging.info(f"Successfully initialized Qdrant client with storage at {storage_path}")
        except Exception as e:
            logging.error(f"Error initializing Qdrant client: {e}")
            raise

    def __del__(self):
        """Cleanup when the instance is destroyed"""
        if self.client is not None:
            try:
                self.client.close()
                logging.info("Qdrant client closed successfully")
            except Exception as e:
                logging.error(f"Error closing Qdrant client: {e}")

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before using it"""
        if self.model is None:
            self._load_model()
            
    def _load_model(self):
        """Load the model optimized for low memory usage"""
        try:
            memory_before = self.get_memory_usage()
            logger.info(f"Memory before model loading: {memory_before:.2f}MB")
            
            # Import here to avoid loading torch until needed
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Force initial garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Use the smaller model to reduce memory usage
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                                           device='cpu')  # Use CPU for stability
            
            # Verify vector size matches what we expect
            model_vector_size = self.model.get_sentence_embedding_dimension()
            if model_vector_size != self.vector_size:
                logger.error(f"Model vector size {model_vector_size} does not match collection size {self.vector_size}")
                raise ValueError("Model vector size mismatch")
            
            logger.info(f"Model loaded successfully with vector size {self.vector_size}")
            
            memory_after = self.get_memory_usage()
            memory_increase = memory_after - memory_before
            logger.info(f"Memory after model loading: {memory_after:.2f}MB")
            logger.info(f"Memory increase: {memory_increase:.2f}MB")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            self._ensure_model_loaded()
            
            # Generate embedding - simple approach like PythonAnywhere
            embedding = self.model.encode(text)
            
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

# Create a singleton instance
_vector_store_instance = None

def get_vector_store(collection_name="bookmarks"):
    """
    Get a singleton instance of the vector store.
    
    Args:
        collection_name: Name of the collection to use
        
    Returns:
        VectorStore instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        try:
            logger.info(f"Creating vector store instance with collection {collection_name}")
            _vector_store_instance = VectorStore(collection_name=collection_name)
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    return _vector_store_instance

def get_multi_user_vector_store(collection_name="bookmarks"):
    """
    Get a vector store instance for the multi-user environment.
    This is a wrapper around get_vector_store for compatibility.
    
    Args:
        collection_name: Name of the collection to use
        
    Returns:
        VectorStore instance
    """
    return get_vector_store(collection_name=collection_name)

def get_memory_usage():
    """Get current memory usage in MB as a float"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    return memory_mb 