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
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store with a specified persist directory.
        
        Args:
            persist_directory: Directory to persist the vector store. If None,
                              a default location in the user's directory is used.
        """
        self.model = None  # Will be loaded lazily when needed
        self.client = None
        
        # Generate a unique instance ID to prevent collisions, like PythonAnywhere
        instance_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        self.collection_name = f"bookmark_embeddings_{instance_id}"
        logger.info(f"Creating Qdrant in-memory instance with ID: {instance_id}")
        
        self.vector_size = 384  # Will be updated when model is loaded
        
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
                timeout=10.0,  # Add timeout for operations
                prefer_grpc=False  # Use HTTP instead of gRPC for better compatibility
            )
            logger.info(f"Qdrant client initialized successfully in memory mode")
            
            # Create collection
            try:
                # Always create a new collection with our unique name
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to create collection {self.collection_name}: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _ensure_model_loaded(self):
        """Ensure the model is loaded before using it"""
        if self.model is None:
            self._initialize_model()
            
    def _initialize_model(self):
        """Initialize the model with the most memory-efficient settings possible"""
        # Log memory before loading
        memory_before = self.get_memory_usage()
        logger.info(f"Memory before model loading: {memory_before}")
        
        # Basic environment settings to reduce memory usage
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers parallelism
        
        # Initialize model - use tiny model to minimize memory
        try:
            logger.info(f"Initializing SentenceTransformer model with minimal memory footprint")
            # Use smallest viable model
            # Use all-MiniLM-L6-v2 which is the same model PythonAnywhere uses but with fp16
            from torch import dtype as torch_dtype
            try:
                # Try to use half precision
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                                               device='cpu', 
                                               torch_dtype=torch_dtype.float16)  # Use fp16 to save memory
                logger.info(f"Model initialized with half-precision (fp16)")
            except Exception as e:
                logger.warning(f"Half-precision failed: {str(e)}. Falling back to full precision.")
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                                               device='cpu')
            
            logger.info(f"Model initialized successfully")
            
            # Update vector size based on the actual model
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Vector size updated to {self.vector_size} based on model dimension")
            
            # Set model to evaluation mode to reduce memory usage
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        # Log memory after loading
        memory_after = self.get_memory_usage()
        logger.info(f"Memory after model loading: {memory_after}")
        logger.info(f"Memory increase: {float(memory_after.rstrip('MB')) - float(memory_before.rstrip('MB')):.2f}MB")
        
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
            
    def rebuild_user_vectors(self, user_id, batch_size=2, session_id=None):
        """
        Rebuild vector embeddings for all of a user's bookmarks
        
        Args:
            user_id: User ID to rebuild vectors for
            batch_size: Number of bookmarks to process in a batch before cleanup
            session_id: Optional session ID for tracking rebuild progress
        
        Returns:
            Boolean indicating success
        """
        rebuild_id = session_id or ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        
        try:
            if not user_id:
                logger.error(f"❌ [REBUILD-{rebuild_id}] Invalid user_id: {user_id}")
                return False
                
            logger.info(f"🔄 [REBUILD-{rebuild_id}] Starting vector rebuild for user {user_id}")
            
            # Get a new db connection
            conn = get_db_connection()
            
            # Get bookmarks for the user
            try:
                stmt = sql_text("""
                    SELECT id, bookmark_id, text, tweet_content 
                    FROM bookmarks
                    WHERE user_id = :user_id
                    ORDER BY id
                """).bindparams(bindparam('user_id', type_=Integer))
                
                result = conn.execute(stmt, {"user_id": user_id})
                bookmark_count = 0
                
                # First pass: count bookmarks and pre-filter empties
                valid_bookmarks = []
                for row in result:
                    bookmark_count += 1
                    # Pre-filter empty bookmarks to save memory later
                    text = row.text or ''
                    if not text.strip():
                        logger.info(f"⚠️ [REBUILD-{rebuild_id}] Pre-filtering bookmark {row.bookmark_id} due to empty text")
                        continue
                    valid_bookmarks.append((row.id, row.bookmark_id, row.text, row.tweet_content))
                
                # Commit and close this connection before processing
                conn.close()
                
                filtered_count = bookmark_count - len(valid_bookmarks)
                logger.info(f"📊 [REBUILD-{rebuild_id}] Found {bookmark_count} bookmarks, filtered {filtered_count} empty ones")
                    
                # Process in very small batches with forced cleanup between each batch
                # This is extremely aggressive memory management for restricted environments
                
                # EXTREME MEMORY LIMITATION: Process only a small subset of bookmarks
                # Hard cap at 50 bookmarks even if there are more - Railway has severe memory constraints
                max_bookmarks = 50  # Hard limit for Railway's constraints
                if len(valid_bookmarks) > max_bookmarks:
                    logger.warning(f"⚠️ [REBUILD-{rebuild_id}] MEMORY CONSTRAINT: Limiting to {max_bookmarks} bookmarks out of {len(valid_bookmarks)}")
                    valid_bookmarks = valid_bookmarks[:max_bookmarks]
                
                total_processed = 0
                batch_num = 0
                
                # Process batches with extremely small batch size
                for i in range(0, len(valid_bookmarks), batch_size):
                    batch = valid_bookmarks[i:i+batch_size]
                    batch_num += 1
                    
                    logger.info(f"🔄 [REBUILD-{rebuild_id}] Processing batch {batch_num} with {len(batch)} bookmarks")
                    
                    # Process each bookmark individually for maximum memory control
                    for bookmark_id, tweet_id, text, tweet_content in batch:
                        # Initialize the model for just this one bookmark
                        start_time = time.time()
                        memory_before = self.get_memory_usage()
                        
                        try:
                            logger.info(f"🔄 [REBUILD-{rebuild_id}] Processing bookmark {bookmark_id}, memory: {memory_before}")
                            
                            # Load model for just this one bookmark
                            self._ensure_model_loaded()
                            
                            # Generate embedding
                            if text and text.strip():
                                # Truncate text to reasonable length
                                text = text[:5000]  # Limit to 5000 chars
                                
                                embedding = self.model.encode(text)
                                
                                # Create a deterministic UUID from user_id and bookmark_id
                                namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
                                seed = f"{user_id}_{bookmark_id}"
                                point_id = str(uuid.uuid5(namespace, seed))
                                
                                # Create payload
                                payload = {
                                    "bookmark_id": str(bookmark_id),
                                    "text": text[:500],  # Store only beginning in payload
                                    "user_id": user_id,
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                # Add to vector store
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
                                
                                total_processed += 1
                                
                                # Clean up immediately
                                del embedding
                            else:
                                logger.info(f"⚠️ [REBUILD-{rebuild_id}] Skipping bookmark {bookmark_id} due to empty text")
                                
                        except Exception as e:
                            logger.error(f"❌ [REBUILD-{rebuild_id}] Error embedding bookmark {bookmark_id}: {str(e)}")
                            
                        finally:
                            # Unload model after each bookmark
                            self._unload_model()
                            
                            # Aggressive memory cleanup
                            self.clean_memory()
                            
                            # Memory tracking
                            memory_after = self.get_memory_usage()
                            end_time = time.time()
                            logger.info(f"⏱️ [REBUILD-{rebuild_id}] Bookmark {bookmark_id} processed in {end_time - start_time:.2f}s, memory: {memory_before} → {memory_after}")
                            
                    # After each batch, force an extra aggressive memory cleanup
                    self.clean_memory()
                    
                    # Log progress
                    progress = (total_processed / len(valid_bookmarks)) * 100
                    logger.info(f"✅ [REBUILD-{rebuild_id}] Batch {batch_num} complete: {total_processed}/{len(valid_bookmarks)} ({progress:.1f}%)")
                    
                    # Add a delay between batches to avoid overwhelming the system
                    time.sleep(1.0)  # Longer delay to let system recover
                
                logger.info(f"✅ [REBUILD-{rebuild_id}] Completed vector rebuild for user {user_id}")
                logger.info(f"📊 [REBUILD-{rebuild_id}] Processed {total_processed} bookmarks out of {bookmark_count} total")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ [REBUILD-{rebuild_id}] Error rebuilding vectors: {str(e)}")
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"❌ [REBUILD-{rebuild_id}] Unexpected error in rebuild_user_vectors: {str(e)}")
            logger.error(traceback.format_exc())
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
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            return f"{memory_mb:.2f}MB"
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return "unknown"

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
        logger.debug(f"Memory cleanup: {memory_before} → {memory_after}")

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