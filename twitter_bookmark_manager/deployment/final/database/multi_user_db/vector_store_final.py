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
        
        # Set default persist directory for Railway if not specified
        if persist_directory is None:
            # Use Railway's volume mount path
            self.persist_directory = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/app/twitter_bookmark_manager/data')
            self.persist_directory = os.path.join(self.persist_directory, 'vector_store')
        else:
            self.persist_directory = persist_directory
            
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        logger.info(f"Using persistent storage at: {self.persist_directory}")
        
        # Generate a unique instance ID to prevent collisions
        instance_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        self.collection_name = f"bookmark_embeddings_{instance_id}"
        
        self.vector_size = 768  # Default for all-mpnet-base-v2
        
        # Initialize Qdrant client with persistent storage
        try:
            logger.info(f"Initializing Qdrant client with persistent storage at {self.persist_directory}")
            self.client = QdrantClient(path=self.persist_directory)
            logger.info(f"Qdrant client initialized successfully with persistent storage")
            
            # Initialize model as None - will load on demand
            self.model = None
            
            # Create or get collection
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                # Check if dimensions match
                collection_info = self.client.get_collection(self.collection_name)
                current_dims = collection_info.config.params.vectors.size
                if current_dims != self.vector_size:
                    logger.warning(f"Collection dimensions mismatch: expected {self.vector_size}, found {current_dims}")
                    # Delete and recreate with correct dimensions
                    self.client.delete_collection(self.collection_name)
                    collection_exists = False
                    
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name} with {self.vector_size} dimensions")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _ensure_model_loaded(self):
        """Ensure the model is loaded before using it"""
        if self.model is None:
            self._load_model()
            
    def _load_model(self):
        """Load the model optimized for performance with 32GB memory"""
        try:
            memory_before = self.get_memory_usage()
            logger.info(f"Memory before model loading: {memory_before:.2f}MB")
            
            # Import here to avoid loading torch until needed
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Force initial garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Use the better model since we have memory
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', 
                                           device='cpu')  # Still use CPU for stability
            
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

    def rebuild_user_vectors(self, user_id: int, rebuild_id: str = None) -> bool:
        """
        Rebuild vectors for all bookmarks of a user.
        
        Args:
            user_id: User ID whose vectors to rebuild
            rebuild_id: Optional ID to track rebuild progress
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not rebuild_id:
                rebuild_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            
            logger.info(f"🔄 [REBUILD-{rebuild_id}] Starting vector rebuild for user {user_id}")
            start_time = time.time()
            
            # Delete existing vectors for user first
            self._delete_vectors_for_user(user_id)
            
            # Get a new db connection
            conn = get_db_connection()
            
            try:
                # First, count total bookmarks
                count_stmt = sql_text("""
                    SELECT COUNT(*) as total 
                    FROM bookmarks 
                    WHERE user_id = :user_id
                """).bindparams(bindparam('user_id', type_=Integer))
                
                result = conn.execute(count_stmt, {"user_id": user_id})
                total_bookmarks = result.scalar()
                
                logger.info(f"📊 [REBUILD-{rebuild_id}] Found {total_bookmarks} total bookmarks to process")
                
                # Process in smaller chunks to manage memory
                CHUNK_SIZE = 20   # Smaller chunks for more frequent progress updates
                BATCH_SIZE = 5    # Small batches for stability
                offset = 0
                total_processed = 0
                errors = 0
                last_progress_time = time.time()
                
                while offset < total_bookmarks:
                    chunk_start = time.time()
                    
                    # Log progress every 60 seconds
                    current_time = time.time()
                    if current_time - last_progress_time >= 60:
                        elapsed = current_time - start_time
                        progress = (offset / total_bookmarks) * 100
                        rate = total_processed / (elapsed / 60) if elapsed > 0 else 0
                        eta_minutes = ((total_bookmarks - total_processed) / rate) if rate > 0 else 0
                        
                        logger.info(f"""
🔄 [REBUILD-{rebuild_id}] Progress Update:
- Processed: {total_processed}/{total_bookmarks} ({progress:.1f}%)
- Errors: {errors}
- Rate: {rate:.1f} bookmarks/minute
- Running for: {elapsed/60:.1f} minutes
- ETA: {eta_minutes:.1f} minutes
""")
                        last_progress_time = current_time
                    
                    stmt = sql_text("""
                        SELECT bookmark_id, text, raw_data 
                        FROM bookmarks
                        WHERE user_id = :user_id
                        ORDER BY bookmark_id
                        LIMIT :limit OFFSET :offset
                    """).bindparams(
                        bindparam('user_id', type_=Integer),
                        bindparam('limit', type_=Integer),
                        bindparam('offset', type_=Integer)
                    )
                    
                    result = conn.execute(stmt, {
                        "user_id": user_id,
                        "limit": CHUNK_SIZE,
                        "offset": offset
                    })
                    
                    rows = result.fetchall()
                    if not rows:
                        break
                        
                    valid_bookmarks = []
                    for row in rows:
                        # Get text content, using text field first
                        text = row.text or ''
                        
                        # If text is empty, try to get full_text from raw_data
                        if not text.strip() and row.raw_data:
                            try:
                                raw_data_dict = json.loads(row.raw_data) if isinstance(row.raw_data, str) else row.raw_data
                                text = raw_data_dict.get('full_text', '')
                            except Exception as e:
                                logger.error(f"❌ [REBUILD-{rebuild_id}] Error parsing raw_data for {row.bookmark_id}: {str(e)}")
                                errors += 1
                                continue
                        
                        if not text.strip():
                            logger.info(f"⚠️ [REBUILD-{rebuild_id}] Skipping bookmark {row.bookmark_id} - no text content found")
                            continue
                            
                        # Truncate text to reasonable length
                        text = text[:10000]
                        valid_bookmarks.append((row.bookmark_id, text))
                    
                    # Process valid bookmarks in small batches
                    for i in range(0, len(valid_bookmarks), BATCH_SIZE):
                        batch = valid_bookmarks[i:i + BATCH_SIZE]
                        batch_start = time.time()
                        
                        try:
                            # Load model once for the batch
                            self._ensure_model_loaded()
                            
                            for bookmark_id, text in batch:
                                try:
                                    # Generate embedding
                                    embedding = self.model.encode(text)
                                    
                                    # Add to Qdrant
                                    success = self._add_vector(bookmark_id, embedding, user_id)
                                    if success:
                                        total_processed += 1
                                        logger.info(f"✅ [REBUILD-{rebuild_id}] Added vector for bookmark {bookmark_id}")
                                    else:
                                        errors += 1
                                        logger.error(f"❌ [REBUILD-{rebuild_id}] Failed to add vector for bookmark {bookmark_id}")
                                    
                                    # Clean up embedding
                                    del embedding
                                    
                                except Exception as e:
                                    errors += 1
                                    logger.error(f"❌ [REBUILD-{rebuild_id}] Error processing bookmark {bookmark_id}: {str(e)}")
                                    continue
                        finally:
                            # Unload model after batch
                            self._unload_model()
                            
                            # Force garbage collection
                            gc.collect()
                            
                            # Log batch timing
                            batch_time = time.time() - batch_start
                            logger.info(f"⏱️ [REBUILD-{rebuild_id}] Batch processed in {batch_time:.2f}s")
                    
                    # Move to next chunk
                    offset += CHUNK_SIZE
                    
                    # Log chunk timing
                    chunk_time = time.time() - chunk_start
                    logger.info(f"⏱️ [REBUILD-{rebuild_id}] Chunk processed in {chunk_time:.2f}s")
                
                # Log final statistics
                total_time = time.time() - start_time
                logger.info(f"""
✅ [REBUILD-{rebuild_id}] Vector rebuild completed:
- Total processed: {total_processed}/{total_bookmarks}
- Success rate: {(total_processed/total_bookmarks*100):.1f}%
- Errors: {errors}
- Total time: {total_time/60:.1f} minutes
- Average speed: {total_processed/(total_time/60):.1f} bookmarks/minute
""")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ [REBUILD-{rebuild_id}] Database error: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ [REBUILD-{rebuild_id}] Unexpected error in rebuild_user_vectors: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
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

def get_vector_store(persist_directory=None):
    """
    Get a singleton instance of the vector store.
    
    Args:
        persist_directory: Directory to persist the vector store
        
    Returns:
        VectorStore instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        try:
            logger.info("Creating vector store instance with persistent storage")
            _vector_store_instance = VectorStore(persist_directory=persist_directory)
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
    """Get current memory usage in MB as a float"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    return memory_mb 