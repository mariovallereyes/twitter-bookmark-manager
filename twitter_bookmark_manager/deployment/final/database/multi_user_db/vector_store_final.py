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
        self.model = None
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
        
        # Initialize the embedding model
        try:
            logger.info(f"Initializing SentenceTransformer model for vector embeddings")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info(f"Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        # Initialize Qdrant client
        try:
            logger.info(f"Initializing Qdrant client with persist directory: {self.persist_directory}")
            self.client = QdrantClient(path=self.persist_directory)
            logger.info(f"Qdrant client initialized successfully")
            
            # Create collection if it doesn't exist
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
                    ),
                    optimizers_config=rest.OptimizersConfigDiff(
                        indexing_threshold=0,  # Index immediately
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def add_bookmark(self, bookmark_id: int, text: str, user_id: int) -> bool:
        """
        Add a bookmark to the vector store.
        
        Args:
            bookmark_id: ID of the bookmark
            text: Text content of the bookmark
            user_id: User ID who owns the bookmark
            
        Returns:
            True if successful, False otherwise
        """
        if not self.model or not self.client:
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
            
            # Generate embedding
            embedding = self.model.encode(text)
            
            # Create a stable hash for the ID that combines the user ID and bookmark ID
            point_id = f"{user_id}_{bookmark_id_str}"
            
            # Create point with metadata
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "bookmark_id": bookmark_id_str,
                    "text": text[:1000] if text else "",  # Limit text size in payload
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Upsert point to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Added bookmark {bookmark_id} to vector store for user {user_id}")
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
        if not self.model or not self.client:
            logger.error("Vector store not properly initialized")
            return []
            
        try:
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
            
    def rebuild_user_vectors(self, user_id, batch_size=20):
        """Rebuild vector store for a specific user's bookmarks with enhanced memory management"""
        from twitter_bookmark_manager.deployment.final.database.multi_user_db.db_final import get_bookmarks_for_user
        
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
            
            # Process in smaller batches
            success_count = 0
            error_count = 0
            
            for i in range(0, total, batch_size):
                batch = bookmarks[i:i+batch_size]
                current_batch = i//batch_size + 1
                total_batches = (total+batch_size-1)//batch_size
                
                logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch)} bookmarks)")
                
                # Process each bookmark in the batch with individual error handling
                for bookmark in batch:
                    try:
                        self.add_bookmark(
                            bookmark_id=bookmark.id,
                            user_id=user_id,
                            text=bookmark.text or "",
                            metadata={
                                'author': bookmark.author_username,
                                'created_at': str(bookmark.created_at) if bookmark.created_at else None
                            }
                        )
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error adding bookmark {bookmark.id} to vector store: {str(e)}")
                
                # Force garbage collection after each batch
                gc.collect()
                logger.info(f"Memory after batch {current_batch}/{total_batches}: {self.get_memory_usage()}")
                
                # Log progress
                progress = ((i + len(batch)) / total) * 100
                logger.info(f"Progress: {progress:.1f}% ({i + len(batch)}/{total})")
            
            logger.info(f"Final memory usage: {self.get_memory_usage()}")
            logger.info(f"Vector rebuild completed: {success_count} successes, {error_count} errors")
            return True
            
        except Exception as e:
            logger.error(f"Error during vector rebuild for user {user_id}: {str(e)}")
            return False
            
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
            
    def delete_bookmark(self, bookmark_id: int) -> bool:
        """
        Delete a specific bookmark from the vector store.
        
        Args:
            bookmark_id: ID of the bookmark to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("Vector store client not initialized")
            return False
            
        try:
            # Delete point by ID
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(
                    points=[bookmark_id]
                )
            )
            
            logger.info(f"Deleted bookmark {bookmark_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting bookmark {bookmark_id} from vector store: {str(e)}")
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

# Create a singleton instance
_vector_store_instance = None

def get_vector_store(persist_directory=None):
    """
    Get a singleton instance of the vector store.
    
    Args:
        persist_directory: Optional directory to persist vectors
        
    Returns:
        VectorStore instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        try:
            _vector_store_instance = VectorStore(persist_directory)
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
    
    Args:
        persist_directory: Optional directory to persist vectors
        
    Returns:
        VectorStore instance
    """
    return get_vector_store(persist_directory) 

def get_memory_usage():
    """Get current memory usage as a formatted string"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    return f"{memory_mb:.1f}MB" 