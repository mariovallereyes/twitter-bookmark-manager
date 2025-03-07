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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

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
            # Generate embedding
            embedding = self.model.encode(text)
            
            # Create point with metadata
            point = PointStruct(
                id=bookmark_id,
                vector=embedding.tolist(),
                payload={
                    "bookmark_id": bookmark_id,
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
            
    def rebuild_user_vectors(self, user_id, bookmarks=None):
        """
        Rebuild vector embeddings for a specific user.
        
        Args:
            user_id (str): User ID to rebuild vectors for
            bookmarks (list, optional): List of bookmark dictionaries. If None, will fetch from database.
            
        Returns:
            tuple: (success_count, error_count)
        """
        try:
            import gc
            import psutil
            import os
            
            # Log memory usage
            process = psutil.Process(os.getpid())
            self.logger.info(f"Starting vector rebuild - Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            self.logger.info(f"Starting vector rebuild for user {user_id}")
            
            # Track counts
            success_count = 0
            error_count = 0
            
            # If no bookmarks provided, fetch them from database
            if not bookmarks:
                self.logger.warning(f"No bookmarks provided for user {user_id}, fetching from database...")
                try:
                    from .search_final_multi_user import BookmarkSearchMultiUser
                    search = BookmarkSearchMultiUser()
                    bookmarks = search.get_all_bookmarks_for_user(user_id)
                    self.logger.info(f"Fetched {len(bookmarks)} bookmarks from database for user {user_id}")
                    
                    # Log memory after fetch
                    self.logger.info(f"After fetching bookmarks - Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                    
                except Exception as e:
                    self.logger.error(f"Error fetching bookmarks from database: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    return 0, 1
            
            if not bookmarks:
                self.logger.error(f"No bookmarks found for user {user_id}, cannot rebuild vectors")
                return 0, 1
            
            total_bookmarks = len(bookmarks)
            self.logger.info(f"Processing {total_bookmarks} bookmarks for user {user_id}")
            
            # Delete existing vectors for this user
            try:
                self.logger.info(f"Deleting existing vectors for user {user_id}")
                deleted = self._delete_user_vectors(user_id)
                self.logger.info(f"Deleted existing vectors: {deleted}")
            except Exception as e:
                self.logger.error(f"Error deleting existing vectors: {str(e)}")
                # Continue anyway
            
            # Force garbage collection
            gc.collect()
            self.logger.info(f"After deletion - Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            # Prepare batches - use smaller batches to avoid memory issues
            batch_size = 25  # Reduced batch size from 100
            bookmarks_batched = [bookmarks[i:i + batch_size] for i in range(0, len(bookmarks), batch_size)]
            
            # Process each batch
            for batch_idx, batch in enumerate(bookmarks_batched):
                self.logger.info(f"Processing batch {batch_idx+1}/{len(bookmarks_batched)} ({len(batch)} bookmarks)")
                
                try:
                    # Process each bookmark in batch
                    batch_success = 0
                    batch_errors = 0
                    
                    for bookmark in batch:
                        try:
                            # Get embedding-ready text
                            bookmark_id = bookmark.get('id')
                            if not bookmark_id:
                                self.logger.error(f"Missing ID in bookmark: {bookmark}")
                                error_count += 1
                                batch_errors += 1
                                continue
                                
                            # Combine text for embedding
                            text = bookmark.get('text', '')
                            content = bookmark.get('tweet_content', '')
                            author = bookmark.get('author', '')
                            
                            # Ensure we have at least some text
                            combined_text = ' '.join([t for t in [text, content, author] if t])
                            if not combined_text.strip():
                                self.logger.error(f"Empty text for bookmark {bookmark_id}")
                                error_count += 1
                                batch_errors += 1
                                continue
                            
                            # Create embedding and store in vector DB
                            vector_id = f"{user_id}:{bookmark_id}"
                            self.add_vector(
                                id=vector_id,
                                text=combined_text,
                                payload={
                                    "bookmark_id": bookmark_id,
                                    "user_id": user_id
                                }
                            )
                            success_count += 1
                            batch_success += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error processing bookmark {bookmark.get('id', 'unknown')}: {str(e)}")
                            error_count += 1
                            batch_errors += 1
                    
                    # Log batch results
                    self.logger.info(f"Batch {batch_idx+1} completed: {batch_success} successful, {batch_errors} errors")
                    
                    # Log memory usage after batch
                    self.logger.info(f"After batch {batch_idx+1} - Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                    
                    # Force garbage collection between batches
                    gc.collect()
                    
                    # Ensure we don't overwhelm the system - small sleep between batches
                    time.sleep(0.1)
                    
                except Exception as batch_error:
                    self.logger.error(f"Error processing batch {batch_idx+1}: {str(batch_error)}")
                    self.logger.error(traceback.format_exc())
                    error_count += len(batch)
            
            self.logger.info(f"Vector rebuild completed for user {user_id}: {success_count} successful, {error_count} errors")
            
            # Final memory usage
            self.logger.info(f"Completed rebuild - Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            return success_count, error_count
            
        except Exception as e:
            self.logger.error(f"Error rebuilding vectors for user {user_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 0, 1
            
    def _delete_user_vectors(self, user_id: int) -> bool:
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