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
            
    def rebuild_user_vectors(self, user_id, bookmarks=None):
        """
        Rebuild vector database for a user with given bookmarks.
        If bookmarks is None, fetch bookmarks from the database.
        
        Args:
            user_id: User ID to rebuild vectors for
            bookmarks: List of bookmark dicts with 'id', 'text', 'tweet_content' fields. 
                      If None, bookmarks will be fetched from DB.
        
        Returns:
            dict: Result with success flag and details
        """
        try:
            logger.info(f"Starting vector rebuild for user {user_id}")
            logger.info(f"Initial memory usage: {get_memory_usage()}")
            
            # Check if bookmarks were provided
            if bookmarks is None or len(bookmarks) == 0:
                logger.warning(f"No bookmarks provided for user {user_id}, attempting to fetch from database")
                
                try:
                    # Import bookmark search class
                    from database.multi_user_db.search_final_multi_user import BookmarkSearchMultiUser
                    
                    # Initialize search object
                    search_obj = BookmarkSearchMultiUser()
                    
                    # Fetch bookmarks from database
                    bookmarks = search_obj.get_all_bookmarks_for_user(user_id)
                    logger.info(f"Fetched {len(bookmarks)} bookmarks from database for user {user_id}")
                    logger.info(f"Memory after fetching bookmarks: {get_memory_usage()}")
                    
                    if not bookmarks:
                        logger.warning(f"No bookmarks found in database for user {user_id}")
                        return {
                            'success': False,
                            'error': 'No bookmarks found for user',
                            'user_id': user_id,
                            'bookmark_count': 0
                        }
                    
                except Exception as fetch_error:
                    logger.error(f"Failed to fetch bookmarks for user {user_id}: {str(fetch_error)}")
                    logger.error(traceback.format_exc())
                    return {
                        'success': False,
                        'error': f"Failed to fetch bookmarks: {str(fetch_error)}",
                        'user_id': user_id
                    }
            
            # Track metrics
            start_time = time.time()
            logger.info(f"Processing {len(bookmarks)} bookmarks for vector rebuild")
            
            # Prepare texts and metadata for embedding
            texts = []
            bookmark_ids = []
            combined_texts = []
            
            # Log memory usage before processing
            logger.info(f"Memory before processing bookmarks: {get_memory_usage()}")
            
            for bookmark in bookmarks:
                # Combine text and tweet content for better embeddings
                # Handle possible formats based on the data source
                text = bookmark.get('text', '')
                tweet_content = bookmark.get('tweet_content', '')
                bookmark_id = bookmark.get('id', bookmark.get('bookmark_id', ''))
                
                if not bookmark_id:
                    logger.warning(f"Skipping bookmark without ID: {bookmark}")
                    continue
                    
                # Create combined text
                combined_text = text
                if tweet_content and tweet_content != text:
                    # Try to parse tweet_content if it's a JSON string
                    if isinstance(tweet_content, str) and tweet_content.startswith('{'):
                        try:
                            tweet_json = json.loads(tweet_content)
                            if 'full_text' in tweet_json:
                                combined_text = tweet_json.get('full_text', text)
                            elif 'text' in tweet_json:
                                combined_text = tweet_json.get('text', text)
                        except:
                            # If JSON parsing fails, just use the text field
                            pass
                    else:
                        combined_text = f"{text}\n{tweet_content}"
                    
                if not combined_text.strip():
                    logger.warning(f"Skipping bookmark {bookmark_id} with empty text")
                    continue
                    
                # Limit text length to avoid memory issues
                if len(combined_text) > 10000:
                    logger.warning(f"Truncating very long bookmark text ({len(combined_text)} chars) for bookmark {bookmark_id}")
                    combined_text = combined_text[:10000]
                
                # Add to lists for batch processing
                texts.append(text)
                bookmark_ids.append(bookmark_id)
                combined_texts.append(combined_text)
            
            if not texts:
                logger.warning(f"No valid bookmark texts to process for user {user_id}")
                return {
                    'success': False,
                    'error': 'No valid bookmark texts found',
                    'user_id': user_id
                }
                
            # Delete existing vectors for this user
            try:
                logger.info(f"Memory before deleting old vectors: {get_memory_usage()}")
                self._delete_vectors_for_user(user_id)
                logger.info(f"Memory after deleting old vectors: {get_memory_usage()}")
            except Exception as delete_error:
                logger.error(f"Error deleting existing vectors for user {user_id}: {str(delete_error)}")
                # Continue with the rebuild process
            
            # Add vectors in batches
            batch_size = 50  # Process in smaller batches to save memory
            total_batches = math.ceil(len(texts) / batch_size)
            successful = 0
            failed = 0
            
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_texts = combined_texts[i:batch_end]
                batch_ids = bookmark_ids[i:batch_end]
                
                try:
                    logger.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch_texts)} bookmarks)")
                    logger.info(f"Memory before batch {i//batch_size + 1}: {get_memory_usage()}")
                    
                    # Process each bookmark in the batch individually
                    for j, (text, bookmark_id) in enumerate(zip(batch_texts, batch_ids)):
                        try:
                            # Use the existing add_bookmark method directly
                            success = self.add_bookmark(
                                bookmark_id=bookmark_id,
                                text=text,
                                user_id=user_id
                            )
                            
                            if success:
                                successful += 1
                            else:
                                logger.warning(f"Failed to add bookmark {bookmark_id} to vector store")
                                failed += 1
                                
                            # Log progress within batch
                            if (j + 1) % 10 == 0:
                                logger.info(f"Processed {j + 1}/{len(batch_texts)} in current batch")
                                
                        except Exception as item_error:
                            logger.error(f"Error processing bookmark {bookmark_id}: {str(item_error)}")
                            failed += 1
                    
                    logger.info(f"Batch {i//batch_size + 1}/{total_batches} processed: {successful} successful, {failed} failed")
                    logger.info(f"Memory after batch {i//batch_size + 1}: {get_memory_usage()}")
                    
                    # Force garbage collection after each batch to free memory
                    import gc
                    gc.collect()
                    logger.info(f"Memory after GC: {get_memory_usage()}")
                    
                except Exception as batch_error:
                    logger.error(f"Error processing batch {i//batch_size + 1}/{total_batches}: {str(batch_error)}")
                    logger.error(traceback.format_exc())
                    failed += len(batch_texts)
            
            # Get processing time
            processing_time = time.time() - start_time
            
            logger.info(f"Vector rebuild completed for user {user_id} in {processing_time:.2f} seconds: {successful} successful, {failed} failed")
            logger.info(f"Final memory usage: {get_memory_usage()}")
            
            return {
                'success': successful > 0,
                'message': f"Processed {successful} bookmarks, {failed} failed",
                'user_id': user_id,
                'details': {
                    'successful': successful,
                    'failed': failed,
                    'total_processed': successful + failed,
                    'processing_time_seconds': processing_time,
                    'memory_usage': get_memory_usage()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in rebuild_user_vectors for user {user_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'user_id': user_id
            }
            
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