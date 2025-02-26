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
    def __init__(self, persist_directory: str = None):
        """Initialize Qdrant client in memory mode to avoid lock issues"""
        try:
            # Import config here to avoid circular imports
            from deployment.pythonanywhere.postgres.config import VECTOR_STORE_CONFIG
            
            # Store path for later use (may be used for backup/restore)
            self.vector_db_path = VECTOR_STORE_CONFIG['persist_directory']
            
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
               query_embedding: List[float] = None,
               query: str = None,
               limit: int = 10,
               n_results: int = None) -> List[Dict[str, Any]]:
        """Search for similar bookmarks"""
        try:
            # Handle both embedding and text-based search
            if query_embedding is None and query is not None:
                query_embedding = self.model.encode(query).tolist()
            elif query_embedding is None and query is None:
                raise ValueError("Either query_embedding or query must be provided")

            # Use n_results if provided (backward compatibility), otherwise use limit
            final_limit = n_results if n_results is not None else limit
            
            # Perform the search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=final_limit
            )
            
            # Process results
            processed_results = []
            for result in results:
                processed_results.append({
                    'bookmark_id': result.payload.get('original_id'),  # Use original ID
                    'score': result.score,
                    'metadata': {k: v for k, v in result.payload.items() if k not in ['text', 'original_id']},
                    'text': result.payload.get('text')
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Search error in VectorStore: {e}")
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