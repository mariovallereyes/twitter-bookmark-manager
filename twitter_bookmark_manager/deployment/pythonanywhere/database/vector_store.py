import chromadb
from typing import List, Dict, Any
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ChromaStore:
    def __init__(self, persist_directory: str = "./vector_db"):
        """Initialize Chroma client with proper configuration"""
        try:
            # Ensure persist directory exists
            persist_directory = os.path.abspath(persist_directory)
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Simple initialization for ChromaDB 0.3.21
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name="bookmarks",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embedding model with local download
            from sentence_transformers import SentenceTransformer
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            try:
                # Try to load from cache first
                cache_dir = os.path.join(persist_directory, 'model_cache')
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            except Exception as model_error:
                logger.warning(f"Error loading model from cache: {model_error}")
                # Fallback to direct loading
                self.model = SentenceTransformer(model_name)
            
            logger.info(f"ChromaStore initialized with {self.collection.count()} embeddings")
        except Exception as e:
            logger.error(f"Error initializing ChromaStore: {e}")
            raise

    def add_bookmark(self, 
                    bookmark_id: str,
                    text: str,
                    metadata: Dict[str, Any] = None):
        """Add a single bookmark to the vector store"""
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[bookmark_id]
            )
        except Exception as e:
            logger.error(f"Error adding bookmark to vector store: {e}")
            raise

    def search(self, 
               query_embedding: List[float] = None,
               query: str = None,
               limit: int = 10,
               n_results: int = None,  # For backward compatibility
               **kwargs) -> List[Dict[str, Any]]:
        """
        Enhanced search method supporting both raw embeddings and text queries.
        Maintains backward compatibility with n_results parameter.
        Returns properly formatted results with scores.
        """
        try:
            # Handle both embedding and text-based search
            if query_embedding is None and query is not None:
                query_embedding = self.model.encode(query).tolist()
            elif query_embedding is None and query is None:
                raise ValueError("Either query_embedding or query must be provided")

            # Use n_results if provided (backward compatibility), otherwise use limit
            final_limit = n_results if n_results is not None else limit
            logger.info(f"Performing vector search with limit={final_limit}")
            
            # Perform the search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=final_limit,
                include=['metadatas', 'distances', 'documents']
            )
            
            # Process and return results
            processed_results = []
            if results and 'ids' in results and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    try:
                        distance = results['distances'][0][i] if 'distances' in results else 0.0
                        processed_results.append({
                            'bookmark_id': str(results['ids'][0][i]),  # Ensure string ID
                            'score': 1.0 - distance,  # Convert distance to similarity score
                            'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                            'text': results['documents'][0][i] if 'documents' in results else None
                        })
                    except Exception as e:
                        logger.warning(f"Error processing result {i}: {e}")
                        continue
            
            logger.info(f"Vector search found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Search error in ChromaStore: {e}")
            return []

    def delete_bookmark(self, bookmark_id: str):
        """Delete a bookmark from the vector store"""
        try:
            self.collection.delete(ids=[bookmark_id])
        except Exception as e:
            logger.error(f"Error deleting bookmark from vector store: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed information about the collection"""
        try:
            return {
                'count': self.collection.count(),
                'name': self.collection.name,
                'metadata': self.collection.get()
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {'error': str(e)}

    def clear_cache(self) -> None:
        """Clear any internal caches"""
        try:
            if hasattr(self.collection, '_client'):
                self.collection._client.clear_cache()
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def query_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar bookmarks"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results to match what rag.py expects
            formatted_results = []
            if results and 'ids' in results:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'bookmark_id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying similar bookmarks: {e}")
            return [] 