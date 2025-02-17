import logging
from typing import List, Dict, Any, Tuple
from database.db import get_db_session, get_vector_store
from database.models import Bookmark
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookmarkDeduplicator:
    def __init__(self, similarity_threshold: float = 0.95):
        """Initialize the deduplicator
        
        Args:
            similarity_threshold: Cosine similarity threshold (0-1) for considering bookmarks as duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.vector_store = get_vector_store()
        logger.info(f"BookmarkDeduplicator initialized with threshold {similarity_threshold}")

    def find_potential_duplicates(self, bookmark_id: str) -> List[Dict[str, Any]]:
        """Find bookmarks that are potentially duplicates of the given bookmark ID
        
        Args:
            bookmark_id: ID of the bookmark to compare against
        
        Returns:
            List of potential duplicates with their similarity scores
        """
        try:
            # Get the bookmark's embedding from vector store
            result = self.vector_store.collection.get(
                ids=[bookmark_id],
                include=['embeddings', 'metadatas']
            )
            
            if not result or 'embeddings' not in result or len(result['embeddings']) == 0:
                raise ValueError(f"No embedding found for bookmark {bookmark_id}")
            
            embedding = result['embeddings'][0]
            
            # Query for similar bookmarks
            similar = self.vector_store.query_similar(
                embedding,
                n_results=10
            )
            
            # Filter by similarity threshold and exclude self
            potential_duplicates = []
            for i, ids in enumerate(similar['ids']):
                distances = similar['distances'][i]
                metadatas = similar['metadatas'][i]
                
                for j, (id_, distance) in enumerate(zip(ids, distances)):
                    # Convert distance to similarity (ChromaDB returns L2 distance)
                    similarity = 1.0 / (1.0 + distance)  # Convert L2 distance to similarity score
                    
                    if (similarity >= self.similarity_threshold and 
                        id_ != bookmark_id):  # Exclude self-comparison
                        potential_duplicates.append({
                            'bookmark_id': id_,
                            'similarity': similarity,
                            'metadata': metadatas[j],
                            'requires_confirmation': True
                        })
            
            return potential_duplicates
            
        except Exception as e:
            logger.error(f"Error finding potential duplicates for {bookmark_id}: {e}")
            raise

    def mark_as_duplicate(self, original_id: str, duplicate_id: str, 
                         similarity: float, user_confirmed: bool = False) -> Dict[str, Any]:
        """Mark a bookmark as a duplicate of another
        
        Args:
            original_id: ID of the original bookmark
            duplicate_id: ID of the duplicate bookmark
            similarity: Similarity score between the bookmarks
            user_confirmed: Whether the duplicate was confirmed by user
        """
        try:
            with get_db_session() as session:
                duplicate = session.query(Bookmark).get(duplicate_id)
                if not duplicate:
                    raise ValueError(f"Bookmark {duplicate_id} not found")
                
                # Update duplicate status
                duplicate.is_duplicate = True
                duplicate.original_bookmark_id = original_id
                duplicate.duplicate_similarity = similarity
                duplicate.user_confirmed = user_confirmed
                duplicate.processed_at = datetime.utcnow()
                
                return {
                    'original_id': original_id,
                    'duplicate_id': duplicate_id,
                    'similarity': similarity,
                    'user_confirmed': user_confirmed,
                    'status': 'marked_as_duplicate'
                }
                
        except Exception as e:
            logger.error(f"Error marking duplicate {duplicate_id} of {original_id}: {e}")
            raise

    def mark_as_not_duplicate(self, original_id: str, potential_duplicate_id: str) -> Dict[str, Any]:
        """Mark a bookmark as not being a duplicate (user rejected)
        
        Args:
            original_id: ID of the original bookmark
            potential_duplicate_id: ID of the potential duplicate bookmark
        """
        try:
            with get_db_session() as session:
                bookmark = session.query(Bookmark).get(potential_duplicate_id)
                if not bookmark:
                    raise ValueError(f"Bookmark {potential_duplicate_id} not found")
                
                # Update to indicate user rejected duplicate status
                bookmark.is_duplicate = False
                bookmark.original_bookmark_id = None
                bookmark.duplicate_similarity = None
                bookmark.user_confirmed = True
                bookmark.processed_at = datetime.utcnow()
                
                return {
                    'original_id': original_id,
                    'potential_duplicate_id': potential_duplicate_id,
                    'status': 'marked_as_not_duplicate'
                }
                
        except Exception as e:
            logger.error(f"Error marking non-duplicate {potential_duplicate_id}: {e}")
            raise

    def process_bookmark(self, bookmark_id: str) -> Dict[str, Any]:
        """Process a single bookmark to find potential duplicates
        
        Args:
            bookmark_id: ID of the bookmark to process
        """
        try:
            # Find potential duplicates
            potential_duplicates = self.find_potential_duplicates(bookmark_id)
            
            return {
                'bookmark_id': bookmark_id,
                'potential_duplicates_found': len(potential_duplicates),
                'potential_duplicates': potential_duplicates,
                'requires_user_confirmation': True
            }
            
        except Exception as e:
            logger.error(f"Error processing bookmark {bookmark_id}: {e}")
            raise

    def confirm_duplicate(self, original_id: str, duplicate_id: str, 
                         similarity: float, user_confirmed: bool) -> Dict[str, Any]:
        """Handle user confirmation of duplicate status
        
        Args:
            original_id: ID of the original bookmark
            duplicate_id: ID of the potential duplicate
            similarity: Similarity score between the bookmarks
            user_confirmed: Whether user confirmed it's a duplicate
        """
        try:
            if user_confirmed:
                return self.mark_as_duplicate(original_id, duplicate_id, similarity, True)
            else:
                return self.mark_as_not_duplicate(original_id, duplicate_id)
        except Exception as e:
            logger.error(f"Error handling duplicate confirmation: {e}")
            raise

    def batch_process_bookmarks(self, bookmark_ids: List[str]) -> List[Dict[str, Any]]:
        """Process multiple bookmarks to find potential duplicates
        
        Args:
            bookmark_ids: List of bookmark IDs to process
        """
        results = []
        for bookmark_id in bookmark_ids:
            try:
                result = self.process_bookmark(bookmark_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch processing for bookmark {bookmark_id}: {e}")
                results.append({
                    'bookmark_id': bookmark_id,
                    'error': str(e)
                })
        return results