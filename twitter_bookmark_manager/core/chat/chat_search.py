import logging
from typing import List, Dict, Any, Optional
from database.db import get_db_session, get_vector_store
from database.models import Bookmark, Category
from sentence_transformers import SentenceTransformer
from ..search import BookmarkSearch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBookmarkSearch:
    """
    A specialized search class for the chat system that provides more lenient
    matching and better context handling for conversational interactions.
    """
    def __init__(self):
        """Initialize search with vector store and SQL capabilities"""
        try:
            self.main_search = BookmarkSearch()  # Initialize main search engine
            self.embedding_model = None  # Will load when needed
            self.vector_store = get_vector_store()
            with get_db_session() as session:
                self.total_tweets = session.query(Bookmark).count()
            logger.info(f"✓ Chat search initialized with {self.total_tweets} bookmarks")
        except Exception as e:
            logger.error(f"❌ Error initializing chat search: {e}")
            raise

    def _ensure_model_loaded(self):
        """Load the embedding model if not already loaded"""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("✓ Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading embedding model: {e}")
                raise

    def search(self, 
              query: str,
              context: Optional[Dict] = None,
              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a context-aware search using both vector and SQL capabilities.
        """
        try:
            logger.info(f"🔍 Chat searching for: '{query}' with {self.total_tweets} total tweets")
            
            # Get results from both search methods
            vector_results = self._vector_search(query, limit=limit*2)  # Get more results for better filtering
            sql_results = self._sql_search(query, limit=limit*2)
            
            # Combine and deduplicate results
            all_results = self._merge_results(vector_results, sql_results, context)
            
            # Sort by combined score and limit
            all_results.sort(key=lambda x: x['score'], reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Chat search error: {e}")
            return []

    def _format_bookmark(self, bookmark, score: float = 0.5) -> Dict[str, Any]:
        """Safely format a bookmark with fallbacks for missing attributes"""
        try:
            # Construct tweet URL if missing
            tweet_url = getattr(bookmark, 'url', None)
            if not tweet_url and hasattr(bookmark, 'tweet_id'):
                tweet_url = f"https://twitter.com/i/web/status/{bookmark.tweet_id}"
            
            return {
                'id': str(bookmark.id),
                'text': bookmark.text,
                'author': f"@{bookmark.author_username}" if hasattr(bookmark, 'author_username') else "Unknown",
                'categories': [c.name for c in bookmark.categories] if hasattr(bookmark, 'categories') else [],
                'score': score,
                'tweet_url': tweet_url or f"https://twitter.com/i/web/status/{bookmark.id}",
                'created_at': bookmark.created_at if hasattr(bookmark, 'created_at') else None
            }
        except Exception as e:
            logger.error(f"Error formatting bookmark {bookmark.id}: {e}")
            return None

    def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Enhanced vector-based semantic search"""
        try:
            self._ensure_model_loaded()
            
            # Generate query embedding using the loaded model
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            logger.info(f"Generated embedding for query: '{query}'")
            
            # Perform the vector search
            vector_results = self.vector_store.search(
                query_embedding=query_embedding.tolist(),
                limit=limit
            )
            
            # Convert vector results to common format
            formatted_results = []
            with get_db_session() as session:
                for result in vector_results:
                    try:
                        bookmark = session.query(Bookmark).filter_by(id=result['bookmark_id']).first()
                        if bookmark:
                            formatted = self._format_bookmark(
                                bookmark, 
                                score=result.get('score', 0.5)
                            )
                            if formatted:
                                formatted_results.append(formatted)
                    except Exception as e:
                        logger.warning(f"Error processing vector result {result.get('bookmark_id')}: {e}")
                        continue
            
            logger.info(f"Vector search found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def _sql_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform SQL-based keyword search with consistent formatting"""
        try:
            sql_results = self.main_search.search(query=query, limit=limit)
            formatted_results = []
            
            for result in sql_results:
                try:
                    with get_db_session() as session:
                        bookmark = session.query(Bookmark).get(result['id'])
                        if bookmark:
                            formatted = self._format_bookmark(
                                bookmark,
                                score=result.get('score', 0.5)
                            )
                            if formatted:
                                formatted_results.append(formatted)
                except Exception as e:
                    logger.warning(f"Error processing SQL result {result.get('id')}: {e}")
                    continue
            
            logger.info(f"SQL search found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.warning(f"SQL search failed: {e}")
            return []

    def _merge_results(self, 
                      vector_results: List[Dict[str, Any]], 
                      sql_results: List[Dict[str, Any]],
                      context: Optional[Dict]) -> List[Dict[str, Any]]:
        """Merge and score results from both search methods"""
        seen_ids = set()
        merged_results = []

        # Process vector results first (typically more semantically relevant)
        for result in vector_results:
            if result['id'] not in seen_ids:
                result['score'] = result.get('score', 0.5) * 1.2  # Boost vector results slightly
                if context:
                    result['score'] *= (1 + self._calculate_context_boost(result, context))
                merged_results.append(result)
                seen_ids.add(result['id'])

        # Add SQL results
        for result in sql_results:
            if result['id'] not in seen_ids:
                # Calculate base score for SQL results
                result['score'] = 0.5  # Base score
                if context:
                    result['score'] *= (1 + self._calculate_context_boost(result, context))
                merged_results.append(result)
                seen_ids.add(result['id'])

        return merged_results

    def _calculate_context_boost(self, result: Dict[str, Any], context: Dict) -> float:
        """Calculate context-based score boost"""
        boost = 0.0
        
        # Boost if matches current topic
        if context.get('topic'):
            if any(word.lower() in result['text'].lower() 
                  for word in context['topic'].split()):
                boost += 0.3

        # Boost if matches recent categories
        if context.get('recent_categories'):
            result_categories = set(cat.lower() for cat in result.get('categories', []))
            if any(cat.lower() in result_categories 
                  for cat in context['recent_categories']):
                boost += 0.2

        return min(boost, 1.0)  # Cap boost at 1.0

    def get_categories(self) -> List[str]:
        """Get all available categories"""
        try:
            with get_db_session() as session:
                categories = session.query(Category.name).all()
                return [cat[0] for cat in categories]
        except Exception as e:
            logger.error(f"❌ Error getting categories: {e}")
            return [] 