import logging
from typing import List, Dict, Any, Optional
from database.db import get_db_session, get_vector_store
from database.models import Bookmark, Category
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookmarkSearch:
    def __init__(self):
        """Initialize search with vector store"""
        try:
            self.embedding_model = None  # Will load when needed
            self.vector_store = get_vector_store()
            # Get total number of bookmarks from DB session
            with get_db_session() as session:
                self.total_tweets = session.query(Bookmark).count()
            logger.info(f"✓ Search initialized successfully with {self.total_tweets} bookmarks")
        except Exception as e:
            logger.error(f"❌ Error initializing search: {e}")
            raise

    def _ensure_model_loaded(self):
        """Load the embedding model if not already loaded"""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_total_tweets(self):
        """Return total number of tweets in the database"""
        return self.total_tweets

    def search(self, 
               query: str = "", 
               categories: List[str] = None,
               limit: int = None) -> List[Dict[str, Any]]:
        """
        Search bookmarks using semantic similarity with optional category filtering
        """
        try:
            # No criteria provided - return all bookmarks
            if not query and not categories:
                return self.get_all_bookmarks(limit=limit if limit else 1000)
            
            # Category-only search (no query)
            if not query and categories:
                with get_db_session() as session:
                    bookmarks = session.query(Bookmark)\
                        .join(Bookmark.categories)\
                        .filter(Category.name.in_(categories))\
                        .order_by(Bookmark.created_at.desc())\
                        .all()
                    
                    return [{
                        'id': b.id,
                        'text': b.text,
                        'author': f"@{b.author_username}",
                        'categories': [cat.name for cat in b.categories],
                        'created_at': b.created_at
                    } for b in bookmarks]
            
            # Text search with fallback
            if query:
                logger.info(f"🔍 Searching for: '{query}'")
                
                try:
                    # Try vector search first
                    self._ensure_model_loaded()
                    vector_results = self.vector_store.search(
                        query=query,
                        n_results=min(self.total_tweets, 100)  # Limit to 100 max
                    )
                    
                    processed_results = []
                    seen_ids = set()
                    query_words = set(word.lower() for word in query.split())
                    
                    for result in vector_results:
                        bookmark_id = result['bookmark_id']
                        if bookmark_id in seen_ids:
                            continue
                        
                        with get_db_session() as session:
                            bookmark = session.query(Bookmark).get(bookmark_id)
                            if bookmark:
                                bookmark_categories = [cat.name for cat in bookmark.categories]
                                
                                if categories and not any(cat in bookmark_categories for cat in categories):
                                    continue
                                
                                distance = result.get('distance', 1.0)
                                text_lower = bookmark.text.lower()
                                
                                # Enhanced matching criteria with more lenient thresholds
                                exact_match = query.lower() in text_lower
                                word_match_ratio = sum(1 for word in query_words if word in text_lower) / len(query_words)
                                
                                # Much more lenient thresholds
                                if (exact_match or 
                                    distance < 0.90 or  # More lenient semantic similarity
                                    (distance < 0.95 and word_match_ratio > 0.3) or  # More lenient partial match
                                    word_match_ratio > 0.5):  # More lenient word match
                                    
                                    # Debug matching info
                                    logger.debug(f"""
                                    Match found:
                                    - Text: {bookmark.text[:100]}...
                                    - Distance: {distance}
                                    - Word match ratio: {word_match_ratio}
                                    - Exact match: {exact_match}
                                    """)
                                    
                                    # Calculate combined score with adjusted weights
                                    semantic_score = 1.0 - distance
                                    word_score = word_match_ratio
                                    combined_score = (semantic_score * 0.6) + (word_score * 0.4)  # Adjusted weights
                                    
                                    processed_results.append({
                                        'id': bookmark.id,
                                        'text': bookmark.text,
                                        'author': f"@{bookmark.author_username}",
                                        'categories': bookmark_categories,
                                        'score': combined_score,
                                        'created_at': bookmark.created_at
                                    })
                                    seen_ids.add(bookmark_id)
                    
                    # Debug results count
                    logger.info(f"📊 Found {len(processed_results)} results for '{query}'")
                    
                    # Sort by score
                    processed_results.sort(key=lambda x: x['score'], reverse=True)
                    return processed_results
                    
                except Exception as e:
                    logger.warning(f"Vector search failed, falling back to SQL search: {e}")
                    # Fallback to SQL LIKE search
                    with get_db_session() as session:
                        query_filter = f"%{query}%"
                        base_query = session.query(Bookmark)\
                            .filter(Bookmark.text.ilike(query_filter))
                        
                        if categories:
                            base_query = base_query.join(Bookmark.categories)\
                                .filter(Category.name.in_(categories))
                        
                        bookmarks = base_query.order_by(Bookmark.created_at.desc()).all()
                        
                        return [{
                            'id': b.id,
                            'text': b.text,
                            'author': f"@{b.author_username}",
                            'categories': [cat.name for cat in b.categories],
                            'created_at': b.created_at
                        } for b in bookmarks]
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def get_all_bookmarks(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all bookmarks"""
        try:
            with get_db_session() as session:
                total = session.query(Bookmark).count()
                logger.info(f"📚 Fetching {limit} of {total} total bookmarks")
                
                # Add order_by to get latest bookmarks first
                bookmarks = session.query(Bookmark)\
                    .order_by(Bookmark.created_at.desc())\
                    .limit(limit)\
                    .all()
                
                return [{
                    'id': b.id,
                    'text': b.text,
                    'author': f"@{b.author_username}",
                    'categories': [cat.name for cat in b.categories],
                    'created_at': b.created_at
                } for b in bookmarks]
                
        except Exception as e:
            logger.error(f"❌ Error getting bookmarks: {e}")
            return []

    def search_by_category(self, 
                          category: str, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """Get bookmarks by category"""
        try:
            with get_db_session() as session:
                bookmarks = session.query(Bookmark)\
                    .filter(Bookmark.category == category)\
                    .limit(limit)\
                    .all()
                
                return [{
                    'id': b.id,
                    'url': b.url,
                    'category': b.category,
                    'created_at': b.created_at
                } for b in bookmarks]
                
        except Exception as e:
            logger.error(f"❌ Category search error: {e}")
            return []

    def get_categories(self) -> List[str]:
        """Get all available categories"""
        try:
            with get_db_session() as session:
                categories = session.query(Category.name).all()
                return [cat[0] for cat in categories]
        except Exception as e:
            logger.error(f"❌ Error getting categories: {e}")
            return []

    def search_by_user(self, username: str) -> List[Dict[str, Any]]:
        """
        Search bookmarks by specific Twitter username
        :param username: Twitter username (with or without @)
        """
        try:
            with get_db_session() as session:
                # Normalize username by removing @ if present
                username = username.lower().strip('@')
                
                # Query bookmarks by author
                bookmarks = session.query(Bookmark)\
                    .filter(Bookmark.author_username.ilike(username))\
                    .all()
                
                logger.info(f"🔍 Found {len(bookmarks)} bookmarks from @{username}")
                
                return [{
                    'id': b.id,
                    'text': b.text,
                    'author': f"@{b.author_username}",
                    'categories': [cat.name for cat in b.categories],
                    'created_at': b.created_at
                } for b in bookmarks]
                
        except Exception as e:
            logger.error(f"❌ User search error: {e}")
            return []