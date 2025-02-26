import logging
from typing import List, Dict, Any, Optional
from .db_pa import get_session, get_vector_store
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
            self.total_tweets = self._get_total_tweets()  # Use new method
            logger.info(f"✓ Search initialized successfully with {self.total_tweets} bookmarks")
        except Exception as e:
            logger.error(f"❌ Error initializing search: {e}")
            raise

    def _get_total_tweets(self):
        """Get current total number of tweets from database"""
        try:
            with get_session() as session:
                return session.query(Bookmark).count()
        except Exception as e:
            logger.error(f"❌ Error getting total tweets: {e}")
            return 0

    def _ensure_model_loaded(self):
        """Load the embedding model if not already loaded"""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def _batched_vector_search(self, query, max_results=None, batch_size=100):
        """Perform vector search in batches"""
        all_results = []
        excluded_ids = set()
        
        if max_results is None:
            max_results = self.total_tweets
            
        max_iterations = 10
        
        logger.info(f"🔎 Starting batched vector search for '{query}' (max results: {max_results})")
        
        for iteration in range(max_iterations):
            try:
                if len(all_results) >= max_results:
                    logger.info(f"✓ Reached max results limit ({max_results})")
                    break
                    
                remaining = max_results - len(all_results)
                current_batch_size = min(batch_size, remaining)
                
                logger.info(f"📊 Batch {iteration+1}: Fetching {current_batch_size} results")
                
                batch_results = self.vector_store.search_with_exclusions(
                    query=query,
                    limit=current_batch_size,
                    excluded_ids=list(excluded_ids)
                )
                
                if not batch_results:
                    logger.info(f"✓ No more results found")
                    break
                
                all_results.extend(batch_results)
                
                for result in batch_results:
                    excluded_ids.add(result['bookmark_id'])
                    
                if len(batch_results) < current_batch_size:
                    logger.info(f"✓ All available results found")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error in batch {iteration+1}: {e}")
                break
        
        logger.info(f"🔍 Found {len(all_results)} total results for '{query}'")
        return all_results

    def get_total_tweets(self):
        """Return total number of tweets in the database with refresh"""
        self.total_tweets = self._get_total_tweets()
        return self.total_tweets

    def search(self, 
               query: str = "", 
               categories: List[str] = None,
               limit: int = None) -> List[Dict[str, Any]]:
        """Search bookmarks using semantic similarity with optional category filtering"""
        try:
            self.total_tweets = self._get_total_tweets()
            
            if not query and not categories:
                return self.get_all_bookmarks(limit=limit if limit else 1000)
            
            if not query and categories:
                with get_session() as session:
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
            
            if query:
                logger.info(f"🔍 Searching for: '{query}'")
                
                try:
                    self._ensure_model_loaded()
                    max_search_results = self.total_tweets
                    
                    try:
                        vector_results = self._batched_vector_search(
                            query=query, 
                            max_results=max_search_results
                        )
                    except Exception as batch_err:
                        logger.warning(f"Batched search failed: {batch_err}")
                        vector_results = self.vector_store.search(
                            query=query,
                            n_results=min(self.total_tweets, 100)
                        )
                    
                    processed_results = []
                    seen_ids = set()
                    query_words = set(word.lower() for word in query.split())
                    
                    for result in vector_results:
                        bookmark_id = result['bookmark_id']
                        if bookmark_id in seen_ids:
                            continue
                        
                        with get_session() as session:
                            bookmark = session.query(Bookmark).get(bookmark_id)
                            if bookmark:
                                bookmark_categories = [cat.name for cat in bookmark.categories]
                                
                                if categories and not any(cat in bookmark_categories for cat in categories):
                                    continue
                                
                                distance = result.get('distance', 1.0)
                                text_lower = bookmark.text.lower()
                                
                                exact_match = query.lower() in text_lower
                                word_match_ratio = sum(1 for word in query_words if word in text_lower) / len(query_words)
                                
                                if (exact_match or 
                                    distance < 0.90 or
                                    (distance < 0.95 and word_match_ratio > 0.3) or
                                    word_match_ratio > 0.5):
                                    
                                    semantic_score = 1.0 - distance
                                    word_score = word_match_ratio
                                    combined_score = (semantic_score * 0.6) + (word_score * 0.4)
                                    
                                    processed_results.append({
                                        'id': bookmark.id,
                                        'text': bookmark.text,
                                        'author': f"@{bookmark.author_username}",
                                        'categories': bookmark_categories,
                                        'score': combined_score,
                                        'created_at': bookmark.created_at
                                    })
                                    seen_ids.add(bookmark_id)
                    
                    logger.info(f"📊 Found {len(processed_results)} results")
                    processed_results.sort(key=lambda x: x['score'], reverse=True)
                    return processed_results
                    
                except Exception as e:
                    logger.warning(f"Vector search failed, using SQL: {e}")
                    with get_session() as session:
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
                        
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            raise

    def get_all_bookmarks(self, limit=1000):
        """Get all bookmarks with optional limit"""
        try:
            with get_session() as session:
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
            logger.error(f"❌ Error getting all bookmarks: {e}")
            raise 