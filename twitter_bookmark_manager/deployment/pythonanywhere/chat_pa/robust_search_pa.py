"""
Robust search handler that combines vector and SQL search capabilities
with error handling and retries.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import random

from .persistent_store_pa import PersistentVectorStore
from .search_config_pa import get_search_config

# Set up logging
logger = logging.getLogger(__name__)

class RobustSearchHandler:
    """
    Provides robust search capabilities by combining vector and SQL searches
    with error handling and retries.
    """
    
    def __init__(self):
        """Initialize the search handler"""
        self._vector_store = None
        self._sql_initialized = False
        self._pg_conn = None
        self._last_error = None
        self._initialized = False
        
        # Create an initialization lock
        self._init_lock = asyncio.Lock()
        
        logger.info("Initialized RobustSearchHandler")
    
    def ensure_initialized(self) -> None:
        """Ensure that the search handler is initialized"""
        if self._initialized:
            return
        
        try:
            # Initialize vector store
            if not self._vector_store:
                self._vector_store = PersistentVectorStore()
                self._vector_store.ensure_initialized()
            
            # Initialize SQL connection
            self._init_sql_connection()
            
            self._initialized = True
            logger.info("RobustSearchHandler initialized successfully")
            
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to initialize RobustSearchHandler: {e}")
            raise
    
    async def initialize_async(self) -> None:
        """Initialize asynchronous components"""
        if not self._initialized:
            # First ensure basic initialization is done
            self.ensure_initialized()
        
        async with self._init_lock:
            # Initialize any async components in the vector store
            if self._vector_store:
                try:
                    await self._vector_store.initialize_async()
                    logger.info("Vector store async components initialized")
                except Exception as e:
                    logger.error(f"Error initializing vector store async components: {e}")
    
    def _init_sql_connection(self) -> None:
        """Initialize SQL connection for keyword search"""
        try:
            # Import psycopg2 only when needed
            import psycopg2
            import psycopg2.extras
            
            # Get database connection configuration
            # We assume the database configuration is available in the current environment
            import os
            
            # Use the existing connection if available from a central connection pool
            try:
                from twitter_bookmark_manager.deployment.pythonanywhere.database import get_db_connection
                self._pg_conn = get_db_connection()
                logger.info("Using existing database connection from pool")
            except ImportError:
                # If not available, create a new connection
                self._pg_conn = psycopg2.connect(
                    dbname=os.environ.get("PGDATABASE", "bookmarks"),
                    user=os.environ.get("PGUSER", "postgres"),
                    password=os.environ.get("PGPASSWORD", ""),
                    host=os.environ.get("PGHOST", "localhost"),
                    port=os.environ.get("PGPORT", "5432")
                )
                logger.info("Created new database connection")
            
            self._sql_initialized = True
            
        except Exception as e:
            self._sql_initialized = False
            self._last_error = str(e)
            logger.error(f"Failed to initialize SQL connection: {e}")
    
    async def _execute_vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Execute vector search with retry logic
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not self._vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Get the embedding and perform search
            results = await asyncio.to_thread(
                self._vector_store.search,
                query=query,
                limit=limit
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    async def _execute_sql_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Execute SQL search with retry logic
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not self._sql_initialized or not self._pg_conn:
            logger.error("SQL connection not initialized")
            return []
        
        # Get config
        config = get_search_config()
        max_retries = config.get("sql_search", "retry_attempts", 3)
        backoff_factor = config.get("sql_search", "retry_backoff", 1.5)
        
        # Try to reconnect if connection is closed
        if self._pg_conn.closed:
            logger.warning("SQL connection closed, attempting to reconnect")
            try:
                self._init_sql_connection()
            except Exception as e:
                logger.error(f"Failed to reconnect to SQL: {e}")
                return []
        
        for attempt in range(max_retries + 1):
            try:
                # Create a cursor
                with self._pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    # Execute the search query
                    search_sql = """
                    SELECT b.id, b.url, b.title, b.description, b.image_url, 
                           b.created_at, b.user_id, u.username
                    FROM bookmarks b
                    JOIN users u ON b.user_id = u.id
                    WHERE to_tsvector('english', b.title || ' ' || COALESCE(b.description, '')) @@ plainto_tsquery('english', %s)
                    ORDER BY ts_rank(to_tsvector('english', b.title || ' ' || COALESCE(b.description, '')), plainto_tsquery('english', %s)) DESC
                    LIMIT %s
                    """
                    
                    # Execute the search query
                    cursor.execute(search_sql, (query, query, limit))
                    
                    # Fetch the results
                    sql_results = cursor.fetchall()
                    
                    # Convert to dictionaries
                    results = []
                    for row in sql_results:
                        result = dict(row)
                        # Add additional metadata for combining results
                        result['source'] = 'sql'
                        result['score'] = 0.0  # Will be normalized later
                        results.append(result)
                    
                    logger.info(f"SQL search returned {len(results)} results")
                    return results
            
            except Exception as e:
                logger.error(f"SQL search error (attempt {attempt+1}/{max_retries+1}): {e}")
                
                # Check if we should retry
                if attempt < max_retries:
                    # Calculate backoff time with jitter
                    backoff_time = (backoff_factor ** attempt) * (0.5 + random.random())
                    
                    # Log and sleep
                    logger.info(f"Retrying SQL search in {backoff_time:.2f} seconds")
                    await asyncio.sleep(backoff_time)
                    
                    # Try to reconnect if necessary
                    try:
                        self._init_sql_connection()
                    except:
                        pass
                else:
                    logger.error(f"SQL search failed after {max_retries+1} attempts")
        
        return []
    
    def _normalize_scores(self, results: List[Dict[str, Any]], max_score: float = 1.0) -> List[Dict[str, Any]]:
        """
        Normalize relevance scores of search results
        
        Args:
            results: List of search results
            max_score: Maximum score to normalize to
            
        Returns:
            List of results with normalized scores
        """
        if not results:
            return []
        
        # Find the maximum score
        scores = [result.get('score', 0.0) for result in results]
        current_max = max(scores) if scores else 1.0
        
        # Avoid division by zero
        if current_max == 0.0:
            current_max = 1.0
        
        # Normalize scores
        for result in results:
            # Get the current score, default to 0.0
            score = result.get('score', 0.0)
            
            # Normalize to the target maximum
            normalized_score = (score / current_max) * max_score
            
            # Set the normalized score
            result['score'] = normalized_score
            
            # Add the original score for reference
            result['original_score'] = score
        
        return results
    
    def _combine_results(self,
                        vector_results: List[Dict[str, Any]],
                        sql_results: List[Dict[str, Any]],
                        alpha: float = 0.7,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Combine results from vector and SQL searches
        
        Args:
            vector_results: Results from vector search
            sql_results: Results from SQL search
            alpha: Weight for vector search scores (0.0 to 1.0)
            limit: Maximum number of results to return
            
        Returns:
            Combined and ranked results
        """
        # Create a map of bookmark IDs to results
        result_map = {}
        
        # Process vector results
        vector_results = self._normalize_scores(vector_results)
        for result in vector_results:
            bookmark_id = result.get('id')
            if bookmark_id:
                result_map[bookmark_id] = {
                    **result,
                    'vector_score': result.get('score', 0.0),
                    'combined_score': result.get('score', 0.0) * alpha,
                    'sources': ['vector']
                }
        
        # Process SQL results
        sql_results = self._normalize_scores(sql_results)
        for result in sql_results:
            bookmark_id = result.get('id')
            if bookmark_id:
                if bookmark_id in result_map:
                    # Update existing result
                    combined = result_map[bookmark_id]
                    sql_score = result.get('score', 0.0)
                    
                    # Calculate new combined score
                    combined['sql_score'] = sql_score
                    combined['combined_score'] = (
                        combined.get('vector_score', 0.0) * alpha +
                        sql_score * (1.0 - alpha)
                    )
                    
                    # Update sources
                    if 'sql' not in combined.get('sources', []):
                        combined['sources'].append('sql')
                else:
                    # Add new result
                    result_map[bookmark_id] = {
                        **result,
                        'sql_score': result.get('score', 0.0),
                        'combined_score': result.get('score', 0.0) * (1.0 - alpha),
                        'sources': ['sql']
                    }
        
        # Convert map to list and sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
        
        # Return top results
        return combined_results[:limit]
    
    async def search(self, query: str, limit: int = 10, 
                 context: Optional[Dict[str, Any]] = None,
                 alpha: float = 0.7) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute the search across multiple sources with proper error handling
        
        Args:
            query: The search query
            limit: Maximum number of results
            context: Additional context and search parameters
            alpha: Weight for vector search (0-1)
            
        Returns:
            Tuple of (results, metadata)
        """
        self.ensure_initialized()
        
        context = context or {}
        results = []
        metadata = {
            'timestamp': time.time(),
            'query': query,
            'limit': limit,
            'alpha': alpha,
            'vector_results': 0,
            'sql_results': 0
        }
        
        # Extract special search parameters from context
        is_acronym_search = context.get('acronym_search', False)
        score_threshold = context.get('score_threshold', 0.4)  # Default threshold
        
        # Adjust parameters based on query
        if not is_acronym_search and self._is_likely_acronym(query):
            is_acronym_search = True
            score_threshold = min(score_threshold, 0.3)  # More permissive for acronyms
        
        # Log search details
        search_type = "acronym" if is_acronym_search else "standard"
        logger.info(f"Performing {search_type} search for: '{query}' (threshold={score_threshold})")
        
        try:
            # Try vector search first if enabled in config
            if self._vector_store and self._sql_initialized:
                try:
                    start_time = time.time()
                    
                    # Get any SQL results to use as candidates
                    candidate_ids = await self._get_sql_candidates(query)
                    
                    # Perform vector search
                    vector_results = await self._vector_store.async_search(
                        query=query,
                        limit=limit * 3,  # Triple limit for better coverage
                        candidate_ids=candidate_ids,
                        score_threshold=score_threshold  # Use context-aware threshold
                    )
                    
                    # Process and score vector results
                    if vector_results:
                        metadata['vector_time'] = time.time() - start_time
                        metadata['vector_results'] = len(vector_results)
                        logger.info(f"Vector search found {len(vector_results)} results in {metadata['vector_time']:.3f}s")
                        
                        # Get formatted results
                        results.extend(await self._format_results(vector_results))
                except Exception as ve:
                    logger.error(f"Vector search error: {ve}")
                    metadata['vector_error'] = str(ve)
            
            # If we didn't get enough results from vector search, try SQL
            if len(results) < limit:
                try:
                    start_time = time.time()
                    
                    # If this is an acronym search, use a different SQL strategy
                    if is_acronym_search:
                        sql_results = await self._search_sql_for_acronym(query, limit * 2)
                    else:
                        # Regular search
                        sql_results = await self._search_sql(query, limit * 2)
                    
                    if sql_results:
                        metadata['sql_time'] = time.time() - start_time
                        metadata['sql_results'] = len(sql_results)
                        logger.info(f"SQL search found {len(sql_results)} results in {metadata['sql_time']:.3f}s")
                        
                        # Add SQL results
                        results.extend(sql_results)
                except Exception as se:
                    logger.error(f"SQL search error: {se}")
                    metadata['sql_error'] = str(se)
            
            # Remove duplicates
            unique_results = self._deduplicate_results(results)
            
            # If we still don't have enough results, try word-by-word search
            if len(unique_results) < limit and ' ' in query:
                word_results = await self._search_by_words(query, limit, score_threshold)
                if word_results:
                    unique_results.extend(word_results)
                    # Deduplicate again
                    unique_results = self._deduplicate_results(unique_results)
            
            # Sort by relevance
            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Log results
            metadata['total_results'] = len(unique_results)
            metadata['returned_results'] = min(limit, len(unique_results))
            logger.info(f"Returning {metadata['returned_results']} results (from {metadata['total_results']} total)")
            
            return unique_results[:limit], metadata
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [], {
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _is_likely_acronym(self, query: str) -> bool:
        """
        Check if the query is likely an acronym or abbreviation
        
        Args:
            query: The query string
            
        Returns:
            True if likely an acronym
        """
        # All uppercase with 2-6 characters
        if query.isupper() and 2 <= len(query) <= 6:
            return True
            
        # Contains uppercase letters and no lowercase
        if any(c.isupper() for c in query) and not any(c.islower() for c in query):
            return True
            
        # Contains periods between letters like "M.C.P."
        if '.' in query and all(c.isupper() or c == '.' for c in query):
            return True
            
        return False
    
    async def _search_by_words(self, query: str, limit: int, score_threshold: float) -> List[Dict[str, Any]]:
        """
        Search by individual words in multi-word queries
        
        Args:
            query: The multi-word query
            limit: Maximum results
            score_threshold: Score threshold for vector search
            
        Returns:
            Search results
        """
        logger.info(f"Searching by individual words for query: '{query}'")
        words = [w for w in query.split() if len(w) > 2]
        if not words:
            return []
            
        all_results = []
        for word in words[:3]:  # Limit to first 3 words to avoid too many searches
            try:
                # Try vector search first
                if self._vector_store:
                    word_vector_results = await self._vector_store.async_search(
                        query=word,
                        limit=limit,
                        score_threshold=score_threshold
                    )
                    if word_vector_results:
                        formatted = await self._format_results(word_vector_results)
                        all_results.extend(formatted)
                
                # Try SQL search
                word_sql_results = await self._search_sql(word, limit)
                if word_sql_results:
                    all_results.extend(word_sql_results)
                    
            except Exception as e:
                logger.error(f"Error in word search for '{word}': {e}")
                
        # Deduplicate and return
        return self._deduplicate_results(all_results)
        
    async def _search_sql_for_acronym(self, acronym: str, limit: int) -> List[Dict[str, Any]]:
        """
        Special SQL search strategy for acronyms
        
        Args:
            acronym: The acronym to search for
            limit: Maximum results
            
        Returns:
            SQL search results
        """
        # Clean up acronym
        clean_acronym = ''.join(c for c in acronym if c.isalnum())
        
        logger.info(f"SQL acronym search for: '{clean_acronym}'")
        
        # Try both exact match and fuzzy containment
        try:
            # This is a sample implementation - adjust based on actual database schema
            # First try exact match with the acronym
            exact_results = await self._search_sql(clean_acronym, limit)
            
            # Then try LIKE search with wildcards
            # This simulates using SQL's LIKE operator: WHERE text LIKE '%M%C%P%'
            # The actual implementation depends on your database access methods
            like_pattern = '%' + '%'.join(clean_acronym) + '%'
            
            # If SQL API supports LIKE directly
            # like_results = await self._sql_api.search_with_like(like_pattern, limit)
            
            # Combine results
            combined = exact_results  # + like_results
            
            return combined
        except Exception as e:
            logger.error(f"Error in SQL acronym search: {e}")
            return []
            
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on bookmark ID
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results
        """
        seen_ids = set()
        unique_results = []
        
        for result in results:
            result_id = result.get('id')
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
                
        return unique_results
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the search components"""
        vector_status = {}
        if self._vector_store:
            try:
                vector_status = self._vector_store.get_status()
            except Exception as e:
                vector_status = {
                    'error': str(e)
                }
        
        return {
            'initialized': self._initialized,
            'vector_store': vector_status,
            'sql_initialized': self._sql_initialized,
            'last_error': self._last_error
        }
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        if self._vector_store:
            try:
                self._vector_store.clear_cache()
                logger.info("Cleared vector store cache")
            except Exception as e:
                logger.error(f"Error clearing vector store cache: {e}") 