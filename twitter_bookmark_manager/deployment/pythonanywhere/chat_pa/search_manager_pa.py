"""
Search manager for chat system that provides a clean interface to the enhanced search capabilities
while maintaining isolation from the main system.
"""

import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from functools import lru_cache

from .search_config_pa import get_search_config
from .robust_search_pa import RobustSearchHandler

# Set up logging
logger = logging.getLogger(__name__)

class SearchManager:
    """
    Manages access to search components and provides a clean interface for the chat system.
    Uses the Singleton pattern to ensure only one instance exists.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._search_handler = None
            self._last_error = None
            self._initialization_lock = asyncio.Lock()
            self._initialized = True
            logger.info("Initialized SearchManager")
    
    async def _ensure_handler_initialized(self) -> None:
        """Initialize the search handler if needed"""
        if self._search_handler is not None:
            return
            
        async with self._initialization_lock:
            if self._search_handler is not None:  # Double-check after acquiring lock
                return
                
            try:
                # Initialize the robust search handler
                self._search_handler = RobustSearchHandler()
                # Call synchronous initialization first
                self._search_handler.ensure_initialized()
                logger.info("Successfully initialized search handler")
                
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Failed to initialize search handler: {e}")
                raise
    
    async def initialize_async(self) -> None:
        """Initialize asynchronous components"""
        await self._ensure_handler_initialized()
        
        # Initialize any async components in the handler
        if self._search_handler:
            await self._search_handler.initialize_async()
            logger.info("Async components initialized")
    
    @lru_cache(maxsize=1000)
    def _generate_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a cache key for the query and context"""
        import hashlib
        import json
        
        # Create a string that includes both query and context
        key_parts = [query]
        if context:
            # Sort the context dictionary to ensure consistent keys
            key_parts.append(json.dumps(context, sort_keys=True))
        
        # Create a hash of the combined string
        key_string = '|'.join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def search(self, 
              query: str, 
              limit: int = 10, 
              use_hybrid: bool = True, 
              use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Synchronous search method that wraps the async search
        
        Args:
            query: The search query
            limit: Maximum number of results
            use_hybrid: Whether to use hybrid search
            use_cache: Whether to use caching
            
        Returns:
            List of bookmark results
        """
        # Process the query first
        processed_query = self._process_query(query)
        
        # If the query is significantly different, log it
        if processed_query != query:
            logger.info(f"Processed query '{query}' to '{processed_query}'")
        
        # Get event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Set up context
        context = {
            'use_hybrid': use_hybrid,
            'use_cache': use_cache,
            'original_query': query
        }
        
        # Run async search and return only the results
        results, _ = loop.run_until_complete(self.async_search(
            query=processed_query,
            limit=limit,
            context=context
        ))
        
        # If no results and the processed query is different, try original query
        if not results and processed_query != query:
            logger.info(f"No results for processed query, trying original: '{query}'")
            results, _ = loop.run_until_complete(self.async_search(
                query=query,
                limit=limit,
                context=context
            ))
        
        # If still no results and it looks like an acronym, try a more specific approach
        if not results and self._is_likely_acronym(query):
            logger.info(f"Query '{query}' appears to be an acronym, trying letter-by-letter search")
            letter_results = self._search_by_acronym_letters(query, limit, loop, context)
            if letter_results:
                results = letter_results
        
        return results
    
    def _process_query(self, query: str) -> str:
        """
        Process the query to enhance search results
        
        Args:
            query: The original query string
            
        Returns:
            Processed query
        """
        # Remove common search phrases
        phrases_to_remove = [
            "find", "search for", "look for", "show me", 
            "tweets about", "i want", "i need", "please",
            "related to", "anything about", "regarding"
        ]
        
        processed = query.lower()
        for phrase in phrases_to_remove:
            processed = re.sub(r'\b' + phrase + r'\b', '', processed, flags=re.IGNORECASE)
        
        # Clean up whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # If query looks like an acronym, preserve case
        if self._is_likely_acronym(query):
            return query
            
        return processed
    
    def _is_likely_acronym(self, query: str) -> bool:
        """
        Check if the query is likely an acronym or abbreviation
        
        Args:
            query: The query string
            
        Returns:
            True if likely an acronym
        """
        # All uppercase with 2-6 characters
        if re.match(r'^[A-Z]{2,6}$', query):
            return True
            
        # MixedCase like "API" in "APIEndpoint"
        if re.match(r'^[A-Z][a-z]+([A-Z][a-z]*)+$', query):
            return True
            
        # Dot-separated like "M.C.P."
        if re.match(r'^([A-Z]\.)+$', query):
            return True
            
        # Dash-separated like "M-C-P"
        if re.match(r'^([A-Z]\-)+[A-Z]$', query):
            return True
            
        return False
    
    def _search_by_acronym_letters(self, acronym: str, limit: int, 
                                 loop: asyncio.AbstractEventLoop, 
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search by expanding acronym letters into words
        
        Args:
            acronym: The acronym to search by
            limit: Maximum results
            loop: Event loop
            context: Search context
            
        Returns:
            Search results
        """
        # Strip any punctuation from the acronym
        clean_acronym = re.sub(r'[^A-Za-z]', '', acronym)
        
        # Build a query searching for each letter
        expanded_query = ' '.join(clean_acronym)
        
        logger.info(f"Searching with expanded acronym: '{expanded_query}'")
        
        # Run search with the expanded query and lower threshold
        context['acronym_search'] = True
        context['score_threshold'] = 0.3  # Lower threshold for acronym searches
        
        results, _ = loop.run_until_complete(self.async_search(
            query=expanded_query,
            limit=limit * 2,  # Search for more to find better matches
            context=context
        ))
        
        return results
    
    async def async_search(self,
                    query: str,
                    limit: int = 10,
                    context: Optional[Dict] = None,
                    alpha: float = 0.7) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a search query with proper error handling and retries
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            context: Additional context for the search
            alpha: Weight for vector search scores
            
        Returns:
            Tuple of (results, metadata)
        """
        try:
            await self._ensure_handler_initialized()
            
            # Get config
            config = get_search_config()
            
            # Override alpha with config if not explicitly provided
            if alpha is None:
                alpha = config.get('vector_search', 'alpha', 0.7)
            
            # Execute search with the handler
            results, metadata = await self._search_handler.search(
                query=query,
                limit=limit,
                context=context,
                alpha=alpha
            )
            
            # Add search manager metadata
            metadata.update({
                'cache_key': self._generate_cache_key(query, context),
                'search_manager_version': '1.0.0'
            })
            
            return results, metadata
            
        except Exception as e:
            logger.error(f"Search error in manager: {e}")
            return [], {
                'error': str(e),
                'last_initialization_error': self._last_error,
                'timestamp': time.time()
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status information about the search system"""
        try:
            await self._ensure_handler_initialized()
            
            if self._search_handler:
                handler_status = self._search_handler.get_status()
            else:
                handler_status = {'initialized': False}
            
            cache_info = None
            try:
                cache_info = self._generate_cache_key.cache_info()._asdict()
            except:
                cache_info = {'hits': 0, 'misses': 0, 'currsize': 0}
            
            return {
                'initialized': self._initialized,
                'handler_status': handler_status,
                'last_error': self._last_error,
                'cache_info': cache_info
            }
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'initialized': self._initialized,
                'error': str(e),
                'last_error': self._last_error
            }
    
    def clear_caches(self) -> None:
        """Clear all internal caches"""
        self._generate_cache_key.cache_clear()
        
        # Clear handler caches if available
        if self._search_handler:
            try:
                self._search_handler.clear_caches()
            except:
                pass
                
        logger.info("Cleared all search manager caches")

# Create a global instance
_search_manager = None

def get_search_manager() -> SearchManager:
    """Get the global search manager instance"""
    global _search_manager
    if _search_manager is None:
        _search_manager = SearchManager()
    return _search_manager 