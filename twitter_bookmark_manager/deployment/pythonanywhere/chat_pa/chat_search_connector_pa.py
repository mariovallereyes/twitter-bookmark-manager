"""
Enhanced search connector for the Twitter Bookmark Manager chat interface.
This connector extends the standard ChatBookmarkSearchPA with enhanced search capabilities
while maintaining the same interface as the original search.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union

# Fix import path
from .chat_search_pa import ChatBookmarkSearchPA
from .search_manager_pa import get_search_manager
from .search_config_pa import get_search_config

# Global instance
_chat_search_connector = None

class ChatSearchConnectorPA(ChatBookmarkSearchPA):
    """
    Enhanced search connector that extends the standard ChatBookmarkSearchPA
    with advanced search capabilities while maintaining API compatibility.
    """
    
    def __init__(self):
        """Initialize the search connector with enhanced capabilities"""
        # Call parent initializer
        super().__init__()
        
        # Initialize additional components
        self._search_manager = None
        self._config = None
        self._event_loop = None
        self._initialized = False
        
        # Configure logging
        self._logger = logging.getLogger(__name__)
        self._logger.info("ChatSearchConnectorPA initialized in lazy-loading mode")
    
    def _ensure_initialized(self) -> None:
        """Ensure that both parent and enhanced components are initialized"""
        # First ensure parent is initialized
        try:
            super()._ensure_initialized()
        except Exception as e:
            self._logger.error(f"Error initializing parent search: {e}")
        
        # Then initialize enhanced components
        if not self._initialized:
            try:
                # Get configuration
                self._config = get_search_config()
                
                # Get search manager
                self._search_manager = get_search_manager()
                
                # Create or get event loop for async operations
                try:
                    self._event_loop = asyncio.get_event_loop()
                except RuntimeError:
                    self._event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._event_loop)
                
                # Run async initialization
                self._event_loop.run_until_complete(self._async_initialize())
                
                self._initialized = True
                self._logger.info("Enhanced search components initialized")
                
            except Exception as e:
                self._logger.error(f"Error initializing enhanced search components: {e}")
                self._logger.warning("Will fall back to standard search")
    
    async def _async_initialize(self) -> None:
        """Asynchronously initialize search components"""
        if self._search_manager:
            try:
                # Initialize vector store
                await self._search_manager.initialize_async()
                self._logger.info("Async initialization of search components completed")
            except Exception as e:
                self._logger.error(f"Error during async initialization: {e}")
                raise
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform enhanced search for bookmarks based on query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of bookmark dictionaries
        """
        self._ensure_initialized()
        
        # Determine if we should use enhanced search
        use_enhanced = (
            self._initialized and 
            self._search_manager is not None and
            self._config.get('hybrid_search', 'enabled', True)
        )
        
        # Try enhanced search first if enabled
        results = []
        if use_enhanced:
            try:
                self._logger.info(f"Attempting enhanced search for '{query}'")
                start_time = time.time()
                
                # Use the search manager to perform the search
                results = self._search_manager.search(
                    query, 
                    limit=limit,
                    use_hybrid=self._config.get('hybrid_search', 'enabled', True),
                    use_cache=self._config.get('caching', 'enabled', True)
                )
                
                elapsed = time.time() - start_time
                self._logger.info(f"Enhanced search completed in {elapsed:.3f}s with {len(results)} results")
                
                # If we got results, return them
                if results:
                    return results
                else:
                    self._logger.info("Enhanced search returned no results, falling back to standard search")
            except Exception as e:
                self._logger.error(f"Enhanced search failed: {e}, falling back to standard search")
        
        # Fall back to standard search
        self._logger.info(f"Using standard search for '{query}'")
        start_time = time.time()
        results = super().search(query, limit)
        elapsed = time.time() - start_time
        self._logger.info(f"Standard search completed in {elapsed:.3f}s with {len(results)} results")
        
        return results
    
    def search_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search bookmarks by category"""
        # For now, just use the parent implementation
        # In the future, we could enhance this with vector search
        return super().search_by_category(category, limit)
    
    def search_by_user(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search bookmarks by username"""
        # For now, just use the parent implementation
        return super().search_by_user(username, limit)
    
    def get_related_bookmarks(self, bookmark_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get bookmarks related to the given bookmark ID"""
        # For now, just use the parent implementation
        return super().get_related_bookmarks(bookmark_id, limit)

def get_chat_search_connector() -> ChatSearchConnectorPA:
    """Get the global chat search connector instance"""
    global _chat_search_connector
    if _chat_search_connector is None:
        _chat_search_connector = ChatSearchConnectorPA()
    return _chat_search_connector 