"""
Conversation manager for the PythonAnywhere chat implementation.
This module tracks conversation history and manages context for the chat session.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationManagerPA:
    """
    Manages conversation history and context for the chat system.
    Optimized for PythonAnywhere deployment.
    """
    
    def __init__(self, context_window_size: int = 10):
        """
        Initialize the conversation manager.
        
        Args:
            context_window_size: Number of recent messages to keep in context
        """
        self.context_window_size = context_window_size
        self.conversation_history = []
        self.current_topic = None
        self.recent_categories = set()
        self.session_id = f"session_{int(time.time())}"
        self.recent_searches = []
        
        logger.info(f"✓ Conversation manager initialized with session ID: {self.session_id}")
    
    def add_message(self, role: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ('user' or 'assistant')
            content: The message content
            metadata: Additional metadata for the message
        """
        timestamp = datetime.now().isoformat()
        message = {
            'role': role,
            'content': content,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # If we have more messages than the context window, remove oldest ones
        if len(self.conversation_history) > self.context_window_size:
            self.conversation_history = self.conversation_history[-self.context_window_size:]
        
        # Update topic tracking
        if role == 'user':
            # Extract potential topic from user message
            # This is a simplified approach - in a real system, 
            # you might use NLP to extract topics
            words = set(content.lower().split())
            potential_topics = words - self._get_stopwords()
            if potential_topics:
                # Use the longest word as a simple heuristic for topic
                self.current_topic = max(potential_topics, key=len)
                logger.info(f"Updated conversation topic to: {self.current_topic}")
        
        # Update category tracking if specified in metadata
        categories = metadata.get('categories', []) if metadata else []
        if categories:
            self.recent_categories.update(categories)
            while len(self.recent_categories) > 5:  # Keep only 5 most recent categories
                self.recent_categories.pop()
    
    def add_search_result(self, query: str, num_results: int, 
                         categories: Optional[List[str]] = None) -> None:
        """
        Track search results for context.
        
        Args:
            query: The search query
            num_results: Number of results found
            categories: Categories associated with the search
        """
        timestamp = datetime.now().isoformat()
        search = {
            'query': query,
            'num_results': num_results,
            'timestamp': timestamp,
            'categories': categories or []
        }
        
        self.recent_searches.append(search)
        
        # Keep only 5 most recent searches
        if len(self.recent_searches) > 5:
            self.recent_searches = self.recent_searches[-5:]
        
        # Update recent categories
        if categories:
            self.recent_categories.update(categories)
            while len(self.recent_categories) > 5:
                self.recent_categories.pop()
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent messages from the conversation.
        
        Args:
            count: Number of recent messages to retrieve
            
        Returns:
            List of recent message dictionaries
        """
        return self.conversation_history[-count:] if len(self.conversation_history) > 0 else []
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current conversation context.
        
        Returns:
            Dictionary with conversation context
        """
        return {
            'session_id': self.session_id,
            'current_topic': self.current_topic,
            'recent_categories': list(self.recent_categories),
            'recent_messages': self.get_recent_messages(),
            'recent_searches': self.recent_searches
        }
    
    def get_recent_categories(self) -> List[str]:
        """
        Get the recent categories mentioned in the conversation.
        
        Returns:
            List of recent category names
        """
        return list(self.recent_categories)
    
    def reset(self) -> None:
        """Reset the conversation state."""
        self.conversation_history = []
        self.current_topic = None
        self.recent_categories = set()
        self.recent_searches = []
        self.session_id = f"session_{int(time.time())}"
        logger.info(f"Conversation reset, new session ID: {self.session_id}")
    
    def _get_stopwords(self) -> Set[str]:
        """Get common stopwords for topic extraction."""
        return {
            'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 
            'from', 'by', 'about', 'like', 'through', 'after', 'over', 'with', 'in',
            'show', 'me', 'find', 'get', 'tell', 'what', 'where', 'when', 'who',
            'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'shall', 'tweet', 'tweets',
            'twitter', 'bookmark', 'bookmarks'
        }
        
    def save_to_database(self) -> bool:
        """
        Save conversation history to database.
        This is a placeholder - the actual implementation would save to PostgreSQL.
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # In the real implementation, this would connect to PostgreSQL
            # and save the conversation history
            logger.info(f"Would save conversation with ID {self.session_id} to database (placeholder)")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation to database: {e}")
            return False
            
    def load_from_database(self, session_id: str) -> bool:
        """
        Load conversation history from database.
        This is a placeholder - the actual implementation would load from PostgreSQL.
        
        Args:
            session_id: The ID of the session to load
            
        Returns:
            True if load was successful, False otherwise
        """
        try:
            # In the real implementation, this would connect to PostgreSQL
            # and load the conversation history
            logger.info(f"Would load conversation with ID {session_id} from database (placeholder)")
            self.session_id = session_id
            return True
        except Exception as e:
            logger.error(f"Error loading conversation from database: {e}")
            return False
