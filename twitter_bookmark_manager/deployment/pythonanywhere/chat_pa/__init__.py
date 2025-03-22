"""
Chat-specific module for Twitter Bookmark Manager.
This module provides chat functionality using a dual-model approach with
fallback mechanisms and enhanced search capabilities.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flags and instances
_chat_engine = None
_use_enhanced_search = os.environ.get('USE_ENHANCED_SEARCH', 'true').lower() == 'true'
_components_initialized = False

# Import the chat engine - Fix path to match actual structure
from .engine_pa import BookmarkChatPA

# No need for alias as we're using the correct class name now
BookmarkChat = BookmarkChatPA

def _initialize_components():
    """Initialize components based on configuration"""
    global _components_initialized
    
    if _components_initialized:
        return
    
    try:
        if _use_enhanced_search:
            # Import enhanced search components
            from .search_config_pa import get_search_config
            from .chat_search_connector_pa import get_chat_search_connector
            
            # Initialize the configuration
            config = get_search_config()
            logger.info("Initialized search configuration")
            
            # Initialize the search connector
            connector = get_chat_search_connector()
            logger.info("Initialized enhanced search connector")
        
        _components_initialized = True
        logger.info(f"Chat module components initialized (enhanced search: {_use_enhanced_search})")
        
    except Exception as e:
        logger.error(f"Error initializing chat components: {e}")
        # Continue with standard initialization

def get_chat_engine() -> BookmarkChatPA:
    """
    Get the chat engine instance.
    
    Returns:
        The global chat engine instance
    """
    global _chat_engine
    
    if _chat_engine is None:
        # Initialize components if needed
        _initialize_components()
        
        # Create the chat engine
        _chat_engine = BookmarkChatPA()
        
        # Set up the chat engine with appropriate search
        if _use_enhanced_search:
            try:
                # Import and use the enhanced search connector
                from .chat_search_connector_pa import get_chat_search_connector
                search = get_chat_search_connector()
                _chat_engine = BookmarkChatPA(search_engine=search)
                logger.info("Chat engine initialized with enhanced search")
            except Exception as e:
                logger.error(f"Error setting up enhanced search: {e}")
                # Fall back to standard search
                _chat_engine = BookmarkChatPA()
                logger.info("Chat engine initialized with standard search (fallback)")
        else:
            # Use standard initialization
            _chat_engine = BookmarkChatPA()
            logger.info("Chat engine initialized with standard search")
    
    return _chat_engine

def chat(message: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Process a chat message and return a response.
    This is the main entry point for the chat API.
    
    Args:
        message: The user's message
        history: Optional conversation history
        
    Returns:
        Dictionary with response_text, bookmarks, and model_name
    """
    # Get the chat engine
    engine = get_chat_engine()
    
    try:
        # Process the message using the engine's chat method
        response_text, bookmarks, model_name = engine.chat(message, history)
        
        # Return a properly formatted dictionary for JSON serialization
        return {
            "response": response_text,
            "bookmarks": bookmarks,
            "model": model_name
        }
    
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        # Return a minimal response in case of error
        return {
            "response": "I'm sorry, I encountered an error while processing your message.",
            "bookmarks": [],
            "model": "error"
        }

def toggle_enhanced_search(enable: bool = True) -> bool:
    """
    Toggle enhanced search on or off.
    
    Args:
        enable: Whether to enable enhanced search
        
    Returns:
        True if successful, False otherwise
    """
    global _use_enhanced_search, _chat_engine
    
    if _use_enhanced_search == enable:
        # No change needed
        return True
    
    try:
        # Update the flag
        _use_enhanced_search = enable
        
        # Reset the chat engine to recreate with new settings
        _chat_engine = None
        
        # Force reinitialization of components
        global _components_initialized
        _components_initialized = False
        
        # Initialize with new settings
        _initialize_components()
        
        logger.info(f"Enhanced search {'enabled' if enable else 'disabled'}")
        return True
        
    except Exception as e:
        logger.error(f"Error toggling enhanced search: {e}")
        return False

def reset_conversation() -> None:
    """Reset the current conversation state"""
    # Get the chat engine
    engine = get_chat_engine()
    
    try:
        # Reset the conversation
        engine.reset_conversation()
        logger.info("Reset conversation state")
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")

# Pre-initialize components
_initialize_components()
