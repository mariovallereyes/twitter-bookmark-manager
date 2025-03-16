"""
Chat package for the Twitter Bookmark Manager - PythonAnywhere implementation.

This package provides a conversational AI chat interface for the Twitter Bookmark Manager,
allowing users to search and interact with their bookmarks through natural language.

The implementation uses Google's Gemini models for LLM capabilities and is designed to
work with the PostgreSQL database and Qdrant vector store used in the PythonAnywhere deployment.
"""

import logging
import os
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Main chat engine import - this will be the primary interface
from .engine_pa import BookmarkChatPA

# Create a singleton instance of the chat engine
_chat_engine = None

def get_chat_engine(search_engine=None) -> BookmarkChatPA:
    """
    Get the singleton instance of the BookmarkChatPA engine.
    
    Args:
        search_engine: Optional search engine to use with the chat engine
        
    Returns:
        BookmarkChatPA instance
    """
    global _chat_engine
    
    try:
        if _chat_engine is None:
            logger.info("Initializing BookmarkChatPA engine")
            _chat_engine = BookmarkChatPA(search_engine=search_engine)
            
        return _chat_engine
    except Exception as e:
        logger.error(f"Error initializing chat engine: {e}")
        logger.error(traceback.format_exc())
        raise

def chat(message: str, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Process a chat message and return a response.
    This is the main entry point for the chat API.
    
    Args:
        message: The user's chat message
        history: Optional conversation history
        
    Returns:
        Tuple of (response_text, bookmarks_used, model_name)
    """
    try:
        # Get the chat engine
        engine = get_chat_engine()
        
        # Process the message
        return engine.chat(message, history)
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        logger.error(traceback.format_exc())
        
        # Return a friendly error message
        error_msg = "I'm sorry, I encountered an error while processing your message. Please try again."
        return error_msg, [], "error"

def reset_conversation() -> None:
    """
    Reset the current conversation state.
    
    Returns:
        None
    """
    if _chat_engine is not None:
        _chat_engine.reset_conversation()
        logger.info("Conversation has been reset")
    else:
        logger.warning("Attempted to reset conversation but chat engine is not initialized")

# Export under the name the existing API expects
BookmarkChat = BookmarkChatPA

# Export other components that might be needed
from .intent_classifier_pa import IntentClassifierPA
from .chat_search_pa import ChatBookmarkSearchPA
from .conversation_manager_pa import ConversationManagerPA
from .prompt_manager_pa import PromptManagerPA

# This allows the existing routes to work without modification
# when importing from twitter_bookmark_manager.deployment.pythonanywhere.chat_pa
