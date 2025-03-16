"""
Main chat engine for the PythonAnywhere chat implementation.
This module provides a Gemini-based implementation of the chat functionality.
"""

import logging
import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import traceback

# Local imports
from .intent_classifier_pa import IntentClassifierPA, IntentType
from .conversation_manager_pa import ConversationManagerPA
from .prompt_manager_pa import PromptManagerPA
from .chat_search_pa import ChatBookmarkSearchPA

logger = logging.getLogger(__name__)

class BookmarkChatPA:
    """
    PythonAnywhere chat engine that provides a conversational interface to bookmarks.
    Uses Google's Gemini 2.0 API for LLM capabilities.
    """
    
    def __init__(self, search_engine=None):
        """
        Initialize the chat engine.
        
        Args:
            search_engine: Optional search engine to use
        """
        # Initialize components
        self.chat_search = ChatBookmarkSearchPA()
        self.intent_classifier = IntentClassifierPA()
        self.conversation_manager = ConversationManagerPA()
        self.prompt_manager = PromptManagerPA()
        
        # Track initialization status
        self._model = None
        self._is_model_initialized = False
        
        # Keep track of external search engine for compatibility
        self.external_search = search_engine
        
        logger.info("✓ BookmarkChatPA initialized")
    
    def _initialize_model(self) -> None:
        """Initialize the LLM model if not already initialized."""
        if self._is_model_initialized:
            return
            
        try:
            # Check which model to use from environment
            model_name = os.getenv('CHAT_MODEL', 'gemini').lower()
            
            if model_name == 'gemini':
                logger.info("Initializing Gemini model")
                self._initialize_gemini()
            else:
                logger.warning(f"Unknown model '{model_name}', defaulting to Gemini")
                self._initialize_gemini()
                
            self._is_model_initialized = True
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _initialize_gemini(self) -> None:
        """Initialize Google's Gemini model."""
        try:
            # Import Google's Gemini SDK
            import google.generativeai as genai
            
            # Get API key from environment
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            
            # Configure the SDK
            genai.configure(api_key=api_key)
            
            # Create model
            # Use 'gemini-pro-1.0' initially and fall back to 'gemini-pro' if not available
            # In the future, we can update this to use 'gemini-pro-1.5' or newer
            try:
                model_id = 'gemini-2.0-flash'
                self._model = genai.GenerativeModel(model_id)
                logger.info(f"✓ Successfully initialized Gemini model: {model_id}")
            except Exception as model_error:
                logger.warning(f"Error with model {model_id}: {model_error}")
                logger.warning("Falling back to gemini-pro")
                model_id = 'gemini-pro'
                self._model = genai.GenerativeModel(model_id)
                logger.info(f"✓ Successfully initialized fallback Gemini model: {model_id}")
                
            logger.info("✓ Gemini model initialized successfully")
        except ImportError:
            logger.error("Failed to import google.generativeai. Please install with: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Process a chat message and generate a response.
        
        Args:
            message: The user's chat message
            history: Optional list of previous messages
            
        Returns:
            A tuple of (response_text, bookmarks_used, model_name)
        """
        # Initialize LLM if not already initialized
        self._initialize_model()
        
        try:
            # Log the chat request with a truncated message
            truncated_message = message[:50] + "..." if len(message) > 50 else message
            logger.info(f"Chat message: '{truncated_message}'")
            
            # Update conversation with user message
            if history:
                # If we have history, first add it to the conversation manager
                for msg in history:
                    self.conversation_manager.add_message(
                        role=msg.get('role', 'unknown'),
                        content=msg.get('content', ''),
                        metadata=msg.get('metadata', {})
                    )
            
            # Add current message
            self.conversation_manager.add_message(role='user', content=message)
            
            # Get recent messages for context
            recent_messages = self.conversation_manager.get_recent_messages()
            
            # Get additional context
            context = self.conversation_manager.get_context()
            
            # Analyze the intent of the message
            intent = self.intent_classifier.analyze(message, recent_messages, context)
            logger.info(f"Detected intent: {intent.type}")
            
            # Process the message based on intent
            if intent.type in [IntentType.SEARCH, IntentType.FOLLOWUP]:
                logger.info(f"Processing as search intent: {intent.type}")
                return self._handle_search_intent(message, intent, context)
            elif intent.type == IntentType.OPINION:
                logger.info(f"Processing as opinion intent")
                return self._handle_opinion_intent(message, intent, context)
            else:
                logger.info(f"Processing as general conversation")
                return self._handle_general_intent(message, context)
                
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            logger.error(traceback.format_exc())
            
            # Return a friendly error message
            error_msg = "I'm sorry, I encountered an error while processing your message. Please try again."
            return error_msg, [], "gemini"
    
    def _handle_search_intent(self, message: str, intent: Any, 
                            context: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Handle a search intent by finding relevant bookmarks.
        
        Args:
            message: The user's message
            intent: The classified intent
            context: The conversation context
            
        Returns:
            A tuple of (response_text, bookmarks_used, model_name)
        """
        try:
            # Extract query from intent if available, otherwise use full message
            query = intent.params.get('query', message)
            
            # Refine query if this is a followup and we have a topic
            if intent.type == IntentType.FOLLOWUP and context.get('current_topic'):
                query = f"{context.get('current_topic', '')} {query}".strip()
                logger.info(f"Refined followup query: '{query}'")
            
            # Search for relevant bookmarks
            results = self.chat_search.search(query=query, limit=5, context=context)
            
            if not results:
                logger.info(f"No search results found for '{query}'")
                # Generate a "no results" response
                no_results_response = (
                    f"I couldn't find any bookmarks related to '{query}'. "
                    f"Would you like to try a different search term?"
                )
                return no_results_response, [], "gemini"
            
            # Log search results
            logger.info(f"Found {len(results)} results for '{query}'")
            
            # Format the search results for the prompt
            bookmark_context = self.prompt_manager.format_bookmark_context(results)
            
            # Generate a prompt for the RAG response
            prompt_params = {
                'user_query': message,
                'bookmark_context': bookmark_context
            }
            prompt = self.prompt_manager.get_prompt('rag_response', prompt_params)
            
            # Generate response with Gemini
            response_text = self._generate_gemini_response(prompt)
            
            # Track the search in conversation context
            categories = set()
            for result in results:
                if result.get('categories'):
                    categories.update(result.get('categories'))
                    
            self.conversation_manager.add_search_result(
                query=query, 
                num_results=len(results),
                categories=list(categories)
            )
            
            # Add assistant message to conversation history
            self.conversation_manager.add_message(
                role='assistant',
                content=response_text,
                metadata={
                    'bookmarks_used': len(results),
                    'query': query,
                    'categories': list(categories)
                }
            )
            
            return response_text, results, "gemini"
            
        except Exception as e:
            logger.error(f"Error handling search intent: {e}")
            logger.error(traceback.format_exc())
            
            # Return a friendly error message
            error_msg = "I'm sorry, I encountered an error searching your bookmarks. Please try again."
            return error_msg, [], "gemini"
    
    def _handle_opinion_intent(self, message: str, intent: Any, 
                              context: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Handle an opinion/analysis intent.
        
        Args:
            message: The user's message
            intent: The classified intent
            context: The conversation context
            
        Returns:
            A tuple of (response_text, bookmarks_used, model_name)
        """
        try:
            # For opinion intents, we'll still try to find relevant information in bookmarks
            # but we'll give the model more freedom to synthesize and analyze
            
            # Extract query from intent if available
            query = intent.params.get('query', message)
            
            # Search for potentially relevant bookmarks
            results = self.chat_search.search(query=query, limit=3, context=context)
            
            if results:
                logger.info(f"Found {len(results)} bookmarks for opinion query")
                
                # Format the search results for the prompt
                bookmark_context = self.prompt_manager.format_bookmark_context(results)
                
                # Generate a prompt that asks for analysis/opinion
                prompt = f"""<SYSTEM>
You are a helpful AI assistant providing thoughtful analysis based on Twitter bookmarks and your own knowledge.
When answering, you can reference the bookmarks provided but also add your own insights and analysis.
</SYSTEM>

<USER_QUERY>
{message}
</USER_QUERY>

<RELEVANT_BOOKMARKS>
{bookmark_context}
</RELEVANT_BOOKMARKS>

<INSTRUCTIONS>
1. Provide a thoughtful analysis or opinion in response to the user's query.
2. You may reference the bookmarks if relevant, but don't feel constrained by them.
3. Clearly distinguish between information from the bookmarks and your own insights.
4. Maintain a conversational and helpful tone.
</INSTRUCTIONS>

Based on the user's question and the available information, here's my analysis:"""
                
                # Generate response
                response_text = self._generate_gemini_response(prompt)
                
            else:
                logger.info("No bookmarks found for opinion query, using general knowledge")
                
                # Generate a prompt for a general opinion/analysis
                prompt = f"""<SYSTEM>
You are a helpful AI assistant providing thoughtful analysis and opinions based on your knowledge.
</SYSTEM>

<USER_QUERY>
{message}
</USER_QUERY>

<INSTRUCTIONS>
1. Provide a thoughtful analysis or opinion in response to the user's query.
2. Draw on your general knowledge to provide helpful insights.
3. If you don't have enough information to provide a definitive answer, acknowledge that limitation.
4. Maintain a conversational and helpful tone.
</INSTRUCTIONS>

Here's my thoughtful response:"""
                
                # Generate response
                response_text = self._generate_gemini_response(prompt)
                results = []  # No bookmarks used
            
            # Add assistant message to conversation history
            self.conversation_manager.add_message(
                role='assistant',
                content=response_text,
                metadata={
                    'bookmarks_used': len(results),
                    'query': query,
                    'intent': 'opinion'
                }
            )
            
            return response_text, results, "gemini"
            
        except Exception as e:
            logger.error(f"Error handling opinion intent: {e}")
            logger.error(traceback.format_exc())
            
            # Return a friendly error message
            error_msg = "I'm sorry, I encountered an error analyzing your question. Please try again."
            return error_msg, [], "gemini"
    
    def _handle_general_intent(self, message: str, 
                             context: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Handle a general conversation intent.
        
        Args:
            message: The user's message
            context: The conversation context
            
        Returns:
            A tuple of (response_text, bookmarks_used, model_name)
        """
        try:
            # For general chat, we won't search bookmarks by default,
            # but we'll provide a natural conversational response
            
            # Format conversation history for context
            conversation_history = self.prompt_manager.format_conversation_history(
                context.get('recent_messages', [])
            )
            
            # Generate a prompt for general conversation
            prompt_params = {
                'user_query': message,
                'conversation_history': conversation_history
            }
            prompt = self.prompt_manager.get_prompt('general_conversation', prompt_params)
            
            # Generate response
            response_text = self._generate_gemini_response(prompt)
            
            # Add assistant message to conversation history
            self.conversation_manager.add_message(
                role='assistant',
                content=response_text,
                metadata={
                    'bookmarks_used': 0,
                    'intent': 'general'
                }
            )
            
            return response_text, [], "gemini"
            
        except Exception as e:
            logger.error(f"Error handling general intent: {e}")
            logger.error(traceback.format_exc())
            
            # Return a friendly error message
            error_msg = "I'm sorry, I encountered an error responding to your message. Please try again."
            return error_msg, [], "gemini"
    
    def _generate_gemini_response(self, prompt: str) -> str:
        """
        Generate a response using Gemini API.
        
        Args:
            prompt: The formatted prompt to send to Gemini
            
        Returns:
            The generated response text
        """
        try:
            # Ensure model is initialized
            if not self._is_model_initialized or not self._model:
                self._initialize_model()
            
            # Log a truncated version of the prompt for debugging
            truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            logger.info(f"Generating Gemini response for prompt: '{truncated_prompt}'")
            
            # Generate response
            generation_result = self._model.generate_content(prompt)
            
            # Extract text from the response
            # Different versions of the API return different response formats
            if hasattr(generation_result, 'text'):
                response_text = generation_result.text
            elif hasattr(generation_result, 'parts'):
                # Handle multi-part responses
                response_text = "".join([part.text for part in generation_result.parts])
            else:
                # Fallback
                response_text = str(generation_result)
                
            # Log a truncated version of the response
            truncated_response = response_text[:100] + "..." if len(response_text) > 100 else response_text
            logger.info(f"Generated response: '{truncated_response}'")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            logger.error(traceback.format_exc())
            
            # Return a friendly error message
            return "I'm sorry, I experienced an issue connecting to the AI service. Please try again in a moment."
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.conversation_manager.reset()
        logger.info("Conversation has been reset")
        
class NotInitializedError(Exception):
    """Exception raised when a required component is not initialized."""
    pass
