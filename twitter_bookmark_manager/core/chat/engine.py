import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Added for Gemini integration
# import google.generativeai as genai
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

from .intent_classifier import IntentClassifier, Intent, IntentType
from .conversation_manager import ConversationManager
from .prompt_manager import PromptManager
from ..search import BookmarkSearch
from .chat_search import ChatBookmarkSearch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


###############################################################################
# Base Chat Engine Interface
###############################################################################
class BaseChatEngine:
    """
    Abstract interface for chat engines.
    All chat engines must implement the generate_response method.
    """
    def generate_response(self, prompt: str, context: Optional[Any] = None) -> Dict[str, Any]:
        raise NotImplementedError("generate_response must be implemented by subclasses")


###############################################################################
# Mistral Chat Engine Implementation
###############################################################################
class MistralChat(BaseChatEngine):
    """
    Chat engine using the Mistral model via llama_cpp.
    """
    def __init__(self, model_path: Optional[str] = None):
        # Determine the model path; if not provided, use default relative to project structure
        if model_path is None:
            current_dir = Path(__file__).resolve().parent.parent.parent
            model_path = current_dir / "models" / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        else:
            model_path = Path(model_path)
        model_path = model_path.resolve()
        logger.info(f"Loading Mistral model from: {model_path}")

        try:
            # Initialize the Llama model with optimized parameters
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=4096,
                n_threads=4,
                n_batch=512,
                f16_kv=True
            )
            logger.info("✓ Mistral chat engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Mistral model: {e}")
            raise

    def generate_response(self, prompt: str, context: Optional[Any] = None) -> Dict[str, Any]:
        try:
            result = self.llm(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7,
                top_p=0.95,
                stop=["User:", "Assistant:", "\n\n"],
                repeat_penalty=1.1,
                presence_penalty=0.6
            )
            response_text = result["choices"][0]["text"].strip()
            return {
                'text': response_text,
                'model': 'mistral',
                'success': True
            }
        except Exception as e:
            logger.error(f"Mistral generation error: {e}")
            return {
                'text': "I encountered an error generating a response. Please try rephrasing your question.",
                'error': str(e),
                'model': 'mistral',
                'success': False
            }


###############################################################################
# Gemini Chat Engine Implementation
###############################################################################
class GeminiChat(BaseChatEngine):
    """
    Gemini 2.0 chat engine implementation using the google.genai library.
    This class integrates with Gemini using your API key. Ensure that the 
    GEMINI_API_KEY environment variable is set before starting the server.
    
    Example usage:
        export GEMINI_API_KEY="your_actual_api_key"
    """
    def __init__(self):
        try:
            from google import genai
        except ImportError as e:
            logger.error("The google.genai library is required for Gemini integration. Install it via 'pip install google-generativeai'")
            raise e

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        self.client = genai.Client(api_key=api_key)
        # You can adjust the model name if needed
        self.model = "gemini-2.0-flash"
        logger.info("Gemini chat engine initialized successfully using Gemini API.")

    def generate_response(self, prompt: str, context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate a response from Gemini 2.0 using the provided prompt.
        
        Parameters:
            prompt (str): The prompt to send to Gemini.
            context (Optional[Any]): (Unused for now; reserved for future use.)
        
        Returns:
            A dictionary with keys:
                'text': The generated text.
                'model': Identifier for the model used ('gemini').
                'success': Boolean indicating if the call was successful.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            response_text = response.text.strip()
            return {
                'text': response_text,
                'model': 'gemini',
                'success': True
            }
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return {
                'text': "I encountered an error generating a response with Gemini. Please try again later.",
                'error': str(e),
                'model': 'gemini',
                'success': False
            }


###############################################################################
# BookmarkChat: Main Chat System with Dynamic Engine Selection
###############################################################################
class BookmarkChat:
    """
    Enhanced chat engine that provides natural conversation about bookmarks
    using either Gemini or Mistral as the underlying model.
    """
    
    def __init__(self, search_engine: BookmarkSearch):
        self.main_search_engine = search_engine  # Keep the main search engine for compatibility
        self.chat_search_engine = ChatBookmarkSearch()  # Add specialized chat search
        self.intent_classifier = IntentClassifier()
        self.conversation_manager = ConversationManager()
        self.prompt_manager = PromptManager()
        
        # Initialize the selected model
        self.model_name = os.getenv('CHAT_MODEL', 'gemini').lower()
        logger.info(f"Initializing chat with model: {self.model_name}")
        
        if self.model_name == 'gemini':
            self._init_gemini()
        else:
            self._init_mistral()
            
    def _init_gemini(self):
        """Initialize Gemini model."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables")
                logger.warning("Falling back to Mistral model")
                self.model_name = 'mistral'
                self._init_mistral()
                return
            
            logger.info("Configuring Gemini...")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("✓ Gemini chat engine initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import google.generativeai: {e}")
            logger.warning("Falling back to Mistral model")
            self.model_name = 'mistral'
            self._init_mistral()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            logger.warning("Falling back to Mistral model")
            self.model_name = 'mistral'
            self._init_mistral()

    def _init_mistral(self):
        """Initialize Mistral model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Please install transformers and torch packages: pip install transformers torch")

        model_path = os.getenv('MISTRAL_MODEL_PATH')
        if not model_path or not Path(model_path).exists():
            raise ValueError(f"Invalid MISTRAL_MODEL_PATH: {model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def _generate_fallback_summary(self, results: List[Dict[str, Any]], query: str) -> str:
        """Generate a simple fallback summary when Gemini fails"""
        try:
            # Get total count
            count = len(results)
            if count == 0:
                return f"I couldn't find any tweets matching your search for '{query}'."
            
            # Get unique authors (up to 3)
            authors = list(set(result['author'] for result in results if 'author' in result))[:3]
            author_text = ""
            if authors:
                if len(authors) == 1:
                    author_text = f" from {authors[0]}"
                else:
                    author_text = f" from {', '.join(authors[:-1])} and {authors[-1]}"
            
            # Get categories
            categories = list(set(cat for result in results 
                                for cat in result.get('categories', [])))
            category_text = ""
            if categories:
                category_text = f" All tweets are categorized under {', '.join(categories)}."
            
            return f"I found {count} tweets about {query}{author_text}.{category_text}"
            
        except Exception as e:
            logger.error(f"Error generating fallback summary: {e}")
            return f"I found {len(results)} tweets matching your search."

    def _validate_results(self, results: List[Dict[str, Any]], query: str) -> bool:
        """Validate search results before generating response"""
        try:
            if not results:
                logger.info(f"No results found for query: {query}")
                return False
                
            # Verify result structure
            for result in results:
                if not all(k in result for k in ['text', 'author']):
                    logger.warning(f"Invalid result structure found: {result}")
                    return False
                    
            logger.info(f"Validated {len(results)} results for query: {query}")
            return True
        except Exception as e:
            logger.error(f"Error validating results: {e}")
            return False

    def _build_anti_hallucination_prompt(self, results: List[Dict[str, Any]], query: str) -> str:
        """Build a strong anti-hallucination prompt with result validation"""
        result_count = len(results)
        authors = [r['author'] for r in results if 'author' in r]
        
        # Build tweet content section
        tweet_content = "\n\n".join([
            f"Tweet by {r['author']}:\n{r['text']}"
            for r in results
        ])
        
        return f"""
        CRITICAL INSTRUCTION: You are summarizing {result_count} specific tweets about {query}.
        
        Here are the tweets to summarize:
        
        {tweet_content}
        
        Instructions for summarization:
        - ONLY reference information from the tweets shown above
        - DO NOT make up or infer any information not present in the tweets
        - If unsure about any detail, exclude it
        - Clearly indicate when you're quoting or referencing a specific tweet
        - Maintain a natural, conversational tone while being accurate
        
        Remember: Accuracy is more important than completeness.
        """

    def _should_perform_search(self, intent: Intent) -> bool:
        """
        Determine if a search should be performed based on the intent.
        """
        logger.info(f"Checking if search needed for intent type: {intent.type}")
        
        # Always perform search for SEARCH and FOLLOWUP intents
        if intent.type in [IntentType.SEARCH, IntentType.FOLLOWUP]:
            logger.info("Search required for SEARCH/FOLLOWUP intent")
            return True
            
        logger.info(f"No search needed for intent type: {intent.type}")
        return False

    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        """
        # Get intent
        intent = self.intent_classifier.analyze(
            message, 
            self.conversation_manager.get_context().get('recent_messages', []),
            None
        )
        logger.info(f"Processed intent: {intent.type}")
        
        try:
            # Handle search/followup intents
            if self._should_perform_search(intent):
                logger.info("Performing search for message")
                search_results = self._handle_search(message)
                
                if not search_results:
                    return {
                        'text': "I couldn't find any relevant tweets for that query.",
                        'model': self.model_name,
                        'bookmarks_used': []
                    }
                    
                try:
                    # Generate summary with anti-hallucination prompt
                    prompt = self._build_anti_hallucination_prompt(search_results, message.replace('find tweets about ', '').strip())
                    response_text = self._generate_response(prompt)
                    
                    return {
                        'text': response_text,
                        'model': self.model_name,
                        'bookmarks_used': search_results
                    }
                    
                except Exception as e:
                    logger.error(f"Error generating summary: {e}")
                    fallback_text = self._generate_fallback_summary(search_results, message.replace('find tweets about ', '').strip())
                    return {
                        'text': fallback_text,
                        'model': self.model_name,
                        'bookmarks_used': search_results
                    }
            
            # Handle general conversation
            response_text = self._generate_response(message)
            return {
                'text': response_text,
                'model': self.model_name,
                'bookmarks_used': []
            }
            
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
            return {
                'text': "I encountered an error processing your message. Please try again.",
                'model': self.model_name,
                'bookmarks_used': []
            }
    
    def _handle_search(self, query: str) -> List[Dict]:
        """Handle search requests using the chat-specific search engine"""
        try:
            # Log the search attempt
            logger.info(f"Performing search for query: {query}")
            
            context = {
                'topic': self.conversation_manager.current_topic,
                'recent_categories': self.conversation_manager.get_recent_categories()
            }
            
            # Try chat-specific search first
            results = self.chat_search_engine.search(query=query, context=context)
            
            # Log the results
            logger.info(f"Search results found: {len(results) if results else 0}")
            
            if not results:
                # Try fallback to main search engine
                logger.info("No results from chat search, trying main search engine")
                results = self.main_search_engine.search(query=query, limit=5)
            
            return results if results else []
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    def _generate_gemini_response(self, prompt: str) -> str:
        """Generate a response using Gemini."""
        try:
            # Gemini's generate_content is synchronous
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini response error: {e}")
            raise

    def _generate_mistral_response(self, prompt: str) -> str:
        """Generate a response using Mistral."""
        try:
            # Mistral's generate is synchronous
            response = self.model.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"Mistral response error: {e}")
            raise

    def _generate_response(self, prompt: str) -> str:
        """Generate a response using the selected model."""
        try:
            if self.model_name == 'gemini':
                response = self._generate_gemini_response(prompt)
            else:
                response = self._generate_mistral_response(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response with {self.model_name}: {e}")
            raise
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.conversation_manager.reset()
        if self.model_name == 'gemini':
            self.chat = self.model.start_chat(history=[])
