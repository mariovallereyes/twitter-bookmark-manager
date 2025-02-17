import logging
from typing import List, Dict, Any, Optional
from database.db import get_db_session, get_vector_store
from database.models import Bookmark, Conversation
from core.search import BookmarkSearch
from datetime import datetime
import json
import uuid
from llama_cpp import Llama
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookmarkRAG:
    def __init__(self):
        # Get absolute path to models directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        
        self.search = BookmarkSearch()
        self.memory_limit = None
        self.vector_store = get_vector_store()
        
        # Initialize Mistral (using llama-cpp-python)
        logger.info(f"Initializing Mistral with model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4
        )
        logger.info("Mistral initialized successfully")

    def load_conversation_history(self, 
                                limit: Optional[int] = None,
                                include_archived: bool = False) -> List[Dict]:
        """
        Load conversation history from database
        
        Args:
            limit: Optional limit on number of conversations to load
            include_archived: Whether to include archived conversations
            
        Returns:
            List of conversation records
        """
        try:
            with get_db_session() as session:
                query = session.query(Conversation)
                
                if not include_archived:
                    query = query.filter(Conversation.is_archived == False)
                
                if limit:
                    query = query.order_by(Conversation.timestamp.desc()).limit(limit)
                else:
                    query = query.order_by(Conversation.timestamp.desc())
                
                return query.all()
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            raise

    def _calculate_importance(self, 
                            response: Dict[str, Any],
                            n_bookmarks_used: int) -> float:
        """
        Calculate importance score for a conversation
        
        Args:
            response: The system's response
            n_bookmarks_used: Number of bookmarks referenced
            
        Returns:
            Importance score (0-1)
        """
        try:
            factors = {
                'n_bookmarks': min(n_bookmarks_used / 10, 1.0),  # More bookmarks = more important
                'response_length': min(len(response['text']) / 1000, 1.0),  # Longer responses might be more important
                'confidence': response.get('confidence', 0.5)  # System's confidence in response
            }
            
            return sum(factors.values()) / len(factors)
        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            return 0.5  # Default middle importance

    def _get_relevant_context(self, 
                            user_input: str,
                            conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Get relevant bookmarks considering full conversation history
        
        Args:
            user_input: Current user query
            conversation_history: Optional loaded conversation history
            
        Returns:
            List of relevant bookmarks with metadata
        """
        try:
            # Get current relevant bookmarks
            current_results = self.search.semantic_search(
                query=user_input,
                n_results=5
            )
            
            if not conversation_history:
                return current_results
            
            # Get historical bookmarks
            historical_bookmark_ids = set()
            for conv in conversation_history[-5:]:  # Last 5 conversations
                if conv.bookmarks_used:
                    historical_bookmark_ids.update(
                        b['id'] for b in conv.bookmarks_used
                    )
            
            # Add historical bookmarks not in current results
            current_ids = {r['bookmark_id'] for r in current_results}
            additional_bookmarks = []
            
            with get_db_session() as session:
                for bookmark_id in historical_bookmark_ids:
                    if bookmark_id not in current_ids:
                        bookmark = session.query(Bookmark).get(bookmark_id)
                        if bookmark:
                            additional_bookmarks.append({
                                'bookmark_id': bookmark_id,
                                'text': bookmark.text,
                                'url': bookmark.url,
                                'created_at': bookmark.created_at,
                                'category': bookmark.category,
                                'from_history': True
                            })
            
            return current_results + additional_bookmarks
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            raise

    def chat(self, 
             user_input: str,
             temperature: float = 0.7,
             max_tokens: int = 500,
             conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main chat function with persistent memory
        
        Args:
            user_input: User's question or command
            temperature: Controls response creativity (0.0-1.0)
            max_tokens: Maximum response length
            conversation_id: Optional ID to group related exchanges
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Load relevant history
            history = self.load_conversation_history(limit=10)  # Last 10 conversations
            
            # Get relevant bookmarks with context
            relevant_bookmarks = self._get_relevant_context(
                user_input=user_input,
                conversation_history=history
            )
            
            # Prepare system prompt
            system_prompt = self._prepare_system_prompt(
                user_input=user_input,
                relevant_bookmarks=relevant_bookmarks,
                conversation_history=history
            )
            
            # Generate response
            response = self._generate_response(
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Calculate importance
            importance_score = self._calculate_importance(
                response=response,
                n_bookmarks_used=len(relevant_bookmarks)
            )
            
            # Save to database
            with get_db_session() as session:
                conversation = Conversation(
                    conversation_id=conversation_id or str(uuid.uuid4()),
                    user_input=user_input,
                    system_response=response,
                    bookmarks_used=[{
                        'id': b['bookmark_id'],
                        'relevance': b.get('similarity', 0)
                    } for b in relevant_bookmarks],
                    importance_score=importance_score,
                    last_accessed=datetime.utcnow()
                )
                session.add(conversation)
                session.commit()
            
            return {
                'response': response,
                'sources': [b['url'] for b in relevant_bookmarks],
                'context_used': len(relevant_bookmarks),
                'conversation_id': conversation_id,
                'metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'bookmarks_referenced': len(relevant_bookmarks),
                    'historical_context_used': any(b.get('from_history', False) 
                                                for b in relevant_bookmarks),
                    'importance_score': importance_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise

    def _prepare_system_prompt(self,
                             user_input: str,
                             relevant_bookmarks: List[Dict[str, Any]],
                             conversation_history: List[Dict]) -> str:
        """
        Prepare system prompt with context from bookmarks and history
        """
        # Format bookmark contexts
        bookmark_contexts = []
        for bookmark in relevant_bookmarks:
            context = f"""
            BOOKMARK: {bookmark['text']}
            URL: {bookmark['url']}
            SAVED: {bookmark['created_at']}
            CATEGORY: {bookmark.get('category', 'Uncategorized')}
            """
            bookmark_contexts.append(context)
        
        # Format conversation history
        history_context = []
        for conv in conversation_history[-3:]:  # Last 3 conversations
            history_context.append(f"""
            USER: {conv.user_input}
            ASSISTANT: {conv.system_response.get('text', '')}
            """)
        
        # Create the system prompt
        system_prompt = f"""
        You are a helpful assistant with access to the user's Twitter bookmarks.
        
        CURRENT QUESTION: {user_input}
        
        RELEVANT BOOKMARKS:
        {' '.join(bookmark_contexts)}
        
        RECENT CONVERSATION HISTORY:
        {' '.join(history_context)}
        
        Please provide a detailed response based on the bookmarks, maintaining context
        from our conversation. Include relevant URLs when appropriate.
        """
        
        return system_prompt

    def manage_memory(self, 
                     action: str,
                     **kwargs) -> Dict[str, Any]:
        """
        User control over conversation memory
        
        Args:
            action: One of ['archive', 'restore', 'stats']
            **kwargs: Action-specific parameters
            
        Returns:
            Dict containing action results
        """
        try:
            if action == 'archive':
                before_date = kwargs.get('before_date')
                with get_db_session() as session:
                    query = session.query(Conversation)
                    if before_date:
                        query = query.filter(Conversation.timestamp < before_date)
                    count = query.update({Conversation.is_archived: True})
                    session.commit()
                    return {'archived_count': count}
            
            elif action == 'restore':
                conversation_ids = kwargs.get('conversation_ids', [])
                with get_db_session() as session:
                    count = session.query(Conversation)\
                                 .filter(Conversation.conversation_id.in_(conversation_ids))\
                                 .update({Conversation.is_archived: False})
                    session.commit()
                    return {'restored_count': count}
            
            elif action == 'stats':
                with get_db_session() as session:
                    total = session.query(Conversation).count()
                    archived = session.query(Conversation)\
                                    .filter(Conversation.is_archived == True).count()
                    active = total - archived
                    return {
                        'total_conversations': total,
                        'active_conversations': active,
                        'archived_conversations': archived
                    }
            
            else:
                raise ValueError(f"Invalid action. Choose from: ['archive', 'restore', 'stats']")
                
        except Exception as e:
            logger.error(f"Error managing memory: {e}")
            raise

    def _generate_response(self,
                          system_prompt: str,
                          temperature: float = 0.7,
                          max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate response using Mistral (Llama.cpp)
        """
        try:
            formatted_prompt = f"""[INST] <<SYS>>
You are a helpful assistant that helps users find and understand their Twitter bookmarks.
<</SYS>>

{system_prompt}[/INST]"""

            # Generate response with enforced JSON format
            response = self.llm.create_completion(
                prompt=formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["[INST]", "</s>"],
                response_format={"type": "json_object"}  # 🔹 This ensures the response is always a dictionary
            )

            # 🔹 Debugging: Print the full raw response
            print("\n🔹 FULL RAW RESPONSE FROM LLM:", response)
            print("🔹 TYPE OF RESPONSE:", type(response))

            # Ensure response is a dictionary
            if not isinstance(response, dict):
                print("❌ ERROR: Unexpected response format!", response)
                return {"text": "ERROR: Unexpected response format"}

            # Ensure "choices" exists and extract text
            choices = response.get("choices", [])
            if not isinstance(choices, list) or not choices:
                print("❌ ERROR: Missing or invalid 'choices' key!", response)
                return {"text": "ERROR: No valid choices"}

            # Extract actual response text
            generated_text = choices[0].get("text", "").strip()

            return {
                "text": generated_text,
                "confidence": 0.9,
                "model_info": {
                    "name": "mistral",
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            }

        except Exception as e:
            print(f"❌ ERROR in _generate_response: {str(e)}")
            return {"text": f"ERROR: {str(e)}"}