from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from .intent_classifier import Intent, IntentType

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[Intent] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

@dataclass
class ConversationContext:
    topic: Optional[str] = None
    reference_id: Optional[str] = None
    last_search_query: Optional[str] = None
    last_search_results: List[Dict] = field(default_factory=list)
    bookmark_context: bool = False
    continuation_count: int = 0

class ConversationManager:
    """
    Manages conversation state and context without persistence.
    Maintains a short-term memory of recent messages and context
    for natural conversation flow.
    """
    
    def __init__(self, context_window: int = 5, expiry_minutes: int = 30):
        self.messages = deque(maxlen=context_window)
        self.current_topic: Optional[str] = None
        self.context: Dict = {}
        self.expiry_minutes = expiry_minutes
        self.last_activity = datetime.now()
        self.max_history = 10
        self.context_obj = ConversationContext()
    
    def add_message(self, role: str, content: str, intent: Optional[Intent] = None, metadata: Dict = None) -> None:
        """Add a new message to the conversation history."""
        message = Message(
            role=role,
            content=content,
            intent=intent,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # Maintain max history
        if len(self.messages) > self.max_history:
            self.messages.popleft()
        
        # Update context based on the new message
        if role == 'user':
            self._update_context(content, metadata)
        
        # Update conversation context
        self._update_conversation_context(message)

    def get_recent_messages(self, count: Optional[int] = None) -> List[Message]:
        """Get the most recent messages, optionally limited to count."""
        if self._is_expired():
            self.reset()
            return []
            
        messages = list(self.messages)
        if count:
            messages = messages[-count:]
        return messages
    
    def get_context(self) -> Dict:
        """Get the current conversation context."""
        if self._is_expired():
            self.reset()
            return {}
            
        return {
            'topic': self.current_topic,
            'recent_messages': [
                f"{msg.role}: {msg.content}" 
                for msg in self.get_recent_messages()
            ],
            'context': self.context,
            'last_activity': self.last_activity.isoformat()
        }
    
    def _update_context(self, message: str, metadata: Dict) -> None:
        """Update conversation context based on new message."""
        # Update topic if provided in metadata
        if metadata and 'topic' in metadata:
            self.current_topic = metadata['topic']
            
        # Update context with any additional metadata
        self.context.update(metadata or {})
        
        # Remove expired context
        self._clean_expired_context()
    
    def _is_expired(self) -> bool:
        """Check if the conversation has expired."""
        expiry_time = datetime.now() - timedelta(minutes=self.expiry_minutes)
        return self.last_activity < expiry_time
    
    def _clean_expired_context(self) -> None:
        """Remove expired items from context."""
        now = datetime.now()
        self.context = {
            k: v for k, v in self.context.items()
            if not (isinstance(v, dict) and 
                   'expiry' in v and 
                   datetime.fromisoformat(v['expiry']) < now)
        }
    
    def reset(self) -> None:
        """Reset the conversation state."""
        self.messages.clear()
        self.current_topic = None
        self.context = {}
        self.last_activity = datetime.now()
        self.context_obj = ConversationContext()

    def update_search_context(self, 
                            query: str, 
                            results: List[Dict],
                            expiry_minutes: int = 5) -> None:
        """Update context with search-related information."""
        expiry = datetime.now() + timedelta(minutes=expiry_minutes)
        self.context['last_search'] = {
            'query': query,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'expiry': expiry.isoformat()
        }
    
    def get_search_context(self) -> Optional[Dict]:
        """Get the most recent search context if not expired."""
        search_context = self.context.get('last_search')
        if not search_context:
            return None
            
        expiry = datetime.fromisoformat(search_context['expiry'])
        if expiry < datetime.now():
            del self.context['last_search']
            return None
            
        return search_context

    def _update_conversation_context(self, message: Message) -> None:
        """Update conversation context based on new message."""
        if not message.intent:
            return

        # Update topic tracking
        if message.intent.type == IntentType.SEARCH:
            self.current_topic = message.content
            self.context_obj.topic = message.content
            self.context_obj.bookmark_context = True
            self.context_obj.continuation_count = 0
        elif message.intent.type == IntentType.FOLLOWUP:
            self.context_obj.continuation_count += 1
        elif message.intent.type == IntentType.GENERAL_CHAT:
            if not self._is_related_to_previous(message.content):
                self.current_topic = None
                self.context_obj.topic = None
                self.context_obj.bookmark_context = False
                self.context_obj.continuation_count = 0

    def _is_related_to_previous(self, message: str) -> bool:
        """
        Determine if the message is related to the previous conversation topic.
        """
        if not self.current_topic:
            return False

        # Check if message contains words from the current topic
        topic_words = set(self.current_topic.lower().split())
        message_words = set(message.lower().split())
        
        # Calculate word overlap
        overlap = topic_words.intersection(message_words)
        return len(overlap) > 0

    def should_search(self, intent: Intent) -> bool:
        """
        Determine if we should perform a search based on intent and context.
        """
        if intent.requires_search:
            return True
            
        if intent.type == IntentType.FOLLOWUP and self.context_obj.bookmark_context:
            return True
            
        return False

    def update_search_results(self, query: str, results: List[Dict]) -> None:
        """Update the context with new search results."""
        self.context_obj.last_search_query = query
        self.context_obj.last_search_results = results

    def get_conversation_summary(self) -> str:
        """
        Get a summary of the current conversation state.
        Useful for debugging and prompt construction.
        """
        return {
            'topic': self.current_topic,
            'messages_count': len(self.messages),
            'in_bookmark_context': self.context_obj.bookmark_context,
            'continuation_depth': self.context_obj.continuation_count
        }

    def get_recent_categories(self) -> List[str]:
        """Get categories from recent conversation context."""
        categories = set()
        
        # Look through recent messages for category information
        for message in self.messages:
            if message.context and 'categories' in message.context:
                categories.update(message.context['categories'])
                
        # Also check current context
        if self.context and 'categories' in self.context:
            categories.update(self.context['categories'])
            
        return list(categories) 