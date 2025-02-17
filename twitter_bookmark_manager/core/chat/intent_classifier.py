from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

class IntentType(Enum):
    SEARCH = auto()
    CONVERSATION = auto()
    FOLLOWUP = auto()
    CLARIFICATION = auto()
    GENERAL_CHAT = auto()

@dataclass
class Intent:
    type: IntentType
    confidence: float
    entities: List[str] = None
    context: Dict = None
    reference: Optional[str] = None
    requires_search: bool = False

class IntentClassifier:
    """
    Sophisticated intent classification system that determines the true intent
    of user messages in a conversational context.
    """
    def __init__(self):
        self.conversation_markers = {
            'followup': [
                'what about', 'tell me more', 'and then', 'what else',
                'continue', 'go on', 'elaborate', 'explain further'
            ],
            'clarification': [
                'what do you mean', 'can you explain', 'i don\'t understand',
                'could you clarify', 'what\'s that', 'what is'
            ],
            'search_implicit': [
                'interested in', 'looking for', 'want to know', 'curious about',
                'any thoughts on', 'what do you think about', 'tell me about'
            ]
        }

    def analyze(self, 
                message: str, 
                conversation_history: List[str],
                last_intent: Optional[Intent] = None) -> Intent:
        """
        Analyze the message to determine its intent, considering conversation history
        and the last known intent.
        """
        message = message.lower().strip()
        logger.info(f"Analyzing message: {message}")
        
        # Check for explicit search indicators first
        if any(marker in message for marker in [
            'search', 'find', 'show me', 'look up', 'get me'
        ]):
            logger.info("Detected explicit search intent")
            return Intent(
                type=IntentType.SEARCH,
                confidence=0.9,
                requires_search=True
            )

        # Enhanced follow-up detection
        followup_markers = [
            'what about', 'tell me more', 'and then', 'what else',
            'continue', 'go on', 'elaborate', 'explain further',
            'how about', 'what of', 'any', 'other'
        ]
        
        if any(marker in message for marker in followup_markers):
            logger.info(f"Detected follow-up intent with marker in: {message}")
            return Intent(
                type=IntentType.FOLLOWUP,
                confidence=0.85,
                reference=last_intent.type.name if last_intent else None,
                requires_search=True  # Important: Mark follow-ups as requiring search
            )

        # Check for clarification requests
        if any(marker in message for marker in self.conversation_markers['clarification']):
            return Intent(
                type=IntentType.CLARIFICATION,
                confidence=0.8,
                reference=last_intent.type.name if last_intent else None
            )

        # Check for implicit search indicators
        if any(marker in message for marker in self.conversation_markers['search_implicit']):
            return Intent(
                type=IntentType.SEARCH,
                confidence=0.7,
                requires_search=True
            )

        # If we have conversation history, analyze the flow
        if conversation_history:
            if self._is_continuing_conversation(message, conversation_history):
                return Intent(
                    type=IntentType.CONVERSATION,
                    confidence=0.75,
                    context={'continuing_topic': True}
                )

        # Default to general chat if no other intent is strongly indicated
        return Intent(
            type=IntentType.GENERAL_CHAT,
            confidence=0.6
        )

    def _is_continuing_conversation(self, message: str, history: List[str]) -> bool:
        """
        Determine if the message is continuing the current conversation topic.
        """
        if not history:
            return False

        # Check for topic continuity markers
        continuity_markers = [
            'it', 'that', 'this', 'those', 'these', 'they', 'them',
            'the', 'your', 'you', 'about', 'regarding'
        ]
        
        return any(marker in message.lower() for marker in continuity_markers)

    def extract_entities(self, message: str) -> List[str]:
        """
        Extract relevant entities from the message that might be useful
        for search or context.
        """
        # Simple entity extraction for now
        # This could be enhanced with NER models if needed
        words = message.lower().split()
        entities = []
        
        # Extract potential entities (nouns and proper nouns)
        current_entity = []
        for word in words:
            if word[0].isupper() or word in ['ai', 'ml', 'nlp']:
                current_entity.append(word)
            elif current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity))
        
        return entities 