"""
Intent classification for the PythonAnywhere chat implementation.
This module analyzes user messages to determine their intent.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of user intents that can be detected."""
    SEARCH = auto()      # User wants to search for bookmarks
    OPINION = auto()     # User wants an opinion or analysis
    FOLLOWUP = auto()    # User is following up on a previous query
    GENERAL = auto()     # General chat/conversation
    HELP = auto()        # User is asking for help with the system

@dataclass
class Intent:
    """Represents a classified user intent with parameters."""
    type: IntentType
    confidence: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

class IntentClassifierPA:
    """
    Classifies user intents based on message content and context.
    Uses rule-based classification to determine intent type.
    """
    
    def __init__(self):
        """Initialize the intent classifier."""
        logger.info("✓ Intent classifier initialized")
        
        # Define search-related keywords
        self.search_keywords = [
            'find', 'search', 'look for', 'looking for', 'show me', 
            'get', 'retrieve', 'bookmarks about', 'bookmarks on',
            'related to', 'tweets about', 'tweets on', 'locate',
            'discover', 'any bookmarks', 'have bookmarks', 'saved bookmarks'
        ]
        
        # Define opinion-related keywords
        self.opinion_keywords = [
            'what do you think', 'your thoughts', 'your opinion',
            'analyze', 'explain why', 'why is', 'thoughts on',
            'elaborate on', 'can you explain', 'reasons for',
            'interpretation', 'your perspective', 'summarize'
        ]
        
        # Define help-related keywords
        self.help_keywords = [
            'help', 'how do i', 'how to', 'how can i', 'what can you do',
            'instructions', 'guide', 'assistance', 'stuck', 'confused',
            'not working', 'don\'t understand', 'functionality', 'features'
        ]
        
        # Define followup keywords and patterns
        self.followup_keywords = [
            'more', 'another', 'additional', 'else', 'other',
            'similar', 'related', 'also', 'too', 'as well',
            'further', 'expand', 'continue', 'furthermore'
        ]
        
        # Compile regex patterns for efficiency
        self.search_patterns = self._compile_patterns(self.search_keywords)
        self.opinion_patterns = self._compile_patterns(self.opinion_keywords)
        self.help_patterns = self._compile_patterns(self.help_keywords)
        self.followup_patterns = self._compile_patterns(self.followup_keywords)
        
        # Compile question patterns
        self.question_pattern = re.compile(r'\b(who|what|when|where|why|how|is|are|can|could|would|should|do|does|did|will)\b', re.IGNORECASE)
        
    def _compile_patterns(self, keywords: List[str]) -> List[re.Pattern]:
        """
        Compile regex patterns for the keywords.
        
        Args:
            keywords: List of keyword strings
            
        Returns:
            List of compiled regex patterns
        """
        patterns = []
        for keyword in keywords:
            # Replace spaces with \s+ to match flexible whitespace
            keyword_regex = keyword.replace(' ', r'\s+')
            # Create word boundary pattern for exact matching
            pattern = re.compile(r'\b' + keyword_regex + r'\b', re.IGNORECASE)
            patterns.append(pattern)
        return patterns
        
    def analyze(self, message: str, recent_messages: List[Dict[str, Any]], 
                context: Dict[str, Any]) -> Intent:
        """
        Analyze a message to determine the user's intent.
        
        Args:
            message: The user's message
            recent_messages: Recent conversation messages for context
            context: Additional context information
            
        Returns:
            An Intent object describing the detected intent
        """
        # First, check if this might be a followup question
        if self._is_followup(message, recent_messages, context):
            query = self._extract_query(message)
            logger.info(f"Classified as FOLLOWUP intent: '{message[:30]}...'")
            return Intent(
                type=IntentType.FOLLOWUP,
                confidence=0.8,
                params={'query': query}
            )
        
        # Next, check for explicit search intent
        if self._is_search(message):
            query = self._extract_query(message)
            logger.info(f"Classified as SEARCH intent: '{message[:30]}...'")
            return Intent(
                type=IntentType.SEARCH,
                confidence=0.9,
                params={'query': query}
            )
            
        # Check for explicit opinion/analysis intent
        if self._is_opinion(message):
            query = self._extract_query(message)
            logger.info(f"Classified as OPINION intent: '{message[:30]}...'")
            return Intent(
                type=IntentType.OPINION,
                confidence=0.8,
                params={'query': query}
            )
            
        # Check for help intent
        if self._is_help(message):
            logger.info(f"Classified as HELP intent: '{message[:30]}...'")
            return Intent(
                type=IntentType.HELP,
                confidence=0.9,
                params={}
            )
            
        # If message contains a question, treat as search by default
        if self._contains_question(message):
            query = message
            logger.info(f"Classified as implicit SEARCH intent (question): '{message[:30]}...'")
            return Intent(
                type=IntentType.SEARCH,
                confidence=0.7,
                params={'query': query}
            )
            
        # Default to general conversation
        logger.info(f"Classified as GENERAL intent (default): '{message[:30]}...'")
        return Intent(
            type=IntentType.GENERAL,
            confidence=0.6,
            params={}
        )
        
    def _is_search(self, message: str) -> bool:
        """
        Check if the message indicates a search intent.
        
        Args:
            message: The user's message
            
        Returns:
            True if the message indicates a search intent
        """
        # Check against search patterns
        for pattern in self.search_patterns:
            if pattern.search(message):
                return True
                
        # Additional heuristic: Check for bookmark-related keywords
        if re.search(r'\b(bookmark|bookmarks|tweet|tweets)\b', message, re.IGNORECASE):
            # If they're asking about bookmarks/tweets directly, 
            # it's likely a search intent
            return True
                
        return False
        
    def _is_opinion(self, message: str) -> bool:
        """
        Check if the message is asking for an opinion or analysis.
        
        Args:
            message: The user's message
            
        Returns:
            True if the message is asking for an opinion
        """
        # Check against opinion patterns
        for pattern in self.opinion_patterns:
            if pattern.search(message):
                return True
                
        # Check for opinion-related sentence structures
        if re.search(r'\bwhy (is|are|do|does|did)\b', message, re.IGNORECASE):
            return True
            
        return False
        
    def _is_help(self, message: str) -> bool:
        """
        Check if the message is asking for help with the system.
        
        Args:
            message: The user's message
            
        Returns:
            True if the message is asking for help
        """
        # Check against help patterns
        for pattern in self.help_patterns:
            if pattern.search(message):
                return True
                
        # Check for help-related sentence structures
        if re.search(r'\bhow (can|do) (i|you|we)\b', message, re.IGNORECASE):
            return True
            
        return False
        
    def _is_followup(self, message: str, recent_messages: List[Dict[str, Any]], 
                   context: Dict[str, Any]) -> bool:
        """
        Check if the message is a followup to a previous query.
        
        Args:
            message: The user's message
            recent_messages: Recent conversation messages
            context: Additional context information
            
        Returns:
            True if the message appears to be a followup
        """
        # Skip if this is the first message
        if not recent_messages:
            return False
            
        # Check for short queries (likely followups)
        if len(message.split()) <= 3 and self._contains_question(message):
            return True
            
        # Check against followup patterns
        for pattern in self.followup_patterns:
            if pattern.search(message):
                return True
                
        # Check for pronouns that might refer to previous results
        pronoun_pattern = re.compile(r'\b(it|they|them|these|those|that|this)\b', re.IGNORECASE)
        if pronoun_pattern.search(message):
            # Look for question indicators as well to confirm it's a followup query
            if self._contains_question(message):
                return True
                
        # If there was a recent search and this message is short, 
        # it's likely a refinement
        if context.get('recent_searches') and len(message.split()) < 5:
            return True
            
        return False
        
    def _contains_question(self, message: str) -> bool:
        """
        Check if the message contains a question.
        
        Args:
            message: The user's message
            
        Returns:
            True if the message contains a question
        """
        # Check for question marks
        if '?' in message:
            return True
            
        # Check for question words/patterns
        return bool(self.question_pattern.search(message))
        
    def _extract_query(self, message: str) -> str:
        """
        Extract the search query from a message.
        
        Args:
            message: The user's message
            
        Returns:
            The extracted search query
        """
        # For now, just use the whole message as the query
        # In the future, this could be made more sophisticated
        return message
