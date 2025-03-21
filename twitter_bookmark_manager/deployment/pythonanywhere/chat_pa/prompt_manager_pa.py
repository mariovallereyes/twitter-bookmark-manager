"""
Prompt manager for the PythonAnywhere chat implementation.
This module manages prompt templates for interacting with LLMs.
"""

import logging
from typing import Dict, Any, List, Optional
import datetime

logger = logging.getLogger(__name__)

class PromptManagerPA:
    """
    Manages prompt templates and generation for the chat system.
    Optimized for PythonAnywhere deployment.
    """
    
    def __init__(self):
        """Initialize with standard prompt templates."""
        self.templates = {
            # System prompt for RAG conversation
            "system_rag": """You are a helpful AI assistant with access to the user's Twitter bookmarks database. 
Answer questions based on the provided bookmarks when relevant.
Only reference bookmarks that are explicitly provided to you. 
Never make up or hallucinate bookmark content.
If you're not sure about something, acknowledge the uncertainty.
For general knowledge questions not related to bookmarks, provide helpful and accurate responses.
Be conversational and engaging, but prioritize accuracy over style.""",
            
            # Prompt for generating responses with bookmark context
            "rag_response": """<SYSTEM>
You are a helpful AI assistant with access to the user's Twitter bookmarks database.
When answering, only use information from the provided bookmarks. Do not make up or hallucinate any bookmark content.
These bookmarks are the user's personal collection and they want accurate information from them.
</SYSTEM>

<BOOKMARK_CONTEXT>
{bookmark_context}
</BOOKMARK_CONTEXT>

<USER_QUERY>
{user_query}
</USER_QUERY>

<INSTRUCTIONS>
1. Use only the information in the BOOKMARK_CONTEXT section to answer the user's query.
2. If the bookmarks don't contain enough information to fully answer, clearly state what you can and cannot answer.
3. Write in a helpful, clear, and conversational tone - like a friendly assistant giving information to a friend.
4. IMPORTANT: Use natural language and conversational flow. Avoid using bullet points unless absolutely necessary.
5. Integrate mentions of bookmarks naturally into your response rather than listing them separately.
6. If citing specific bookmarks, you can mention the author or related context, but do so in a natural way.
7. Don't repeat all the bookmarks verbatim - synthesize a helpful response while maintaining accuracy.
</INSTRUCTIONS>

Based on the Twitter bookmarks provided, here's my response:""",
            
            # Prompt for general conversation without bookmarks
            "general_conversation": """<SYSTEM>
You are a helpful AI assistant having a conversation with a user.
Provide thoughtful, accurate, and engaging responses to their questions.
</SYSTEM>

<USER_QUERY>
{user_query}
</USER_QUERY>

<CONVERSATION_HISTORY>
{conversation_history}
</CONVERSATION_HISTORY>

<INSTRUCTIONS>
1. Respond in a helpful, concise, and conversational manner.
2. If the question requires specialized knowledge beyond your capabilities, acknowledge your limitations.
3. Stay on topic and provide the most accurate information you can.
4. If the query relates to Twitter bookmarks, politely explain that you don't have access to the user's bookmarks 
   for this specific question, but you're happy to answer based on your general knowledge.
</INSTRUCTIONS>

Based on the conversation, here's my response:""",
            
            # Prompt for summarizing search results
            "search_summary": """<SYSTEM>
You are summarizing Twitter bookmarks that were retrieved based on the user's search.
</SYSTEM>

<SEARCH_QUERY>
{search_query}
</SEARCH_QUERY>

<SEARCH_RESULTS>
{search_results}
</SEARCH_RESULTS>

<INSTRUCTIONS>
1. Summarize the key information from the search results in a helpful and concise way.
2. Only include information that is actually present in the search results.
3. If the search results are limited or don't directly address the query, acknowledge this.
4. Mention the authors of the tweets when relevant.
5. Organize the information in a logical way that helps the user understand the content.
6. Use natural, conversational language rather than bullet points or lists.
</INSTRUCTIONS>

Here's a summary of the Twitter bookmarks related to your search:"""
        }
        
        logger.info("✓ Prompt manager initialized")
    
    def get_prompt(self, prompt_type: str, params: Dict[str, Any]) -> str:
        """
        Get a formatted prompt by type with the provided parameters.
        
        Args:
            prompt_type: Type of prompt to retrieve
            params: Dictionary of parameters to fill in the prompt template
            
        Returns:
            Formatted prompt string
        """
        if prompt_type not in self.templates:
            logger.warning(f"Unknown prompt type: {prompt_type}, using general_conversation")
            prompt_type = "general_conversation"
            
        try:
            prompt_template = self.templates[prompt_type]
            return prompt_template.format(**params)
        except KeyError as e:
            logger.error(f"Missing parameter in prompt: {e}")
            # Fall back to a simpler prompt that doesn't require the missing parameter
            fallback_prompt = f"Please respond to: {params.get('user_query', 'the user query')}"
            return fallback_prompt
            
    def format_bookmark_context(self, bookmarks: List[Dict[str, Any]]) -> str:
        """
        Format a list of bookmarks into a context string for the LLM.
        
        Args:
            bookmarks: List of bookmark dictionaries
            
        Returns:
            Formatted string with bookmark information
        """
        if not bookmarks:
            return "No relevant bookmarks found."
            
        context = []
        for i, bookmark in enumerate(bookmarks, 1):
            # Extract fields with fallbacks for different structures
            username = bookmark.get('username', bookmark.get('author_username', 'Unknown')).strip('@')
            title = bookmark.get('title', '')
            description = bookmark.get('description', bookmark.get('text', 'No content'))
            date_str = bookmark.get('created_at', 'Unknown date')
            
            # Try to parse the date if it's an ISO string
            date = date_str
            if isinstance(date_str, str):
                try:
                    # Convert date to more readable format if possible
                    date_obj = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    date = date_obj.strftime('%B %d, %Y')
                except (ValueError, TypeError):
                    pass
                    
            # Get categories or tags
            categories = bookmark.get('categories', [])
            categories_str = ", ".join(categories) if categories else "Uncategorized"
            
            # Format entry with available data
            entry = f"BOOKMARK {i}:\n"
            entry += f"Username: @{username}\n"
            
            if title:
                entry += f"Title: {title}\n"
                
            entry += f"Content: {description}\n"
            entry += f"Date: {date}\n"
            
            if categories:
                entry += f"Categories: {categories_str}\n"
                
            # Add URL if we have it, either directly or constructed from ID
            url = bookmark.get('url', '')
            if not url and username and bookmark.get('id'):
                url = f"https://twitter.com/{username}/status/{bookmark.get('id')}"
            
            if url:
                entry += f"URL: {url}\n"
            
            context.append(entry)
            
        return "\n\n".join(context)
        
    def format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history for context.
        
        Args:
            history: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted conversation history
        """
        if not history:
            return "No previous conversation."
            
        formatted = []
        for msg in history:
            role = msg.get('role', '').upper()
            content = msg.get('content', '')
            formatted.append(f"{role}: {content}")
            
        return "\n\n".join(formatted)
