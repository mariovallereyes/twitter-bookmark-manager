from typing import Dict, List, Optional
from .intent_classifier import Intent, IntentType

class PromptManager:
    """
    Manages prompt generation and optimization for Gemini 2.0.
    Focuses on creating natural, contextual prompts that leverage
    Gemini's advanced capabilities.
    """
    
    def __init__(self):
        self.system_prompt = """You are an intelligent and friendly AI assistant specialized in managing Twitter bookmarks while being capable of engaging in natural conversations about any topic. Your responses should be:
1. Natural and conversational, as if chatting with a knowledgeable friend
2. Contextually aware, maintaining conversation flow
3. Informative but concise
4. Capable of seamlessly blending bookmark information with general knowledge

When discussing bookmarks:
- Summarize relevant tweets naturally within the conversation
- Provide insights and connections between related bookmarks
- Offer relevant context from your general knowledge

For general conversations:
- Engage naturally while maintaining awareness of the user's interests
- Draw connections to bookmarked content when relevant
- Be helpful and informative while maintaining a friendly tone"""

    def build_prompt(self,
                    message: str,
                    intent: Intent,
                    context: Dict,
                    search_results: Optional[List[Dict]] = None) -> str:
        """
        Build a context-aware prompt optimized for Gemini 2.0.
        """
        prompt_parts = [self.system_prompt]

        # Add conversation context if available
        if context.get('recent_messages'):
            prompt_parts.append("\nRecent conversation:")
            prompt_parts.extend(context['recent_messages'])

        # Add search context for bookmark-related queries
        if search_results and intent.type in [IntentType.SEARCH, IntentType.FOLLOWUP]:
            prompt_parts.append("\nRelevant bookmarks:")
            for idx, result in enumerate(search_results[:5], 1):
                prompt_parts.append(
                    f"[{idx}] @{result['author'].replace('@', '')}: {result['text']}"
                )

        # Add specific instructions based on intent
        if intent.type == IntentType.SEARCH:
            prompt_parts.append(
                "\nFocus on providing a natural summary of the relevant bookmarks, "
                "highlighting key points and connections between them."
            )
        elif intent.type == IntentType.FOLLOWUP:
            prompt_parts.append(
                "\nContinue the previous discussion, maintaining context while "
                "incorporating any new relevant information."
            )
        elif intent.type == IntentType.CLARIFICATION:
            prompt_parts.append(
                "\nProvide a clear explanation, using examples from bookmarks "
                "if relevant to help illustrate the point."
            )
        elif intent.type == IntentType.GENERAL_CHAT:
            prompt_parts.append(
                "\nEngage in natural conversation, drawing on both general knowledge "
                "and relevant bookmarked content when appropriate."
            )

        # Add the current message
        prompt_parts.append(f"\nUser: {message}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def build_search_prompt(self, query: str, context: Dict) -> str:
        """
        Build a prompt specifically for improving search queries.
        """
        return f"""Based on the user's message and conversation context, help formulate an effective search query.

Context:
{context.get('topic', 'No specific topic')}
Recent messages:
{chr(10).join(context.get('recent_messages', ['No recent messages']))}

User's message: {query}

Suggest a search query that will find relevant bookmarks while considering:
1. Key terms and concepts
2. Related topics
3. Potential synonyms or alternative phrasings

Search query:"""

    def build_summary_prompt(self, bookmarks: List[Dict], context: Dict) -> str:
        """
        Build a prompt for summarizing search results in a natural way.
        """
        return f"""Summarize these bookmarks in a natural, conversational way that fits the current context.

Context:
Topic: {context.get('topic', 'General discussion')}
Bookmark context: {'Yes' if context.get('bookmark_context') else 'No'}

Bookmarks:
{chr(10).join(f"- @{b['author']}: {b['text']}" for b in bookmarks)}

Create a natural response that:
1. Highlights key information
2. Makes connections between related tweets
3. Adds relevant context or insights
4. Maintains a conversational tone

Response:""" 