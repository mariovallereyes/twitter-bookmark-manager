"""
Chat-specific search module for the PythonAnywhere chat implementation.
This module provides search functionality optimized for chat interactions.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import re
import numpy as np

logger = logging.getLogger(__name__)

class ChatBookmarkSearchPA:
    """
    Provides search capabilities over bookmarks specifically optimized for chat interactions.
    Uses both PostgreSQL for metadata filtering and Qdrant for vector search.
    """
    
    def __init__(self):
        """Initialize the chat-specific search engine."""
        # We'll lazily initialize the search components when needed
        self._search_engine = None
        self._vector_store = None
        self._db_engine = None
        self._initialized = False
        
        logger.info("✓ Chat bookmark search initialized in lazy-loading mode")
    
    def _ensure_initialized(self) -> None:
        """Ensure search components are initialized."""
        if self._initialized:
            return
            
        try:
            # Import PA-specific search module
            from twitter_bookmark_manager.deployment.pythonanywhere.database.search_pa import BookmarkSearch
            self._search_engine = BookmarkSearch()
            
            # Import PA-specific vector store
            from twitter_bookmark_manager.deployment.pythonanywhere.database.vector_store_pa import VectorStore
            self._vector_store = VectorStore()
            
            # Import PostgreSQL database connection
            from twitter_bookmark_manager.deployment.pythonanywhere.database.db_pa import get_session
            self._db_engine = get_session  # This is a function that returns a session
            
            self._initialized = True
            logger.info("✓ Chat bookmark search components initialized")
        except ImportError as e:
            logger.error(f"Error initializing search components: {e}")
            raise
    
    def search(self, query: str, limit: int = 5, 
              context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for bookmarks matching the query, with context-aware boosting.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            context: Context information such as conversation topic and recent categories
            
        Returns:
            List of matching bookmark dictionaries
        """
        self._ensure_initialized()
        
        try:
            # Preprocess the query for better results
            cleaned_query = self._preprocess_query(query)
            
            # Log the search attempt with context info
            context_log = f"topic={context.get('topic', 'None')}" if context else "no context"
            logger.info(f"Chat search: '{query}' -> '{cleaned_query}' (limit={limit}, {context_log})")
            
            # Try the main search engine first with different strategies
            results = self._hybrid_search(cleaned_query, query, limit, context)
            
            if results:
                logger.info(f"Found {len(results)} results for chat query '{query}'")
                return results
            
            # If no results, try word-by-word search for multi-word queries
            if " " in cleaned_query:
                logger.info(f"No results for '{cleaned_query}', trying word-by-word search")
                words = [w for w in cleaned_query.split() if len(w) > 2]
                if words:
                    word_results = []
                    for word in words[:3]:  # Try the first 3 words only
                        logger.info(f"Searching for individual word: '{word}'")
                        word_search = self._hybrid_search(word, word, limit, context)
                        if word_search:
                            word_results.extend(word_search)
                    
                    if word_results:
                        # Remove duplicates
                        seen_ids = set()
                        unique_results = []
                        for res in word_results:
                            if res['id'] not in seen_ids:
                                seen_ids.add(res['id'])
                                unique_results.append(res)
                        
                        logger.info(f"Found {len(unique_results)} results in word-by-word search")
                        return unique_results[:limit]
            
            # If still no results, try a more permissive search with simplified query
            simplified_query = self._simplify_query(query)
            if simplified_query != cleaned_query:
                logger.info(f"Trying simplified query: '{simplified_query}'")
                simple_results = self._hybrid_search(simplified_query, simplified_query, limit, context)
                if simple_results:
                    return simple_results
            
            # Use category-based search if we have context
            if context and context.get('recent_categories'):
                categories = context.get('recent_categories', [])
                logger.info(f"Trying category search with categories: {categories}")
                category_results = self._search_engine.search(categories=categories, limit=limit)
                
                if category_results:
                    logger.info(f"Found {len(category_results)} results using category search")
                    return category_results
            
            # If still no results, get most recent bookmarks as fallback
            logger.info("No results with permissive search, returning recent bookmarks")
            return self._search_engine.get_all_bookmarks(limit=limit)
            
        except Exception as e:
            logger.error(f"Error in chat search: {e}")
            # Return empty results on error
            return []
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to improve search results.
        
        Args:
            query: The original search query
            
        Returns:
            Cleaned query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove common search phrases
        phrases_to_remove = [
            "find tweets about", "find me tweets about", 
            "search for", "look for", "show me", 
            "can you find", "i need", "i want", 
            "please find", "please show", "tweets about",
            "related to", "talking about", "mentions of"
        ]
        
        for phrase in phrases_to_remove:
            query = query.replace(phrase, "")
        
        # Remove excess whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # If query has become too short, use original
        if len(query) < 2:
            return query.strip()
        
        return query
    
    def _simplify_query(self, query: str) -> str:
        """
        Create a simplified version of the query with only essential terms.
        
        Args:
            query: The original query
            
        Returns:
            Simplified query with only key terms
        """
        # Use only uppercase terms and potential acronyms
        uppercase_terms = re.findall(r'\b[A-Z]{2,}\b', query)
        if uppercase_terms:
            return ' '.join(uppercase_terms)
        
        # Extract potential keywords (nouns and entities)
        words = query.split()
        # Keep words with uppercase letters or longer than 4 characters
        keywords = [w for w in words if any(c.isupper() for c in w) or len(w) > 4]
        
        if keywords:
            return ' '.join(keywords)
        
        return query
    
    def _hybrid_search(self, query: str, original_query: str, limit: int, 
                      context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and SQL approaches with context boosting.
        
        Args:
            query: The preprocessed search query
            original_query: The original search query
            limit: Maximum number of results
            context: Search context
            
        Returns:
            List of search results
        """
        try:
            # Check for acronyms and abbreviations
            is_acronym = query.isupper() and len(query) <= 5
            
            # First try direct search through main search engine
            sql_results = self._search_engine.search_bookmarks(query=query, limit=limit)
            
            # Try original query if preprocessed query returned no results
            if not sql_results and query != original_query:
                sql_results = self._search_engine.search_bookmarks(query=original_query, limit=limit)
            
            # If query looks like an acronym, try more permissive SQL search with LIKE
            if is_acronym and not sql_results:
                try:
                    # This is a mock implementation - you'll need to adapt this to your actual DB structure
                    # This assumes there's a way to do a LIKE search in your search engine
                    logger.info(f"Trying LIKE search for acronym: '{query}'")
                    acronym_pattern = f"%{query}%"
                    sql_session = self._db_engine()
                    # Placeholder for actual SQL implementation - will need modification
                    # Example using SQLAlchemy: bookmarks = session.query(Bookmark).filter(Bookmark.text.ilike(acronym_pattern)).limit(limit).all()
                    # Then convert to dict format compatible with your app
                except Exception as sql_error:
                    logger.error(f"Error in SQL acronym search: {sql_error}")
            
            # If SQL search got good results, return them
            if sql_results and len(sql_results) >= min(3, limit):
                logger.info(f"SQL search found {len(sql_results)} results")
                return self._boost_results_by_context(sql_results, context, limit)
            
            # Otherwise try vector search
            if self._vector_store:
                try:
                    logger.info(f"Performing vector search for '{query}'")
                    # Lower score threshold for acronyms to be more permissive
                    score_threshold = 0.4
                    if is_acronym:
                        score_threshold = 0.3  # More permissive for acronyms
                    
                    # Use search() with proper parameters
                    vector_results = self._vector_store.search(
                        query=query,
                        limit=limit * 3,  # Triple limit for better coverage
                        score_threshold=score_threshold
                    )
                    
                    # If no results with preprocessed query, try original query
                    if not vector_results and query != original_query:
                        vector_results = self._vector_store.search(
                            query=original_query,
                            limit=limit * 3,
                            score_threshold=score_threshold
                        )
                    
                    if vector_results:
                        logger.info(f"Vector search found {len(vector_results)} results")
                        # Convert vector results to the same format as SQL results
                        formatted_results = self._format_vector_results(vector_results)
                        # Apply context boosting to vector results
                        return self._boost_results_by_context(formatted_results, context, limit)
                except Exception as vector_error:
                    logger.error(f"Vector search error: {vector_error}")
                    # Log the error but continue with fallback strategies
            
            # Try individual words if query has multiple words
            if " " in query and len(query.split()) > 1:
                logger.info(f"Trying individual word search for '{query}'")
                all_word_results = []
                
                for word in query.split():
                    if len(word) > 2:  # Skip very short words
                        word_results = self._search_engine.search_bookmarks(query=word, limit=limit)
                        if word_results:
                            all_word_results.extend(word_results)
                
                # Remove duplicates
                seen_ids = set()
                unique_results = []
                for res in all_word_results:
                    if res['id'] not in seen_ids:
                        seen_ids.add(res['id'])
                        unique_results.append(res)
                
                if unique_results:
                    logger.info(f"Word-by-word search found {len(unique_results)} results")
                    return self._boost_results_by_context(unique_results, context, limit)
            
            # If we have some SQL results even if fewer than limit, return them
            if sql_results:
                return self._boost_results_by_context(sql_results, context, limit)
            
            # If no results from either approach, try full-text search with the main engine
            return self._search_engine.search(query=query, limit=limit)
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _format_vector_results(self, vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format vector search results to match the expected bookmark format.
        
        Args:
            vector_results: Results from vector search
            
        Returns:
            Formatted bookmark dictionaries
        """
        formatted = []
        
        # Note: This assumes a specific format from the vector store
        # In a real implementation, you'd need to adapt this to your vector store's output format
        for result in vector_results:
            # Extract the document ID (tweet ID)
            doc_id = result.get('id') or result.get('document_id') or result.get('payload', {}).get('id')
            
            if not doc_id:
                logger.warning(f"Missing document ID in vector result: {result}")
                continue
                
            # Fetch the full bookmark from the database
            try:
                # Use the main search engine to get the bookmark by ID
                bookmark = self._search_engine.get_bookmark_by_id(doc_id)
                
                if bookmark:
                    # Add relevance score from vector search
                    bookmark['relevance_score'] = result.get('score', 1.0)
                    formatted.append(bookmark)
            except Exception as e:
                logger.error(f"Error fetching bookmark {doc_id}: {e}")
        
        return formatted
    
    def _boost_results_by_context(self, results: List[Dict[str, Any]], 
                                context: Optional[Dict[str, Any]],
                                limit: int) -> List[Dict[str, Any]]:
        """
        Boost search results based on conversation context.
        
        Args:
            results: The search results to boost
            context: The conversation context
            limit: Maximum number of results to return
            
        Returns:
            Re-ranked search results
        """
        if not context or not results:
            return results
            
        # Extract context elements that influence ranking
        topic = context.get('topic')
        recent_categories = set(context.get('recent_categories', []))
        
        # Store original order for stable sorting
        for i, result in enumerate(results):
            result['original_order'] = i
            
        # Calculate boost for each result
        for result in results:
            boost = 1.0  # Base score
            
            # Boost by categories
            if recent_categories and result.get('categories'):
                matching_categories = recent_categories.intersection(set(result['categories']))
                category_boost = len(matching_categories) * 0.1
                boost += category_boost
            
            # Boost by topic match in text
            if topic and result.get('text'):
                if topic.lower() in result['text'].lower():
                    boost += 0.2
            
            # Store the boost score
            result['context_boost'] = boost
        
        # Sort by combined score (relevance * boost) and then by original order for stability
        results.sort(key=lambda x: (
            x.get('relevance_score', 1.0) * x.get('context_boost', 1.0), 
            -x['original_order']  # Negative to maintain original order as tiebreaker
        ), reverse=True)
        
        # Remove the temporary scoring fields
        for result in results:
            result.pop('original_order', None)
            result.pop('context_boost', None)
        
        return results[:limit]  # Return only up to the limit
    
    def search_by_category(self, categories: List[str], 
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for bookmarks by category.
        
        Args:
            categories: List of category names to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching bookmark dictionaries
        """
        self._ensure_initialized()
        
        try:
            logger.info(f"Chat category search for: {categories}")
            results = self._search_engine.search(categories=categories, limit=limit)
            logger.info(f"Found {len(results)} results in categories {categories}")
            return results
        except Exception as e:
            logger.error(f"Error in category search: {e}")
            return []
    
    def search_by_user(self, username: str, 
                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for bookmarks by Twitter username.
        
        Args:
            username: Twitter username (with or without @)
            limit: Maximum number of results to return
            
        Returns:
            List of matching bookmark dictionaries
        """
        self._ensure_initialized()
        
        try:
            # Remove @ if present
            clean_username = username.lstrip('@')
            logger.info(f"Chat user search for: @{clean_username}")
            
            results = self._search_engine.search_by_user(clean_username, limit=limit)
            logger.info(f"Found {len(results)} results for user @{clean_username}")
            return results
        except Exception as e:
            logger.error(f"Error in user search: {e}")
            return []
    
    def get_related_bookmarks(self, bookmark_id: str, 
                            limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get bookmarks related to a specific bookmark.
        
        Args:
            bookmark_id: The ID of the bookmark to find related items for
            limit: Maximum number of results to return
            
        Returns:
            List of related bookmark dictionaries
        """
        self._ensure_initialized()
        
        try:
            logger.info(f"Finding bookmarks related to ID: {bookmark_id}")
            
            # Get the source bookmark first
            source_bookmark = self._search_engine.get_bookmark_by_id(bookmark_id)
            if not source_bookmark:
                logger.error(f"Source bookmark {bookmark_id} not found")
                return []
            
            # Try vector search for semantic similarity
            if self._vector_store:
                try:
                    related = self._vector_store.search_by_id(bookmark_id, top_k=limit)
                    if related:
                        logger.info(f"Found {len(related)} related bookmarks via vector search")
                        return self._format_vector_results(related)
                except Exception as vector_error:
                    logger.error(f"Vector search for related bookmarks failed: {vector_error}")
            
            # Fall back to category-based related bookmarks
            if source_bookmark.get('categories'):
                logger.info(f"Falling back to category-based related bookmarks")
                category_results = self._search_engine.search(
                    categories=source_bookmark['categories'],
                    limit=limit + 1  # Get one extra to remove the source
                )
                
                # Filter out the source bookmark
                filtered_results = [r for r in category_results if r['id'] != bookmark_id]
                
                return filtered_results[:limit]
            
            # If nothing else works, return recent bookmarks
            return self._search_engine.get_all_bookmarks(limit=limit)
            
        except Exception as e:
            logger.error(f"Error finding related bookmarks: {e}")
            return []
