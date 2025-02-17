"""
Category analyzer for bookmark categories and word frequencies.
Processes category data and generates word clouds.
"""

from typing import Dict, List, Any
import logging
from database.db import get_db_session
from database.models import Bookmark, Category
from sqlalchemy import func, and_
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

logger = logging.getLogger(__name__)

class CategoryAnalyzer:
    """Analyzes bookmark categories and generates word clouds."""
    
    def __init__(self):
        """Initialize the category analyzer."""
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
            self.stop_words = set()
    
    def get_category_distribution(self) -> Dict[str, Any]:
        """
        Get distribution of bookmarks across categories.
        Returns data formatted for pie chart visualization.
        """
        try:
            with get_db_session() as session:
                # Get category counts with percentage
                category_counts = session.query(
                    Category.name,
                    func.count(Bookmark.id).label('count')
                ).join(
                    Category.bookmarks
                ).group_by(
                    Category.name
                ).order_by(
                    func.count(Bookmark.id).desc()
                ).all()
                
                total_bookmarks = sum(count for _, count in category_counts)
                
                # Format data for visualization
                categories = [{
                    'name': cat_name,
                    'count': count,
                    'percentage': round((count / total_bookmarks) * 100, 2) if total_bookmarks > 0 else 0
                } for cat_name, count in category_counts]
                
                # Get category growth over time
                category_growth = self._get_category_growth()
                
                logger.info(f"Found {len(categories)} categories with {total_bookmarks} total bookmarks")
                logger.debug(f"Category distribution: {categories}")
                
                return {
                    'distribution': categories,
                    'growth': category_growth,
                    'total_categories': len(categories),
                    'total_bookmarks': total_bookmarks
                }
                
        except Exception as e:
            logger.error(f"Error in category distribution: {e}")
            raise Exception(f"Failed to get category distribution: {str(e)}")
    
    def get_category_wordcloud(self, category: str) -> Dict[str, Any]:
        """
        Generate word cloud data for a specific category.
        Returns word frequency data for visualization.
        """
        try:
            with get_db_session() as session:
                # Get all bookmark texts for the category
                bookmarks = session.query(
                    Bookmark.text
                ).join(
                    Bookmark.categories
                ).filter(
                    Category.name == category
                ).all()
                
                # Process text and generate word frequencies
                word_freq = self._process_text_for_wordcloud(
                    [bookmark.text for bookmark in bookmarks]
                )
                
                # Get category metadata
                bookmark_count = len(bookmarks)
                
                logger.info(f"Generated word cloud for category '{category}' with {bookmark_count} bookmarks")
                
                return {
                    'category': category,
                    'word_frequencies': [
                        {'text': word, 'value': count}
                        for word, count in word_freq.most_common(100)
                    ],
                    'metadata': {
                        'total_bookmarks': bookmark_count,
                        'unique_words': len(word_freq)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in word cloud generation: {e}")
            raise Exception(f"Failed to generate word cloud: {str(e)}")
    
    def _process_text_for_wordcloud(self, texts: List[str]) -> Counter:
        """Process text content and return word frequencies."""
        word_freq = Counter()
        
        for text in texts:
            # Tokenize and clean text
            try:
                # Remove URLs and mentions
                text = re.sub(r'http\S+|@\w+', '', text.lower())
                # Tokenize
                words = word_tokenize(text)
                # Filter words
                words = [
                    word for word in words
                    if word.isalnum() and  # Only alphanumeric
                    len(word) > 2 and      # More than 2 chars
                    word not in self.stop_words and  # Not a stop word
                    not word.isnumeric()   # Not just numbers
                ]
                word_freq.update(words)
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                continue
        
        return word_freq
    
    def _get_category_growth(self) -> List[Dict[str, Any]]:
        """Track category growth over time."""
        try:
            with get_db_session() as session:
                # Get category counts by month
                growth_data = session.query(
                    Category.name,
                    func.date_trunc('month', Bookmark.created_at).label('month'),
                    func.count(Bookmark.id).label('count')
                ).join(
                    Category.bookmarks
                ).group_by(
                    Category.name,
                    func.date_trunc('month', Bookmark.created_at)
                ).order_by(
                    'month'
                ).all()
                
                # Format data for time series visualization
                return [{
                    'category': entry[0],
                    'month': entry[1].strftime('%Y-%m'),
                    'count': entry[2]
                } for entry in growth_data]
                
        except Exception as e:
            logger.error(f"Error getting category growth: {e}")
            return [] 