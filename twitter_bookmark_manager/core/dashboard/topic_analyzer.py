"""
Topic analyzer for bookmark content analysis.
Processes text content to generate topic bubble visualizations.
"""

from typing import Dict, List, Any, Tuple
import logging
from database.db import get_db_session
from database.models import Bookmark, Category
from sqlalchemy import func, and_
from collections import Counter, defaultdict
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import nltk
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TopicAnalyzer:
    """Analyzes bookmark content for topic visualization."""
    
    def __init__(self):
        """Initialize the topic analyzer."""
        # Initialize NLTK resources
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            # Add custom stop words relevant to Twitter content
            self.stop_words.update(['rt', 'thread', 'tweet', 'twitter'])
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
            self.stop_words = set()
    
    def get_topic_bubbles(self) -> Dict[str, Any]:
        """
        Generate topic bubble data from bookmark content.
        Returns data formatted for bubble chart visualization.
        """
        try:
            # Get recent bookmarks for topic analysis
            recent_bookmarks = self._get_recent_bookmarks()
            
            # Extract topics and their relationships
            topics = self._extract_topics(recent_bookmarks)
            relationships = self._find_topic_relationships(recent_bookmarks)
            
            # Get topic trends over time
            trends = self._analyze_topic_trends(topics.keys())
            
            return {
                'topics': [
                    {
                        'name': topic,
                        'value': count,
                        'sentiment': self._get_topic_sentiment(topic, recent_bookmarks),
                        'categories': self._get_topic_categories(topic, recent_bookmarks)
                    }
                    for topic, count in topics.most_common(50)
                ],
                'relationships': [
                    {
                        'source': source,
                        'target': target,
                        'strength': strength
                    }
                    for (source, target), strength in relationships.items()
                ],
                'trends': trends,
                'metadata': {
                    'total_topics': len(topics),
                    'analysis_period': '30 days',
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_recent_bookmarks(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent bookmarks for analysis."""
        try:
            with get_db_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                bookmarks = session.query(
                    Bookmark.text,
                    Bookmark.created_at,
                    func.array_agg(Category.name).label('categories')
                ).join(
                    Bookmark.categories
                ).filter(
                    Bookmark.created_at >= cutoff_date
                ).group_by(
                    Bookmark.id,
                    Bookmark.text,
                    Bookmark.created_at
                ).all()
                
                return [{
                    'text': b.text,
                    'created_at': b.created_at,
                    'categories': b.categories
                } for b in bookmarks]
                
        except Exception as e:
            logger.error(f"Error getting recent bookmarks: {e}")
            return []
    
    def _extract_topics(self, bookmarks: List[Dict[str, Any]]) -> Counter:
        """Extract main topics from bookmark content."""
        topics = Counter()
        
        for bookmark in bookmarks:
            try:
                # Clean and tokenize text
                text = re.sub(r'http\S+|@\w+|#\w+', '', bookmark['text'].lower())
                sentences = sent_tokenize(text)
                
                for sentence in sentences:
                    # Extract noun phrases as potential topics
                    words = word_tokenize(sentence)
                    tagged = nltk.pos_tag(words)
                    
                    # Look for noun phrases (consecutive nouns)
                    current_topic = []
                    for word, tag in tagged:
                        if tag.startswith('NN') and word not in self.stop_words:
                            current_topic.append(word)
                        elif current_topic:
                            if len(current_topic) > 0:
                                topic = ' '.join(current_topic)
                                topics[topic] += 1
                            current_topic = []
                    
                    if current_topic:  # Handle last topic in sentence
                        topic = ' '.join(current_topic)
                        topics[topic] += 1
                        
            except Exception as e:
                logger.warning(f"Error processing bookmark text: {e}")
                continue
        
        return topics
    
    def _find_topic_relationships(self, bookmarks: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
        """Find relationships between topics based on co-occurrence."""
        relationships = defaultdict(int)
        
        for bookmark in bookmarks:
            try:
                # Get topics in this bookmark
                text = re.sub(r'http\S+|@\w+|#\w+', '', bookmark['text'].lower())
                words = word_tokenize(text)
                
                # Find bigrams
                finder = BigramCollocationFinder.from_words(words)
                finder.apply_word_filter(lambda w: w in self.stop_words or not w.isalnum())
                
                # Score bigrams using PMI
                bigram_measures = BigramAssocMeasures()
                scored = finder.score_ngrams(bigram_measures.pmi)
                
                # Add to relationships
                for (w1, w2), score in scored:
                    if score > 0:  # Only keep positive associations
                        relationships[(w1, w2)] = score
                        
            except Exception as e:
                logger.warning(f"Error finding topic relationships: {e}")
                continue
        
        return relationships
    
    def _analyze_topic_trends(self, topics: List[str], days: int = 30) -> List[Dict[str, Any]]:
        """Analyze how topics trend over time."""
        try:
            with get_db_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                trends = []
                
                for topic in topics:
                    # Count occurrences by day
                    daily_counts = session.query(
                        func.date(Bookmark.created_at).label('date'),
                        func.count(Bookmark.id).label('count')
                    ).filter(
                        and_(
                            Bookmark.created_at >= cutoff_date,
                            func.lower(Bookmark.text).like(f'%{topic.lower()}%')
                        )
                    ).group_by(
                        func.date(Bookmark.created_at)
                    ).all()
                    
                    trends.append({
                        'topic': topic,
                        'trend': [{
                            'date': str(day.date),
                            'count': day.count
                        } for day in daily_counts]
                    })
                
                return trends
                
        except Exception as e:
            logger.error(f"Error analyzing topic trends: {e}")
            return []
    
    def _get_topic_sentiment(self, topic: str, bookmarks: List[Dict[str, Any]]) -> float:
        """Simple sentiment analysis for topics (placeholder for future enhancement)."""
        return 0.0  # Neutral sentiment
    
    def _get_topic_categories(self, topic: str, bookmarks: List[Dict[str, Any]]) -> List[str]:
        """Get categories associated with a topic."""
        topic_categories = Counter()
        
        for bookmark in bookmarks:
            if topic.lower() in bookmark['text'].lower():
                topic_categories.update(bookmark['categories'])
        
        return [cat for cat, _ in topic_categories.most_common(3)] 