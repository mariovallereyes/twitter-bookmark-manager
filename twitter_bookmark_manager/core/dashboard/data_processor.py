"""
Main data processor for dashboard visualizations.
Coordinates data retrieval and processing from various analyzers.
"""

from typing import Dict, List, Any
import logging
import traceback
from database.db import get_db_session
from database.models import Bookmark, Category
from .heatmap_analyzer import HeatmapAnalyzer
from .category_analyzer import CategoryAnalyzer
from .author_analyzer import AuthorAnalyzer
from .topic_analyzer import TopicAnalyzer

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardDataProcessor:
    """Coordinates data processing for dashboard visualizations."""
    
    def __init__(self):
        """Initialize the dashboard data processor and its analyzers."""
        logger.info("🔧 Initializing DashboardDataProcessor")
        self.heatmap_analyzer = HeatmapAnalyzer()
        self.category_analyzer = CategoryAnalyzer()
        self.author_analyzer = AuthorAnalyzer()
        self.topic_analyzer = TopicAnalyzer()
        logger.info("✅ Successfully initialized all analyzers")
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get all necessary data for the dashboard.
        Returns a dictionary with data for each visualization.
        """
        logger.info("📊 Starting dashboard data collection")
        try:
            with get_db_session() as session:
                # Get total bookmark count
                total_bookmarks = session.query(Bookmark).count()
                logger.info(f"📚 Found {total_bookmarks} total bookmarks")
                
                if total_bookmarks == 0:
                    logger.warning("No bookmarks found in database")
                    return {
                        'total_bookmarks': 0,
                        'statistics': self._get_empty_stats(),
                        'visualizations': self._get_empty_visualizations()
                    }
                
                # Get data from each analyzer
                logger.info("🔄 Fetching heatmap data...")
                heatmap_data = self.heatmap_analyzer.get_activity_heatmap()
                logger.info(f"Heatmap data: {heatmap_data}")
                
                logger.info("🔄 Fetching category data...")
                category_data = self.category_analyzer.get_category_distribution()
                logger.info(f"Category data: {category_data}")
                
                logger.info("🔄 Fetching author data...")
                author_data = self.author_analyzer.get_top_authors(limit=10)
                logger.info(f"Author data: {author_data}")
                
                logger.info("🔄 Fetching topic data...")
                topic_data = self.topic_analyzer.get_topic_bubbles()
                logger.info(f"Topic data: {topic_data}")
                
                # Get overall statistics
                logger.info("🔄 Calculating overall statistics...")
                stats = self._get_overall_stats(session)
                
                logger.info("✅ Successfully collected all dashboard data")
                
                # Ensure all data is properly structured
                visualizations = {
                    'heatmap': heatmap_data if isinstance(heatmap_data, dict) else {},
                    'categories': category_data if isinstance(category_data, dict) else {
                        'distribution': [],
                        'growth': [],
                        'total_categories': 0,
                        'total_bookmarks': 0
                    },
                    'authors': author_data if isinstance(author_data, dict) else {
                        'top_authors': [],
                        'recent_activity': [],
                        'total_authors': 0
                    },
                    'topics': topic_data if isinstance(topic_data, dict) else {
                        'topics': [],
                        'relationships': [],
                        'trends': [],
                        'metadata': {
                            'total_topics': 0,
                            'analysis_period': '30 days',
                            'last_updated': None
                        }
                    }
                }
                
                return {
                    'total_bookmarks': total_bookmarks,
                    'statistics': stats,
                    'visualizations': visualizations
                }
        except Exception as e:
            error_msg = f"❌ Error gathering dashboard data: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(error_msg)
    
    def _get_overall_stats(self, session) -> Dict[str, Any]:
        """Calculate overall statistics for the dashboard."""
        logger.info("📊 Calculating overall statistics")
        try:
            total_bookmarks = session.query(Bookmark).count()
            total_categories = session.query(Category).count()
            
            # Get unique authors count
            unique_authors = session.query(Bookmark.author_username)\
                .distinct()\
                .count()
            
            # Get bookmark count by month (last 6 months)
            monthly_counts = session.query(Bookmark)\
                .filter(Bookmark.created_at >= '2024-01-01')\
                .count()
            
            last_updated = session.query(Bookmark.created_at)\
                .order_by(Bookmark.created_at.desc())\
                .first()
            
            stats = {
                'total_bookmarks': total_bookmarks,
                'total_categories': total_categories,
                'unique_authors': unique_authors,
                'monthly_activity': monthly_counts,
                'last_updated': last_updated[0].strftime('%Y-%m-%d %H:%M:%S') if last_updated else None
            }
            
            logger.info(f"📈 Statistics calculated: {stats}")
            return stats
                
        except Exception as e:
            error_msg = f"❌ Error calculating overall stats: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_empty_stats()
    
    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics structure."""
        return {
            'total_bookmarks': 0,
            'total_categories': 0,
            'unique_authors': 0,
            'monthly_activity': 0,
            'last_updated': None
        }
    
    def _get_empty_visualizations(self) -> Dict[str, Any]:
        """Return empty visualization structure."""
        return {
            'heatmap': {
                'daily_activity': [],
                'hourly_distribution': [],
                'weekly_distribution': []
            },
            'categories': {
                'distribution': [],
                'growth': [],
                'total_categories': 0,
                'total_bookmarks': 0
            },
            'authors': {
                'top_authors': [],
                'recent_activity': [],
                'total_authors': 0
            },
            'topics': {
                'topics': [],
                'relationships': [],
                'trends': [],
                'metadata': {
                    'total_topics': 0,
                    'analysis_period': '30 days',
                    'last_updated': None
                }
            }
        } 