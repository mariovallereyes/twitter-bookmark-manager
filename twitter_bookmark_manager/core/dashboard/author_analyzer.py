"""
Author analyzer for bookmark author statistics and geographic distribution.
Processes author data and location information.
"""

from typing import Dict, List, Any
import logging
from database.db import get_db_session
from database.models import Bookmark
from sqlalchemy import func, desc, and_
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class AuthorAnalyzer:
    """Analyzes bookmark authors and their geographic distribution."""
    
    def __init__(self):
        """Initialize the author analyzer."""
        pass
    
    def get_top_authors(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get top authors by bookmark count and engagement metrics.
        Returns data formatted for ranking visualization.
        """
        try:
            with get_db_session() as session:
                # Get basic author counts
                author_counts = session.query(
                    Bookmark.author_username,
                    Bookmark.author_name,
                    func.count(Bookmark.id).label('bookmark_count'),
                    func.min(Bookmark.created_at).label('first_bookmark'),
                    func.max(Bookmark.created_at).label('last_bookmark')
                ).group_by(
                    Bookmark.author_username,
                    Bookmark.author_name
                ).order_by(
                    desc('bookmark_count')
                ).limit(limit).all()
                
                # Calculate engagement periods and frequencies
                authors = []
                for author in author_counts:
                    engagement_period = (author.last_bookmark - author.first_bookmark).days
                    bookmarks_per_month = (author.bookmark_count / max(1, engagement_period/30))
                    
                    authors.append({
                        'username': author.author_username,
                        'display_name': author.author_name or author.author_username,
                        'bookmark_count': author.bookmark_count,
                        'engagement_metrics': {
                            'first_bookmark': author.first_bookmark.strftime('%Y-%m-%d'),
                            'last_bookmark': author.last_bookmark.strftime('%Y-%m-%d'),
                            'engagement_period_days': engagement_period,
                            'bookmarks_per_month': round(bookmarks_per_month, 2)
                        }
                    })
                
                # Get recent activity for top authors
                recent_activity = self._get_recent_author_activity(
                    [author['username'] for author in authors]
                )
                
                return {
                    'top_authors': authors,
                    'recent_activity': recent_activity,
                    'total_authors': session.query(
                        Bookmark.author_username
                    ).distinct().count()
                }
                
        except Exception as e:
            logger.error(f"Error in author ranking: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_geographic_distribution(self) -> Dict[str, Any]:
        """
        Get geographic distribution of authors from their profile locations.
        Returns data formatted for map visualization.
        """
        try:
            with get_db_session() as session:
                # Get author locations from raw_data
                author_locations = session.query(
                    Bookmark.author_username,
                    Bookmark.raw_data['user']['location'].label('location'),
                    func.count(Bookmark.id).label('bookmark_count')
                ).filter(
                    Bookmark.raw_data['user']['location'].isnot(None)
                ).group_by(
                    Bookmark.author_username,
                    Bookmark.raw_data['user']['location']
                ).all()
                
                # Process and aggregate location data
                location_data = self._process_location_data(author_locations)
                
                return {
                    'locations': location_data,
                    'total_authors_with_location': len(author_locations),
                    'top_locations': self._get_top_locations(location_data)
                }
                
        except Exception as e:
            logger.error(f"Error in geographic distribution: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_recent_author_activity(self, usernames: List[str], days: int = 30) -> List[Dict[str, Any]]:
        """Get recent activity data for specified authors."""
        try:
            with get_db_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                recent_activity = session.query(
                    Bookmark.author_username,
                    func.date(Bookmark.created_at).label('date'),
                    func.count(Bookmark.id).label('count')
                ).filter(
                    and_(
                        Bookmark.author_username.in_(usernames),
                        Bookmark.created_at >= cutoff_date
                    )
                ).group_by(
                    Bookmark.author_username,
                    func.date(Bookmark.created_at)
                ).all()
                
                return [{
                    'username': activity.author_username,
                    'date': activity.date.strftime('%Y-%m-%d'),
                    'count': activity.count
                } for activity in recent_activity]
                
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return []
    
    def _process_location_data(self, author_locations) -> List[Dict[str, Any]]:
        """Process and normalize location data from author profiles."""
        location_data = []
        
        for author in author_locations:
            try:
                location = json.loads(author.location) if isinstance(author.location, str) else author.location
                if location and isinstance(location, str):
                    location_data.append({
                        'location': location.strip(),
                        'author_count': 1,
                        'bookmark_count': author.bookmark_count
                    })
            except Exception as e:
                logger.warning(f"Error processing location for {author.author_username}: {e}")
                continue
        
        # Aggregate similar locations
        aggregated = {}
        for loc in location_data:
            location = loc['location']
            if location in aggregated:
                aggregated[location]['author_count'] += 1
                aggregated[location]['bookmark_count'] += loc['bookmark_count']
            else:
                aggregated[location] = {
                    'location': location,
                    'author_count': 1,
                    'bookmark_count': loc['bookmark_count']
                }
        
        return list(aggregated.values())
    
    def _get_top_locations(self, location_data: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top locations by author count."""
        sorted_locations = sorted(
            location_data,
            key=lambda x: (x['author_count'], x['bookmark_count']),
            reverse=True
        )
        return sorted_locations[:limit] 