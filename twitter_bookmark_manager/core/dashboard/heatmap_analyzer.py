"""
Heatmap analyzer for bookmark activity patterns.
Processes timestamp data to generate heatmap visualization data.
"""

from typing import Dict, List, Any
import logging
from database.db import get_db_session
from database.models import Bookmark
from sqlalchemy import func, extract
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HeatmapAnalyzer:
    """Analyzes bookmark timestamps for heatmap visualization."""
    
    def __init__(self):
        """Initialize the heatmap analyzer."""
        pass
    
    def get_activity_heatmap(self) -> Dict[str, Any]:
        """
        Generate heatmap data from bookmark timestamps.
        Returns data formatted for heatmap visualization.
        """
        try:
            # Get activity data for the last 12 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            with get_db_session() as session:
                # Query daily activity counts
                daily_counts = session.query(
                    func.date(Bookmark.created_at).label('date'),
                    func.count(Bookmark.id).label('count')
                ).filter(
                    Bookmark.created_at.between(start_date, end_date)
                ).group_by(
                    func.date(Bookmark.created_at)
                ).all()
                
                # Query hour-of-day distribution
                hourly_dist = session.query(
                    extract('hour', Bookmark.created_at).label('hour'),
                    func.count(Bookmark.id).label('count')
                ).filter(
                    Bookmark.created_at.between(start_date, end_date)
                ).group_by(
                    extract('hour', Bookmark.created_at)
                ).all()
                
                # Query day-of-week distribution
                weekly_dist = session.query(
                    extract('dow', Bookmark.created_at).label('day'),
                    func.count(Bookmark.id).label('count')
                ).filter(
                    Bookmark.created_at.between(start_date, end_date)
                ).group_by(
                    extract('dow', Bookmark.created_at)
                ).all()
            
            # Format data for visualization
            return {
                'daily_activity': [
                    {
                        'date': str(day.date),
                        'count': day.count
                    } for day in daily_counts
                ],
                'hourly_distribution': [
                    {
                        'hour': hour.hour,
                        'count': hour.count
                    } for hour in hourly_dist
                ],
                'weekly_distribution': [
                    {
                        'day': day.day,
                        'count': day.count
                    } for day in weekly_dist
                ],
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating heatmap data: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_peak_activity_times(self) -> Dict[str, Any]:
        """
        Analyze and return peak activity times.
        Useful for identifying optimal times for engagement.
        """
        try:
            with get_db_session() as session:
                # Get the hour with most activity
                peak_hour = session.query(
                    extract('hour', Bookmark.created_at).label('hour'),
                    func.count(Bookmark.id).label('count')
                ).group_by(
                    extract('hour', Bookmark.created_at)
                ).order_by(
                    func.count(Bookmark.id).desc()
                ).first()
                
                # Get the day of week with most activity
                peak_day = session.query(
                    extract('dow', Bookmark.created_at).label('day'),
                    func.count(Bookmark.id).label('count')
                ).group_by(
                    extract('dow', Bookmark.created_at)
                ).order_by(
                    func.count(Bookmark.id).desc()
                ).first()
                
                return {
                    'peak_hour': {
                        'hour': peak_hour.hour if peak_hour else None,
                        'count': peak_hour.count if peak_hour else 0
                    },
                    'peak_day': {
                        'day': peak_day.day if peak_day else None,
                        'count': peak_day.count if peak_day else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Error analyzing peak activity: {e}")
            return {} 