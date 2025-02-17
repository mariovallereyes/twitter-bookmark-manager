"""
Dashboard routes for Twitter Bookmarks Manager.
Uses Flask Blueprint to handle all dashboard-related routes.
"""

from flask import Blueprint, jsonify, render_template, current_app, request
from typing import Dict, Any
import logging
from .data_processor import DashboardDataProcessor
from .heatmap_analyzer import HeatmapAnalyzer
from .category_analyzer import CategoryAnalyzer
from .author_analyzer import AuthorAnalyzer
from .topic_analyzer import TopicAnalyzer
import traceback

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Blueprint with url_prefix
dashboard = Blueprint('dashboard', __name__, url_prefix='/dashboard')

# Initialize analyzers
try:
    data_processor = DashboardDataProcessor()
    heatmap_analyzer = HeatmapAnalyzer()
    category_analyzer = CategoryAnalyzer()
    author_analyzer = AuthorAnalyzer()
    topic_analyzer = TopicAnalyzer()
    logger.info("✅ Successfully initialized all analyzers")
except Exception as e:
    logger.error(f"❌ Error initializing analyzers: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

@dashboard.route('/')
def show_dashboard():
    """Render the dashboard page."""
    logger.info("🎯 Rendering dashboard template")
    return render_template('dashboard.html')

@dashboard.route('/api/data')
def get_dashboard_data():
    """Get all dashboard data."""
    try:
        logger.info("📊 Processing dashboard data request")
        logger.info("🔍 Current app context: %s", str(current_app))
        
        # Get filter parameters
        date_range = request.args.get('dateRange', 'all')
        category = request.args.get('category', 'all')
        logger.info(f"Filters: dateRange={date_range}, category={category}")
        
        # Use existing data_processor instance
        logger.info("⚙️ Using data_processor instance")
        try:
            dashboard_data = data_processor.get_dashboard_data()
            logger.info("✅ Successfully retrieved dashboard data")
            logger.debug(f"Dashboard data: {dashboard_data}")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            logger.error(f"Database traceback: {traceback.format_exc()}")
            raise
        
        response_data = {
            'status': 'success',
            'data': dashboard_data
        }
        logger.info("📤 Sending response: %s", str(response_data))
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error processing dashboard data: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@dashboard.route('/api/heatmap')
def get_heatmap_data():
    """Get heatmap visualization data."""
    logger.info("📊 API Request: Fetching heatmap data")
    try:
        data = heatmap_analyzer.get_activity_heatmap()
        logger.info("✅ Successfully retrieved heatmap data")
        return jsonify({
            'status': 'success',
            'data': {
                'heatmap': data
            }
        })
    except Exception as e:
        error_msg = f"❌ Error getting heatmap data: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@dashboard.route('/api/categories')
def get_category_data():
    """Get category distribution data."""
    logger.info("📊 API Request: Fetching category data")
    try:
        data = category_analyzer.get_category_distribution()
        logger.info("✅ Successfully retrieved category data")
        return jsonify({
            'status': 'success',
            'data': {
                'categories': data
            }
        })
    except Exception as e:
        error_msg = f"❌ Error getting category data: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@dashboard.route('/api/categories/<category>/wordcloud')
def get_category_wordcloud(category: str):
    """Get word cloud data for a specific category."""
    logger.info(f"📊 API Request: Fetching word cloud data for category: {category}")
    try:
        data = category_analyzer.get_category_wordcloud(category)
        logger.info("✅ Successfully retrieved word cloud data")
        return jsonify({
            'status': 'success',
            'data': {
                'wordcloud': data
            }
        })
    except Exception as e:
        error_msg = f"❌ Error getting word cloud data: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@dashboard.route('/api/authors')
def get_author_data():
    """Get author statistics data."""
    logger.info("📊 API Request: Fetching author data")
    try:
        data = author_analyzer.get_top_authors()
        logger.info("✅ Successfully retrieved author data")
        return jsonify({
            'status': 'success',
            'data': {
                'authors': data
            }
        })
    except Exception as e:
        error_msg = f"❌ Error getting author data: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@dashboard.route('/api/authors/geo')
def get_author_geo_data():
    """Get author geographic distribution data."""
    logger.info("📊 API Request: Fetching author geographic data")
    try:
        data = author_analyzer.get_geographic_distribution()
        logger.info("✅ Successfully retrieved geographic data")
        return jsonify({
            'status': 'success',
            'data': {
                'geo': data
            }
        })
    except Exception as e:
        error_msg = f"❌ Error getting geographic data: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@dashboard.route('/api/topics')
def get_topic_data():
    """Get topic bubble chart data."""
    logger.info("📊 API Request: Fetching topic data")
    try:
        data = topic_analyzer.get_topic_bubbles()
        logger.info("✅ Successfully retrieved topic data")
        return jsonify({
            'status': 'success',
            'data': {
                'topics': data
            }
        })
    except Exception as e:
        error_msg = f"❌ Error getting topic data: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500 