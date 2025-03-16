"""
User API routes for the PythonAnywhere implementation.
Provides endpoints for user-specific data and actions.
"""

import logging
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, current_app, request
from auth.user_context_final import UserContext, login_required
from database.multi_user_db.search_final_multi_user import BookmarkSearchMultiUser
import traceback
import os
import json
from database.multi_user_db.user_model_final import (
    get_user_by_provider_id,
    get_user_by_id,
    create_user as db_create_user,
    update_last_login
)
from database.multi_user_db.db_final import get_db_connection

# Set up logging
logger = logging.getLogger('user_api_final')

# Create blueprint
user_api_bp = Blueprint('user_api', __name__, url_prefix='/api/user')

def get_db_connection():
    """Get a database connection from the app context"""
    return current_app.config['get_db_connection']()

@user_api_bp.route('/stats', methods=['GET'])
@login_required
def get_user_stats():
    """Get statistics for the current user"""
    user_id = UserContext.get_user_id()
    conn = get_db_connection()
    
    try:
        cursor = conn.cursor()
        
        # Get bookmark count
        cursor.execute(
            "SELECT COUNT(*) FROM bookmarks WHERE user_id = %s",
            (user_id,)
        )
        bookmark_count = cursor.fetchone()[0]
        
        # Get category count
        cursor.execute(
            "SELECT COUNT(*) FROM categories WHERE user_id = %s",
            (user_id,)
        )
        category_count = cursor.fetchone()[0]
        
        return jsonify({
            'bookmarks': bookmark_count,
            'categories': category_count,
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({
            'error': 'Failed to retrieve user statistics',
            'bookmarks': 0,
            'categories': 0
        }), 500
    finally:
        conn.close()

@user_api_bp.route('/activity', methods=['GET'])
@login_required
def get_user_activity():
    """Get recent activity for the current user"""
    user_id = UserContext.get_user_id()
    conn = get_db_connection()
    
    try:
        cursor = conn.cursor()
        
        # This is a placeholder implementation
        # In a real implementation, you would have an activity log table
        # For now, we'll return some mock data based on actual user data
        
        # Check for recent uploads
        one_week_ago = datetime.now() - timedelta(days=7)
        cursor.execute(
            """
            SELECT COUNT(*), MAX(created_at) 
            FROM bookmarks 
            WHERE user_id = %s AND created_at > %s
            """,
            (user_id, one_week_ago)
        )
        recent_uploads = cursor.fetchone()
        
        # Check for categories
        cursor.execute(
            """
            SELECT name FROM categories 
            WHERE user_id = %s 
            ORDER BY id DESC LIMIT 5
            """,
            (user_id,)
        )
        recent_categories = cursor.fetchall()
        
        # Build activity list
        activity = []
        
        if recent_uploads and recent_uploads[0] > 0:
            activity.append({
                'action': f"Added {recent_uploads[0]} new bookmarks",
                'timestamp': recent_uploads[1].strftime('%B %d, %Y at %H:%M') if recent_uploads[1] else 'Recently'
            })
            
        for category in recent_categories:
            activity.append({
                'action': f"Used category: {category[0]}",
                'timestamp': "Recently"
            })
            
        # Add a login activity
        user = UserContext.get_current_user()
        if user and user.last_login:
            activity.append({
                'action': "Logged in",
                'timestamp': user.last_login.strftime('%B %d, %Y at %H:%M')
            })
            
        return jsonify(activity)
        
    except Exception as e:
        logger.error(f"Error getting user activity: {e}")
        return jsonify([
            {
                'action': 'Account created',
                'timestamp': 'Recently'
            }
        ])
    finally:
        conn.close()

@user_api_bp.route('/profile', methods=['GET'])
@login_required
def get_user_profile():
    """Get profile information for the current user"""
    user = UserContext.get_current_user()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    return jsonify(user.to_dict())

@user_api_bp.route('/bookmarks', methods=['GET'])
@login_required
def get_user_bookmarks():
    """Get bookmarks for the current user"""
    user_id = UserContext.get_user_id()
    conn = get_db_connection()
    
    # Get query parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    # Calculate offset
    offset = (page - 1) * per_page
    
    try:
        cursor = conn.cursor()
        
        # Get bookmarks
        cursor.execute(
            """
            SELECT b.bookmark_id, b.text, b.author_name, b.author_username, b.created_at
            FROM bookmarks b
            WHERE b.user_id = %s
            ORDER BY b.created_at DESC
            LIMIT %s OFFSET %s
            """,
            (user_id, per_page, offset)
        )
        
        bookmarks = []
        for row in cursor.fetchall():
            bookmarks.append({
                'id': row[0],
                'text': row[1],
                'author_name': row[2],
                'author_username': row[3],
                'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if row[4] else None
            })
            
        # Get total count
        cursor.execute(
            "SELECT COUNT(*) FROM bookmarks WHERE user_id = %s",
            (user_id,)
        )
        total = cursor.fetchone()[0]
        
        # Calculate pagination info
        total_pages = (total + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1
        
        return jsonify({
            'bookmarks': bookmarks,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': total_pages,
                'has_next': has_next,
                'has_prev': has_prev
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting user bookmarks: {e}")
        return jsonify({
            'error': 'Failed to retrieve user bookmarks',
            'bookmarks': []
        }), 500
    finally:
        conn.close()

@user_api_bp.route('/api/categories', methods=['GET'])
@login_required
def get_categories():
    """Get all categories for the current user"""
    try:
        user_id = UserContext.get_user_id()
        conn = get_db_connection()
        
        # Create search instance - user_id is set here
        searcher = BookmarkSearchMultiUser(conn, user_id)
        
        try:
            # Get categories with counts - no parameters needed
            categories = searcher.get_categories()
            
            return jsonify(categories)
        except Exception as e:
            logger.error(f"Error calling get_categories method: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f'Error retrieving categories: {str(e)}',
                'categories': []
            }), 500
    except Exception as e:
        logger.error(f"Error in get_categories route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve categories',
            'categories': []
        }), 500

def find_or_create_user(user_data):
    """
    Find existing user or create a new one based on OAuth provider data.
    
    Args:
        user_data (dict): User data from OAuth provider containing:
            - provider: The OAuth provider name (e.g., 'twitter')
            - provider_id: The user's ID from the provider
            - username: Username
            - display_name: Display name (optional)
            - email: Email address (optional)
            - profile_image_url: URL to profile image (optional)
            - provider_data: JSON string with additional provider data (optional)
    
    Returns:
        dict: User object with user information if successful, None otherwise
    """
    logger.info(f"Finding or creating user for provider: {user_data.get('provider')}")
    
    try:
        # Get database connection
        db_conn = get_db_connection()
        
        # Required fields
        provider = user_data.get('provider')
        provider_id = user_data.get('provider_id')
        username = user_data.get('username')
        
        if not provider or not provider_id or not username:
            logger.error("Missing required user data fields")
            logger.error(f"Provider: {provider}, ID: {provider_id}, Username: {username}")
            return None
        
        # Optional fields with defaults
        display_name = user_data.get('display_name', username)
        email = user_data.get('email', f"{username}@{provider}.placeholder")
        profile_image_url = user_data.get('profile_image_url', '')
        provider_data = user_data.get('provider_data', '{}')
        
        # Find existing user by provider ID
        user = get_user_by_provider_id(db_conn, provider, provider_id)
        
        if user:
            logger.info(f"Found existing user: {user.username} (ID: {user.id})")
            
            # Update last login timestamp
            update_last_login(db_conn, user.id)
            
            # Convert user object to dictionary
            user_dict = {
                'id': user.id,
                'username': user.username,
                'display_name': user.display_name,
                'email': user.email,
                'auth_provider': user.auth_provider,
                'provider_id': user.provider_id,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'profile_image_url': user.profile_image_url
            }
            
            return user_dict
        
        # Create new user
        logger.info(f"Creating new user for {provider} user: {username}")
        
        new_user = db_create_user(
            db_conn,
            username=username,
            email=email,
            auth_provider=provider,
            provider_id=provider_id,
            display_name=display_name,
            profile_image_url=profile_image_url,
            provider_data=provider_data
        )
        
        if not new_user:
            logger.error("Failed to create new user")
            return None
        
        logger.info(f"Created new user: {new_user.username} (ID: {new_user.id})")
        
        # Convert user object to dictionary
        user_dict = {
            'id': new_user.id,
            'username': new_user.username,
            'display_name': new_user.display_name,
            'email': new_user.email,
            'auth_provider': new_user.auth_provider,
            'provider_id': new_user.provider_id,
            'created_at': new_user.created_at.isoformat() if new_user.created_at else None,
            'last_login': new_user.last_login.isoformat() if new_user.last_login else None,
            'profile_image_url': new_user.profile_image_url
        }
        
        return user_dict
    
    except Exception as e:
        logger.error(f"Error in find_or_create_user: {e}")
        logger.error(traceback.format_exc())
        return None 