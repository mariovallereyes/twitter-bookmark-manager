"""
Multi-user API server code for final environment.
This extends api_server.py with user authentication and multi-user support.
"""

import os
import sys
import logging
import json
import time
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask.sessions import SecureCookieSessionInterface

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('api_server_multi_user')

# Import user authentication components
from auth.auth_routes_final import auth_bp
from auth.user_api_final import user_api_bp
from auth.user_context_final import UserContextMiddleware, UserContext
from database.multi_user_db.user_model_final import get_user_by_id

# Import database modules
from database.multi_user_db.db_final import get_db_connection, create_tables
from database.multi_user_db.search_final_multi_user import BookmarkSearchMultiUser

# Create Flask app
app = Flask(__name__, 
            template_folder='../web_final/templates',
            static_folder='../web_final/static')

# Configure app
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', secrets.token_hex(32)),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    get_db_connection=get_db_connection,
    # Force HTTPS for all URL generation
    PREFERRED_URL_SCHEME='https'
)

# Set session to be permanent by default
@app.before_request
def make_session_permanent():
    session.permanent = True

# Register authentication blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(user_api_bp)

# Initialize user context middleware
UserContextMiddleware(app, lambda user_id: get_user_by_id(get_db_connection(), user_id))

# Home page
@app.route('/')
def index():
    """Home page - now aware of user context"""
    user = UserContext.get_current_user()
    
    # Choose template based on authentication
    if user:
        template = 'index_final.html'
    else:
        # Show login page for unauthenticated users
        return redirect(url_for('auth.login'))
    
    # Get categories for the current user
    conn = get_db_connection()
    try:
        searcher = BookmarkSearchMultiUser(conn, user.id if user else 1)
        categories = searcher.get_categories(user.id if user else 1)
        
        return render_template(template, categories=categories, user=user)
    finally:
        conn.close()

# Search endpoint
@app.route('/search', methods=['GET'])
def search():
    """Search endpoint - now filters by user_id"""
    # Get current user
    user = UserContext.get_current_user()
    
    # Redirect to login if not authenticated
    if not user:
        return redirect(url_for('auth.login', next=request.url))
    
    # Get search parameters
    query = request.args.get('q', '')
    author = request.args.get('author', '')
    categories = request.args.getlist('category')
    
    # Get connection and search
    conn = get_db_connection()
    try:
        searcher = BookmarkSearchMultiUser(conn, user.id)
        
        # Convert category IDs to integers
        category_ids = [int(c) for c in categories if c.isdigit()]
        
        # Perform search
        results = searcher.search(
            query=query, 
            user=author, 
            category_ids=category_ids,
            user_id=user.id
        )
        
        # Get all categories for display
        all_categories = searcher.get_categories(user.id)
        
        return render_template(
            'search_results_final.html',
            query=query,
            author=author,
            results=results,
            categories=all_categories,
            selected_categories=category_ids,
            user=user
        )
    finally:
        conn.close()

# Recent bookmarks endpoint
@app.route('/recent', methods=['GET'])
def recent():
    """Recent bookmarks endpoint - now filters by user_id"""
    # Get current user
    user = UserContext.get_current_user()
    
    # Redirect to login if not authenticated
    if not user:
        return redirect(url_for('auth.login', next=request.url))
    
    # Get connection and fetch recent bookmarks
    conn = get_db_connection()
    try:
        searcher = BookmarkSearchMultiUser(conn, user.id)
        results = searcher.get_recent(user_id=user.id)
        
        # Get all categories for display
        categories = searcher.get_categories(user_id=user.id)
        
        return render_template(
            'recent_final.html',
            results=results,
            categories=categories,
            user=user
        )
    finally:
        conn.close()

# Category management page
@app.route('/categories', methods=['GET'])
def categories():
    """Category management page - now filters by user_id"""
    # Get current user
    user = UserContext.get_current_user()
    
    # Redirect to login if not authenticated
    if not user:
        return redirect(url_for('auth.login', next=request.url))
    
    # Get connection and fetch categories
    conn = get_db_connection()
    try:
        searcher = BookmarkSearchMultiUser(conn, user.id)
        categories = searcher.get_categories(user_id=user.id)
        
        return render_template(
            'categories_final.html',
            categories=categories,
            user=user
        )
    finally:
        conn.close()

# API category list endpoint
@app.route('/api/categories', methods=['GET'])
def api_categories():
    """API endpoint for categories - now filters by user_id"""
    # Get current user
    user = UserContext.get_current_user()
    
    # Return unauthorized for unauthenticated users
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get connection and fetch categories
    conn = get_db_connection()
    try:
        searcher = BookmarkSearchMultiUser(conn, user.id)
        categories = searcher.get_categories(user_id=user.id)
        
        return jsonify(categories)
    finally:
        conn.close()

# Other API endpoints would be similarly updated with user_id filtering
# Including upload-bookmarks, update-database, etc.

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """404 error handler"""
    return render_template('error_final.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    logger.error(f"Server error: {error}")
    return render_template('error_final.html', error='Server error'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 