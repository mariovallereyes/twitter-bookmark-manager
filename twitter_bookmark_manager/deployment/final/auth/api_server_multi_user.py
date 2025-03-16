"""
Multi-user API server for Twitter Bookmark Manager.
Core application file with all routes and functionality.
"""

import os
import sys
import json
import traceback
import logging
import tempfile
import shutil
import time
import random
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from threading import Thread
from logging.handlers import RotatingFileHandler
from urllib.parse import urlencode
import secrets

# Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment from .env file")
except ImportError:
    print("python-dotenv not installed, using environment variables as is")

# Configure environment defaults
ENV_VARS = {
    'DEBUG': os.environ.get('DEBUG', 'false').lower() in ('true', '1', 't'),
    'TESTING': os.environ.get('TESTING', 'false').lower() in ('true', '1', 't'),
    'PORT': int(os.environ.get('PORT', 5000)),
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev_key_for_flask_session'),
    'VECTOR_STORE_DIR': os.environ.get('VECTOR_STORE_DIR', 'vector_store'),
    'QDRANT_HOST': os.environ.get('QDRANT_HOST', 'localhost'),
    'QDRANT_PORT': os.environ.get('QDRANT_PORT', '6333'),
    'QDRANT_GRPC_PORT': os.environ.get('QDRANT_GRPC_PORT', '6334'),
    'DISABLE_VECTOR_STORE': os.environ.get('DISABLE_VECTOR_STORE', 'false').lower() in ('true', '1', 't'),
    'SESSION_TYPE': os.environ.get('SESSION_TYPE', 'filesystem'),
    'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO'),
    'DATABASE_URL': os.environ.get('DATABASE_URL', None),
    'DB_HOST': os.environ.get('DB_HOST', 'localhost'),
    'DB_PORT': os.environ.get('DB_PORT', '5432'),
    'DB_NAME': os.environ.get('DB_NAME', 'twitter_bookmarks'),
    'DB_USER': os.environ.get('DB_USER', 'postgres'),
    'DB_PASSWORD': os.environ.get('DB_PASSWORD', ''),
    'APPLICATION_ROOT': os.environ.get('APPLICATION_ROOT', '/'),
    'UPLOAD_FOLDER': os.environ.get('UPLOAD_FOLDER', 'uploads'),
    'TWITTER_CLIENT_ID': os.environ.get('TWITTER_CLIENT_ID', ''),
    'TWITTER_CLIENT_SECRET': os.environ.get('TWITTER_CLIENT_SECRET', ''),
    'TWITTER_REDIRECT_URI': os.environ.get('TWITTER_REDIRECT_URI', 'https://twitter-bookmark-manager-production.up.railway.app/oauth/callback/twitter'),
}

# Ensure any required env vars are set in os.environ for other modules
for key, value in ENV_VARS.items():
    if key not in os.environ or not os.environ[key]:
        os.environ[key] = str(value)

# Flask imports
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, g, flash, abort, send_from_directory
from flask.sessions import SecureCookieSessionInterface
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import database utilities first to avoid circular imports
from database.multi_user_db.db_final import (
    get_db_connection, 
    init_database,
    cleanup_db_connections,
    check_database_status
)

# Import update functionality
from database.multi_user_db.update_bookmarks_final import (
    final_update_bookmarks
)

# Configure logging
log_level = getattr(logging, ENV_VARS.get('LOG_LEVEL', 'INFO'))
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger('api_server_multi_user')

# Set template folder path to the correct location
template_dir = os.path.join(Path(__file__).parent.parent, 'web_final', 'templates')
static_dir = os.path.join(Path(__file__).parent.parent, 'web_final', 'static')

# Create Flask app with custom template folder
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

# Configure application
app.config.update(
    SECRET_KEY=ENV_VARS['SECRET_KEY'],
    SESSION_TYPE=ENV_VARS['SESSION_TYPE'],
    SESSION_FILE_DIR=os.path.join(tempfile.gettempdir(), 'flask_session'),
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max upload size
    UPLOAD_FOLDER=os.path.join(Path(__file__).parent.parent, ENV_VARS['UPLOAD_FOLDER']),
    DATABASE_DIR=os.path.join(Path(__file__).parent.parent, 'database'),
    DEBUG=ENV_VARS['DEBUG'],
    TESTING=ENV_VARS['TESTING'],
    APPLICATION_ROOT=ENV_VARS['APPLICATION_ROOT'],
    DISABLE_VECTOR_STORE=ENV_VARS['DISABLE_VECTOR_STORE'],
)

# If Railway environment, add session refresh
if os.environ.get('RAILWAY_ENVIRONMENT') == 'production':
    app.config.update(
        SESSION_REFRESH_EACH_REQUEST=True,
    )

# Add database connection function to app config
app.config['get_db_connection'] = get_db_connection

# Import auth modules after database to avoid circular imports
try:
    # Import auth components
    from auth.user_context_final import UserContext, login_required, UserContextMiddleware
    from auth.auth_routes_final import auth_bp
    from auth.user_api_final import user_api_bp
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    logger.info("Registered auth blueprint")
    
    app.register_blueprint(user_api_bp, url_prefix='/api/user')
    logger.info("Registered user API blueprint")
    
    # Set up user context middleware
    from database.multi_user_db.user_model_final import get_user_by_id
    user_middleware = UserContextMiddleware(app, get_user_by_id)
    logger.info("User context middleware initialized")
except Exception as e:
    logger.error(f"Failed to set up authentication: {str(e)}")
    logger.error(traceback.format_exc())

# Enable CORS
CORS(app)

# Ensure upload and database directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)

# Constants
UPLOADS_DIR = app.config['UPLOAD_FOLDER']
DATABASE_DIR = app.config['DATABASE_DIR']

# Set up session management
try:
    from flask_session import Session
    Session(app)
    logger.info("Using Flask-Session for session management")
except ImportError:
    logger.warning("Flask-Session not available, using Flask's default sessions")

# Session status database for background tasks
session_status_db = {}

def init_status_db():
    """Initialize the session status database"""
    global session_status_db
    session_status_db = {}
    logger.info("Initialized session status database")

def save_session_status(session_id, status_data):
    """Save status for a background session"""
    global session_status_db
    session_status_db[session_id] = status_data
    # Prune old entries
    now = datetime.now()
    old_sessions = [sid for sid, data in session_status_db.items() 
                   if 'timestamp' in data and 
                   (now - datetime.fromisoformat(data['timestamp'])).total_seconds() > 86400]
    for sid in old_sessions:
        del session_status_db[sid]
    logger.info(f"Saved session status for {session_id}")

def get_session_status(session_id):
    """Get status for a background session"""
    try:
        status = session_status_db.get(session_id)
        if not status:
            # Try to read from file if not in memory
            user = UserContext.get_current_user()
            if not user:
                logger.warning(f"No user context for status request: {session_id}")
                return None
                
            user_dir = get_user_directory(user.id)
            status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
            
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                        # Add to in-memory cache
                        session_status_db[session_id] = status
                except Exception as e:
                    logger.error(f"Error reading status file: {e}")
            return None
        return status
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return None

def cleanup_old_sessions():
    """Clean up old session status data"""
    try:
        # Clean up in-memory status db
        global session_status_db
        now = datetime.now()
        old_sessions = [sid for sid, data in session_status_db.items() 
                       if 'timestamp' in data and 
                       (now - datetime.fromisoformat(data['timestamp'])).total_seconds() > 86400]
        for sid in old_sessions:
            del session_status_db[sid]
            
        # Schedule next cleanup
        t = Thread(target=lambda: time.sleep(3600) and cleanup_old_sessions())
        t.daemon = True
        t.start()
    except Exception as e:
        logger.error(f"Error cleaning up old sessions: {e}")

# Initialize session status DB
init_status_db()

# Start cleanup thread
cleanup_thread = Thread(target=cleanup_old_sessions)
cleanup_thread.daemon = True
cleanup_thread.start()

# Register teardown function
@app.teardown_appcontext
def shutdown_cleanup(exception=None):
    """Clean up resources when the application context ends"""
    if exception:
        logger.error(f"Error during context teardown: {str(exception)}")
    logger.debug("Application context tearing down, cleaning up resources")
    
    # Clean up database connections
    cleanup_db_connections()

# Request handlers
@app.before_request
def log_request_details():
    """Log details about each request for debugging"""
    # Log request details
    logger.info(f"Request: {request.method} {request.path}")
    logger.info(f"Blueprint: {request.blueprint}")
    logger.info(f"Endpoint: {request.endpoint}")
    logger.info(f"Session keys: {list(session.keys())}")
    # Don't log sensitive data like cookies or headers with auth info

@app.before_request
def log_session_info():
    """Log session information for debugging"""
    try:
        # Log basic request info
        if request.path not in ['/status', '/-/health']:  # Skip logging for health checks
            user_agent = request.headers.get('User-Agent', 'Unknown')
            logger.debug(f"Request: {request.method} {request.path} - UA: {user_agent[:50]}...")
    except Exception as e:
        logger.error(f"Error logging session info: {e}")

@app.before_request
def check_user_authentication():
    """Check if user is authenticated and redirect to login if not"""
    # Skip auth check for certain paths
    if request.path.startswith('/static/') or \
       request.path in ['/login', '/login/twitter', '/logout', '/oauth/callback/twitter', 
                        '/check-auth', '/-/health', '/debug/auth-config', '/debug/session', 
                        '/debug/twitter-oauth', '/debug/session-config'] or \
       request.path.startswith('/auth/'):
        return
    
    # Check authentication
    if hasattr(g, 'user') and g.user is not None:
        return
    
    # Redirect to login unless it's an API request
    if request.path.startswith('/api/'):
        return jsonify({'error': 'User not authenticated', 'authenticated': False}), 401
    
    # Store the requested URL in the session for post-login redirect
    session['next'] = request.url
    
    # Log warning
    logger.info('User not authenticated, redirecting to login')
    
    # Direct URL redirect instead of using url_for
    return redirect('/login')
    """Check if the user is authenticated and redirect to login if not"""
    # Skip authentication for login, logout, static files and health check routes
    auth_exempt_paths = [
        '/auth/login', 
        '/auth/twitter',
        '/oauth/callback/twitter',
        '/auth/twitter_auth',
        '/auth/logout', 
        '/static/', 
        '/favicon.ico',
        '/-/health',
        '/api/health',
        '/check-auth',
        '/debug/'
    ]
    
    # Skip authentication for exempt paths
    if any(request.path.startswith(path) for path in auth_exempt_paths):
        logger.debug(f"Skipping authentication check for {request.path}")
        return
    
    # Skip authentication for auth blueprint routes
    if request.blueprint == 'auth':
        logger.debug(f"Skipping authentication check for auth blueprint: {request.path}")
        return
        
    # Check if user is authenticated
    user_id = session.get('user_id')
    if not user_id:
        logger.info(f"User not authenticated, redirecting to login")
        # Store the current URL for redirect after login
        if request.endpoint and 'static' not in request.endpoint:
            session['next'] = request.url
        # Return redirect response
        return redirect(url_for('auth.login'))
    
    # Log authenticated access
    logger.debug(f"Authenticated access by user {user_id} to {request.path}")

@app.before_request
def check_db_health():
    """Check database health before processing request"""
    try:
        # Skip health check for static files and status endpoints
        if request.path.startswith('/static') or request.path in ['/status', '/-/health']:
            return
            
        # Check if we have a DB connection function
        if 'get_db_connection' not in app.config:
            logger.error("Database connection function not configured")
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Database not configured'}), 500
            else:
                flash("Database connection not available", "error")
                return render_template('error_final.html', error="Database connection not available")
    
    except Exception as e:
        logger.error(f"Error checking database health: {e}")
        # Let the request continue - the error will be handled at the specific endpoint

@app.before_request
def make_session_permanent():
    """Make the session permanent to avoid frequent re-logins"""
    try:
        session.permanent = True
    except Exception as e:
        logger.error(f"Error making session permanent: {e}")

@app.before_request
def ensure_json_response():
    """Ensure all API endpoints return proper JSON responses"""
    try:
        # For API endpoints, add JSON headers
        if request.path.startswith('/api/'):
            if request.method != 'OPTIONS':  # Skip for CORS preflight
                g.return_json = True
    except Exception as e:
        logger.error(f"Error setting up JSON response: {e}")

# Initialize app in debug mode
def init_app_debug():
    """Initialize the app with debug data if needed"""
    try:
        # Check if database is initialized
        db_status = check_database_status()
        
        if db_status.get('healthy', False):
            logger.info("Database is healthy")
        else:
            logger.warning(f"Database may have issues: {db_status.get('message', 'Unknown')}")
            
        # Ensure directories exist
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        os.makedirs(DATABASE_DIR, exist_ok=True)
        logger.info(f"Ensured directories exist: {UPLOADS_DIR}, {DATABASE_DIR}")
        
        # Initialize the database
        init_database()
        
    except Exception as e:
        logger.error(f"Error in debug initialization: {e}")
        logger.error(traceback.format_exc())

# Call debug initialization
init_app_debug()

# Helper functions
def get_user_directory(user_id):
    """Get the user's data directory, creating it if it doesn't exist"""
    try:
        user_dir = os.path.join(DATABASE_DIR, f'user_{user_id}')
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    except Exception as e:
        logger.error(f"Error creating user directory: {e}")
        return None

# Decorator for vector store operations
def safe_vector_operation(func):
    """Decorator to safely handle vector store operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as re:
            error_str = str(re)
            if "Storage folder" in error_str and "already accessed by another instance" in error_str:
                logger.error(f"Vector store access error in {func.__name__}: {error_str}")
                return jsonify({
                    'success': False,
                    'error': "Vector database is busy. Please try again in a few minutes.",
                    'retry_after': 300
                }), 503
            raise
    return wrapper

# Safe vector store getter
def safe_get_vector_store():
    """Safely attempt to get a vector store instance with retry logic."""
    MAX_RETRIES = 3
    initial_backoff = 1
    session_id = str(uuid.uuid4())[:8]
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"[RETRY-{session_id}] Attempting to import VectorStoreMultiUser (attempt {attempt+1}/{MAX_RETRIES})")
            from database.multi_user_db.vector_store_final import VectorStoreMultiUser, DummyVectorStore
            logger.info(f"[RETRY-{session_id}] Successfully imported VectorStoreMultiUser")
            
            user = UserContext.get_current_user()
            if not user:
                logger.warning(f"[RETRY-{session_id}] No user context found, cannot initialize vector store")
                return None
                
            user_id = user.id
            
            # Get the vector store instance
            try:
                from database.multi_user_db.vector_store_final import get_multi_user_vector_store
                vector_store = get_multi_user_vector_store()
                
                # Check if we got a dummy vector store
                if hasattr(vector_store, 'is_dummy') and vector_store.is_dummy:
                    logger.warning(f"[RETRY-{session_id}] Received DummyVectorStore: {vector_store.error_message}")
                    # Return the dummy store, it's designed to handle operations gracefully
                    return vector_store
                    
                logger.info(f"[RETRY-{session_id}] Successfully initialized vector store for user {user_id} on attempt {attempt+1}")
                return vector_store
            except Exception as vs_error:
                logger.error(f"[RETRY-{session_id}] Error initializing vector store: {vs_error}")
                logger.error(traceback.format_exc())
                if attempt < MAX_RETRIES - 1:
                    backoff = initial_backoff * (2 ** attempt)
                    logger.info(f"[RETRY-{session_id}] Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                else:
                    logger.error(f"[RETRY-{session_id}] Failed to initialize vector store after {MAX_RETRIES} attempts")
                    # Return the dummy store as a last resort
                    return DummyVectorStore(f"Failed after {MAX_RETRIES} attempts: {vs_error}")
        
        except ImportError as e:
            logger.error(f"[RETRY-{session_id}] Import error for VectorStoreMultiUser: {e}")
            logger.error(traceback.format_exc())
            if attempt < MAX_RETRIES - 1:
                backoff = initial_backoff * (2 ** attempt)
                logger.info(f"[RETRY-{session_id}] Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                logger.error(f"[RETRY-{session_id}] Failed to initialize vector store after {MAX_RETRIES} attempts")
                return None
        except Exception as e:
            logger.error(f"[RETRY-{session_id}] Unexpected error: {e}")
            logger.error(traceback.format_exc())
            if attempt < MAX_RETRIES - 1:
                backoff = initial_backoff * (2 ** attempt)
                logger.info(f"[RETRY-{session_id}] Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                logger.error(f"[RETRY-{session_id}] Failed to initialize vector store after {MAX_RETRIES} attempts")
                return None
    
    return None

# Routes
@app.route('/')
def index():
    """Home page with multiple database connection fallbacks."""
    logger.info("Home page requested")
    try:
        ensure_string_session()
    except Exception as session_error:
        logger.error(f"Error clearing session: {session_error}")
    try:
        user = UserContext.get_current_user()
        if user:
            logger.info(f"Authenticated user: ID={user.id}, Name={getattr(user, 'username', 'unknown')}")
        else:
            logger.info("No authenticated user")
        
        if user:
            # User is authenticated, load the main page
            try:
                # Get user ID
                user_id = user.id
                
                # Get database connection
                conn = get_db_connection()
                
                # Create bookmark search instance
                from database.multi_user_db.search_final_multi_user import BookmarkSearchMultiUser
                searcher = BookmarkSearchMultiUser(conn, user_id)
                
                # Get recent bookmarks - limit to 5 for homepage display
                latest_tweets = searcher.get_recent_bookmarks(limit=5, user_id=user_id)
                
                # Get categories for the navigation
                categories = searcher.get_categories(user_id=user_id)
                
                # Debug logging to check structure and content
                logger.info(f"Found {len(latest_tweets)} recent bookmarks for user {user_id}")
                logger.info(f"Found {len(categories)} categories for user {user_id}")
                
                if latest_tweets:
                    # Log first bookmark to verify structure
                    first_tweet = latest_tweets[0]
                    logger.info(f"First bookmark: id={first_tweet.get('id')}, author={first_tweet.get('author_username')}")
                    
                    # Ensure all required fields are present
                    for i, tweet in enumerate(latest_tweets):
                        if not tweet.get('id'):
                            tweet['id'] = f"missing-id-{i}"
                        if not tweet.get('author_username'):
                            tweet['author_username'] = "unknown_user"
                        if not tweet.get('categories'):
                            tweet['categories'] = []
                else:
                    logger.warning("No bookmarks found in database for this user")
                
                # Render template with bookmarks and categories
                return render_template('index_final.html', latest_tweets=latest_tweets, categories=categories)
            except Exception as e:
                logger.error(f"Error fetching bookmarks: {e}")
                logger.error(traceback.format_exc())
                # Fallback to render without bookmarks
                return render_template('index_final.html', latest_tweets=[], categories=[])
        else:
            # User is not authenticated, redirect to login
            logger.info("User not authenticated, redirecting to login")
            try:
                return redirect('/auth/login')
            except Exception as redirect_error:
                logger.error(f"Error redirecting to login: {redirect_error}")
                return render_template('login_final.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.route('/chat')
@login_required
def chat():
    """Chat interface page."""
    logger.info("Chat page requested")
    try:
        user = UserContext.get_current_user()
        return render_template('chat_final.html')
    except Exception as e:
        logger.error(f"Error rendering chat page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.route('/search')
@login_required
def search():
    """Search interface page."""
    logger.info("Search page requested")
    try:
        return render_template('search_final.html')
    except Exception as e:
        logger.error(f"Error rendering search page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.route('/recent')
@login_required
def recent():
    """Recent bookmarks page."""
    logger.info("Recent bookmarks page requested")
    try:
        return render_template('recent_final.html')
    except Exception as e:
        logger.error(f"Error rendering recent page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    """Chat API endpoint."""
    try:
        user = UserContext.get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
            
        user_id = user.id
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
            
        message = data.get('message')
        context = data.get('context', [])
        logger.info(f"Chat request from user {user_id}: {message[:50]}...")
        
        # Get vector store with retry
        vector_store = safe_get_vector_store()
        if vector_store is None:
            return jsonify({
                'success': False, 
                'error': 'Vector store not available', 
                'response': 'I apologize, but the vector search service is currently unavailable. Please try again later.'
            }), 503
            
        # Check if we're using a dummy vector store
        if hasattr(vector_store, 'is_dummy') and vector_store.is_dummy:
            # Return a helpful message instead of an error
            return jsonify({
                'success': True,
                'response': 'I apologize, but the vector search service is temporarily unavailable. Your bookmarks have been saved and will be searchable when the service is restored.',
                'is_fallback': True,
                'similar_tweets': []
            })
            
        # Search for similar tweets
        try:
            similar_results = vector_store.find_similar(message, limit=5)
            
            # Format response
            response = f"Here are some bookmarks related to your question:"
            similar_tweets = []
            
            for i, result in enumerate(similar_results):
                if result and isinstance(result, dict):
                    tweet_text = result.get('text', '').strip()
                    tweet_url = result.get('url', '#')
                    tweet_id = result.get('id', '')
                    similar_tweets.append({
                        'id': tweet_id,
                        'text': tweet_text,
                        'url': tweet_url,
                        'similarity': result.get('similarity', 0)
                    })
            
            return jsonify({
                'success': True,
                'response': response,
                'similar_tweets': similar_tweets
            })
        except Exception as search_error:
            logger.error(f"Error searching for similar tweets: {search_error}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(search_error),
                'response': 'I encountered an error while searching your bookmarks.'
            }), 500
    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
@login_required
def api_search():
    """Search API endpoint."""
    try:
        user = UserContext.get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
            
        user_id = user.id
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
            
        query = data.get('query')
        limit = int(data.get('limit', 10))
        logger.info(f"Search request from user {user_id}: {query[:50]}...")
        
        # Get vector store with retry
        vector_store = safe_get_vector_store()
        if vector_store is None:
            return jsonify({
                'success': False, 
                'error': 'Vector store not available'
            }), 503
            
        # Check if we're using a dummy vector store
        if hasattr(vector_store, 'is_dummy') and vector_store.is_dummy:
            # Return empty results instead of an error
            return jsonify({
                'success': True,
                'results': [],
                'count': 0,
                'is_fallback': True,
                'message': 'Search service is temporarily unavailable. Your bookmarks have been saved and will be searchable when the service is restored.'
            })
            
        # Search for similar tweets
        try:
            results = vector_store.find_similar(query, limit=limit)
            
            # Format response
            formatted_results = []
            for result in results:
                if result and isinstance(result, dict):
                    tweet_text = result.get('text', '').strip()
                    tweet_url = result.get('url', '#')
                    tweet_id = result.get('id', '')
                    formatted_results.append({
                        'id': tweet_id,
                        'text': tweet_text,
                        'url': tweet_url,
                        'similarity': result.get('similarity', 0)
                    })
            
            return jsonify({
                'success': True, 
                'results': formatted_results,
                'count': len(formatted_results)
            })
        except Exception as search_error:
            logger.error(f"Error searching for tweets: {search_error}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(search_error)
            }), 500
    except Exception as e:
        logger.error(f"Error in search API: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recent', methods=['GET'])
@login_required
def api_recent():
    """Get recent bookmarks."""
    try:
        user = UserContext.get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
        
        user_id = user.id
        limit = int(request.args.get('limit', 10))
        logger.info(f"Recent bookmarks request from user {user_id}, limit {limit}")
        
        try:
            # Connect to database and query recent bookmarks
            db_conn = get_db_connection()
            cursor = db_conn.cursor()
            
            # Query for user's recent bookmarks
            cursor.execute("""
                SELECT b.id, b.text, b.url, b.timestamp 
                FROM bookmarks b
                JOIN user_bookmarks ub ON b.id = ub.bookmark_id
                WHERE ub.user_id = %s
                ORDER BY b.timestamp DESC
                LIMIT %s
            """, (user_id, limit))
            
            results = cursor.fetchall()
            
            # Format response
            bookmarks = []
            for row in results:
                bookmarks.append({
                    'id': row[0],
                    'text': row[1],
                    'url': row[2],
                    'timestamp': row[3].isoformat() if row[3] else None
                })
            
            cursor.close()
            return jsonify({
                'success': True,
                'bookmarks': bookmarks,
                'count': len(bookmarks)
            })
        except Exception as db_error:
            logger.error(f"Error querying recent bookmarks: {db_error}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(db_error)
            }), 500
    except Exception as e:
        logger.error(f"Error in recent API: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/update-database')
@login_required
def update_database():
    """Update database page."""
    logger.info("Update database page requested")
    try:
        return render_template('update_database_final.html')
    except Exception as e:
        logger.error(f"Error rendering update database page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.route('/rebuild-vector-store')
@login_required
def rebuild_vector_store():
    """Rebuild vector store page."""
    logger.info("Rebuild vector store page requested")
    try:
        return render_template('rebuild_vector_store_final.html')
    except Exception as e:
        logger.error(f"Error rendering rebuild vector store page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.route('/api/rebuild-vector-store', methods=['POST'])
@login_required
@safe_vector_operation
def api_rebuild_vector_store():
    """API endpoint to rebuild the vector store."""
    try:
        user = UserContext.get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
            
        user_id = user.id
        logger.info(f"Rebuild vector store request from user {user_id}")
        
        # Generate a session ID for status tracking
        session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        
        # Start processing in background
        def rebuild_task():
            try:
                with app.app_context():
                    # Save initial status
                    save_session_status(session_id, {
                        'status': 'processing',
                        'message': 'Starting vector store rebuild...',
                        'progress': 0,
                        'timestamp': datetime.now().isoformat(),
                        'user_id': user_id
                    })
                    
                    try:
                        # Get vector store with retry
                        vector_store = safe_get_vector_store()
                        if vector_store is None or (hasattr(vector_store, 'is_dummy') and vector_store.is_dummy):
                            logger.error(f"Vector store not available for rebuild")
                            save_session_status(session_id, {
                                'status': 'error',
                                'error': 'Vector store not available',
                                'timestamp': datetime.now().isoformat(),
                                'user_id': user_id
                            })
                            return
                            
                        # Rebuild vector store
                        logger.info(f"Starting vector store rebuild for user {user_id}")
                        result = vector_store.rebuild_user_vectors(user_id)
                        
                        logger.info(f"Vector store rebuild completed for user {user_id}: {result}")
                        save_session_status(session_id, {
                            'status': 'completed',
                            'result': result,
                            'timestamp': datetime.now().isoformat(),
                            'user_id': user_id
                        })
                    except Exception as e:
                        logger.error(f"Error rebuilding vector store: {e}")
                        logger.error(traceback.format_exc())
                        save_session_status(session_id, {
                            'status': 'error',
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                            'user_id': user_id
                        })
            except Exception as e:
                logger.error(f"Unexpected error in rebuild task: {e}")
                logger.error(traceback.format_exc())
        
        # Start processing thread
        thread = Thread(target=rebuild_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Vector store rebuild started',
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error starting vector store rebuild: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rebuild-status', methods=['GET'])
@login_required
def api_rebuild_status():
    """Check status of vector store rebuild."""
    return process_status()

@app.route('/api/statistics', methods=['GET'])
@login_required
def api_statistics():
    """Get user statistics."""
    try:
        user = UserContext.get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
            
        user_id = user.id
        logger.info(f"Statistics request from user {user_id}")
        
        # Connect to database
        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        
        # Get bookmark count
        cursor.execute("""
            SELECT COUNT(bookmark_id) FROM user_bookmarks
            WHERE user_id = %s
        """, (user_id,))
        bookmark_count = cursor.fetchone()[0]
        
        # Get vector store stats if available
        vector_stats = {}
        try:
            vector_store = safe_get_vector_store()
            if vector_store and not (hasattr(vector_store, 'is_dummy') and vector_store.is_dummy):
                collection_info = vector_store.get_collection_info()
                if collection_info:
                    vector_stats = {
                        'vectors': collection_info.get('vector_count', 0),
                        'size': collection_info.get('vector_size', 0),
                        'collection': collection_info.get('collection_name', 'unknown')
                    }
        except Exception as vs_error:
            logger.error(f"Error getting vector stats: {vs_error}")
            vector_stats = {'error': str(vs_error)}
        
        cursor.close()
        return jsonify({
            'success': True,
            'statistics': {
                'bookmark_count': bookmark_count,
                'vector_store': vector_stats,
                'user_id': user_id
            }
        })
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

# File upload route
@app.route('/upload-bookmarks', methods=['POST'])
@login_required
@safe_vector_operation
def upload_bookmarks():
    """Handle bookmark file uploads."""
    try:
        user = UserContext.get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
            
        user_id = user.id
        logger.info(f"Processing bookmark upload for user {user_id}")
        
        # Check if file is in the request
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return render_template('upload.html', error="No file part")
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logger.warning("No selected file")
            return render_template('upload.html', error="No selected file")
            
        if file:
            # Create uploads directory if it doesn't exist
            user_upload_dir = os.path.join(UPLOADS_DIR, f'user_{user_id}')
            os.makedirs(user_upload_dir, exist_ok=True)
            
            # Save file with timestamp in filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(user_upload_dir, filename)
            file.save(filepath)
            logger.info(f"File saved to {filepath}")
            
            # Generate a session ID for status tracking
            session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            
            # Start processing in background
            def process_task():
                try:
                    with app.app_context():
                        # Process the file
                        logger.info(f"Starting bookmark processing for session {session_id}")
                        try:
                            result = final_update_bookmarks(
                                session_id=session_id,
                                start_index=0,
                                rebuild_vector=True,
                                user_id=user_id,
                                skip_vector=True
                            )
                            logger.info(f"Bookmark processing completed for session {session_id}: {result}")
                            save_session_status(session_id, {
                                'status': 'completed',
                                'result': result,
                                'timestamp': datetime.now().isoformat(),
                                'user_id': user_id
                            })
                        except Exception as e:
                            logger.error(f"Error processing bookmarks: {e}")
                            logger.error(traceback.format_exc())
                            save_session_status(session_id, {
                                'status': 'error',
                                'error': str(e),
                                'timestamp': datetime.now().isoformat(),
                                'user_id': user_id
                            })
                except Exception as e:
                    logger.error(f"Unexpected error in process task: {e}")
                    logger.error(traceback.format_exc())
            
            # Start processing thread
            thread = Thread(target=process_task)
            thread.daemon = True
            thread.start()
            
            # Redirect to status page
            return redirect(url_for('process_status', session_id=session_id))
            
    except Exception as e:
        logger.error(f"Error in upload_bookmarks: {e}")
        logger.error(traceback.format_exc())
        return render_template('upload.html', error=str(e))

# Process status route
@app.route('/api/process-status', methods=['GET'])
def process_status():
    """Check status of background processing."""
    try:
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user context found for status request")
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
        user_id = user.id
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'No session_id provided'}), 400
            
        process_type = request.args.get('type', 'upload')
        logger.info(f"Checking {process_type} status for session {session_id}, user {user_id}")
        
        # Get status from memory or file
        status = get_session_status(session_id)
        
        if not status:
            return jsonify({
                'success': True,
                'status': 'unknown',
                'message': 'Processing status unknown',
                'session_id': session_id
            })
            
        # Check for error
        if status.get('status') == 'error':
            return jsonify({
                'success': False,
                'status': 'error',
                'message': status.get('error', 'Unknown error'),
                'session_id': session_id
            })
            
        # Get progress
        result = status.get('result', {})
        return jsonify({
            'success': True,
            'status': status.get('status', 'processing'),
            'message': result.get('message', status.get('message', 'Processing...')),
            'progress': result.get('progress', 0),
            'session_id': session_id,
            'result': result
        })
    except Exception as e:
        logger.error(f"Error checking process status: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': str(e),
            'type': e.__class__.__name__
        }), 500
    
    return render_template('error_final.html', error=str(e)), 500

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    logger.warning(f"404 Not Found: {request.path}")
    
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Resource not found',
            'path': request.path
        }), 404
    
    return render_template('404.html'), 404

# Health endpoint
@app.route('/status')
def app_status():
    """Application status endpoint"""
    try:
        # Check database health
        db_status = check_database_status()
        
        # Check user authentication
        user = UserContext.get_current_user()
        auth_status = {'authenticated': user is not None}
        if user:
            auth_status['user_id'] = user.id
            auth_status['username'] = getattr(user, 'username', 'unknown')
        
        return jsonify({
            'status': 'healthy',
            'database': db_status,
            'auth': auth_status,
            'environment': os.environ.get('RAILWAY_ENVIRONMENT', 'unknown'),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/-/health')
def health_check():
    """Simple health check for container orchestration"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload')
@login_required
def upload():
    """Upload bookmarks page."""
    logger.info("Upload page requested")
    try:
        return render_template('upload_final.html')
    except Exception as e:
        logger.error(f"Error rendering upload page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.route('/upload-status')
@login_required
def upload_status():
    """Upload status page."""
    try:
        session_id = request.args.get('session_id', '')
        if not session_id:
            logger.warning("No session ID provided for upload status")
            return redirect(url_for('upload'))
            
        logger.info(f"Upload status page requested for session {session_id}")
        return render_template('upload_status_final.html', session_id=session_id)
    except Exception as e:
        logger.error(f"Error rendering upload status page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.route('/debug-data', methods=['GET'])
@login_required
def debug_data():
    """Debug data page for administrators only."""
    try:
        user = UserContext.get_current_user()
        if not user or not getattr(user, 'is_admin', False):
            logger.warning(f"Unauthorized debug access attempt by user {getattr(user, 'id', 'unknown')}")
            return redirect(url_for('index'))
            
        # Get system info
        system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'app_directory': os.path.abspath(os.path.dirname(__file__)),
            'upload_directory': os.path.abspath(UPLOADS_DIR),
            'database_directory': os.path.abspath(DATABASE_DIR),
            'environment': {k: v for k, v in os.environ.items() if k.startswith(('PYTHONPATH', 'SECRET_KEY', 'DB_', 'QDRANT_', 'PORT', 'HOST'))},
            'session_keys': list(session.keys()) if session else []
        }
        
        # Database connection status
        try:
            db_status = check_database_status()
            system_info['database_status'] = db_status
        except Exception as db_error:
            system_info['database_status'] = {'error': str(db_error)}
            
        # Vector store status
        try:
            # Try to import and check vector store
            from database.multi_user_db.vector_store_final import get_multi_user_vector_store
            vector_store = safe_get_vector_store()
            if vector_store:
                if hasattr(vector_store, 'is_dummy') and vector_store.is_dummy:
                    system_info['vector_store_status'] = {
                        'status': 'dummy',
                        'error': getattr(vector_store, 'error_message', 'Unknown error')
                    }
                else:
                    system_info['vector_store_status'] = {
                        'status': 'connected',
                        'info': vector_store.get_collection_info() if hasattr(vector_store, 'get_collection_info') else {}
                    }
            else:
                system_info['vector_store_status'] = {'status': 'not_connected'}
        except Exception as vs_error:
            system_info['vector_store_status'] = {'status': 'error', 'error': str(vs_error)}
            
        return render_template('debug_data_final.html', system_info=system_info)
    except Exception as e:
        logger.error(f"Error rendering debug page: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/process-status')
@login_required
def process_status_page():
    """Process status page for background operations."""
    try:
        session_id = request.args.get('session_id', '')
        if not session_id:
            logger.warning("No session ID provided for process status")
            return redirect(url_for('index'))
            
        return render_template('process_status_final.html', session_id=session_id)
    except Exception as e:
        logger.error(f"Error rendering process status page: {e}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file too large errors"""
    logger.warning("File upload too large")
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'File too large'
        }), 413
        
    return render_template('upload.html', error="The file you tried to upload is too large. Maximum size is 50MB."), 413

# Add health check endpoint
@app.route('/api/health')
def api_health_check():
    """Simple health check endpoint"""
    try:
        # Check if we can get a database connection
        from database.multi_user_db.db_final import get_db_connection, check_database_status
        
        # Check database status
        db_status = check_database_status()
        
        # Check vector store status if not disabled
        vector_store_status = {"status": "disabled"}
        if not ENV_VARS['DISABLE_VECTOR_STORE']:
            try:
                from database.multi_user_db.vector_store_final import get_vector_store
                vs = get_vector_store(allow_dummy=True)
                collection_info = vs.get_collection_info() if vs else None
                vector_store_status = {
                    "status": "healthy" if vs and not getattr(vs, 'is_dummy', False) else "fallback",
                    "info": collection_info
                }
            except Exception as vs_error:
                vector_store_status = {
                    "status": "error",
                    "message": str(vs_error)
                }
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "vector_store": vector_store_status,
            "environment": {
                "debug": ENV_VARS['DEBUG'],
                "testing": ENV_VARS['TESTING'],
                "disable_vector_store": ENV_VARS['DISABLE_VECTOR_STORE']
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
    
# Force disable vector store
os.environ['DISABLE_VECTOR_STORE'] = 'true'

# Modify Python path to avoid circular imports
import sys
if not hasattr(sys, '_vector_store_disabled'):
    sys._vector_store_disabled = True
    # Avoid trying to import vector store modules if disabled
    class DummyModule:
        def __getattr__(self, name):
            return None
    
    # Register dummy modules for potentially problematic imports
    vector_store_modules = [
        'database.multi_user_db.vector_store_final',
        'database.multi_user_db.minimal_vector_store'
    ]
    for module_name in vector_store_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = DummyModule()
    
    logger.info("Registered dummy modules to prevent import errors")

# Check authentication status endpoint with improved error handling
@app.route('/check-auth')
def check_auth():
    """Check if the user is authenticated and return user info"""
    try:
        user_id = session.get('user_id')
        logger.info(f"Checking authentication for user_id: {user_id}")
        
        # Convert from bytes to string if needed
        if isinstance(user_id, bytes):
            try:
                user_id = user_id.decode('utf-8')
                # Update session
                session['user_id'] = user_id
                logger.info(f"Converted user_id from bytes to string: {user_id}")
            except Exception as e:
                logger.error(f"Error decoding user_id from bytes: {e}")
                user_id = None
        
        if user_id:
            # Get user info from database
            db = get_db_connection()
            from database.multi_user_db.user_model_final import get_user_by_id
            user = get_user_by_id(db, user_id)
            
            if user:
                logger.info(f"User authenticated: {user.username}")
                return jsonify({
                    'authenticated': True,
                    'user': {
                        'id': user.id,
                        'username': user.username,
                        'name': getattr(user, 'display_name', user.username),
                        'avatar_url': getattr(user, 'profile_image_url', None)
                    }
                })
            else:
                logger.warning(f"User ID {user_id} not found in database")
        
        # If no user_id in session or user not found
        logger.info("User not authenticated")
        return jsonify({
            'authenticated': False,
            'message': 'User not authenticated'
        })
    except Exception as e:
        logger.error(f"Error in check_auth: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'authenticated': False,
            'error': str(e)
        }), 500

# API categories endpoint - map to the user API endpoint
@app.route('/api/categories', methods=['GET'])
@login_required
def api_categories():
    """Get categories - proxy to user API endpoint"""
    try:
        from auth.user_api_final import get_categories
        return get_categories()
    except Exception as e:
        logger.error(f"Error in api_categories proxy: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving categories: {str(e)}'
        }), 500

# Route to handle Twitter OAuth callback at the root level
@app.route('/oauth/callback/twitter')
def twitter_oauth_callback():
    """Route to handle Twitter OAuth callback at the root level"""
    logger.info("-------- Twitter OAuth callback received at root level --------")
    logger.info(f"Full request URL: {request.url}")
    logger.info(f"Request args: {dict(request.args)}")
    logger.info(f"Session keys before processing: {list(session.keys())}")
    
    # Import the specific Twitter callback handler
    try:
        from auth.auth_routes_final import oauth_callback_twitter
        logger.info("Successfully imported oauth_callback_twitter function")
        return oauth_callback_twitter()
    except Exception as e:
        logger.error(f"Error in root twitter_oauth_callback: {e}")
        logger.error(traceback.format_exc())
        
        # Check if we have the specific error about the import
        if "No module named" in str(e) or "cannot import name" in str(e):
            logger.error("Import error - trying to handle callback directly")
            # Add code here to handle the callback directly if needed
            return redirect(url_for('auth.login'))
        
        return redirect(url_for('auth.login'))

# Debug endpoint to check Twitter auth configuration
@app.route('/debug/auth-config')
def debug_auth_config():
    """Debugging endpoint to check Twitter auth configuration"""
    # Only allow in debug mode
    if not ENV_VARS['DEBUG']:
        return jsonify({'error': 'Debug mode not enabled'}), 403
        
    try:
        # Check for Twitter credentials
        twitter_client_id = os.environ.get('TWITTER_CLIENT_ID', '')
        twitter_client_secret = os.environ.get('TWITTER_CLIENT_SECRET', '')
        twitter_redirect_uri = os.environ.get('TWITTER_REDIRECT_URI', '')
        
        # Check if auth blueprint is registered
        blueprints = [bp.name for bp in app.blueprints.values()]
        
        # Check if auth components are loaded
        auth_components = {}
        try:
            from auth.auth_routes_final import auth_bp
            auth_components['auth_routes'] = True
        except Exception as e:
            auth_components['auth_routes'] = False
            auth_components['auth_routes_error'] = str(e)
            
        try:
            from auth.user_context_final import UserContext
            auth_components['user_context'] = True
        except Exception as e:
            auth_components['user_context'] = False
            auth_components['user_context_error'] = str(e)
            
        try:
            from auth.oauth_final import OAuthManager
            oauth_manager = OAuthManager({
                'twitter': {
                    'client_id': 'REDACTED',
                    'client_secret': 'REDACTED',
                    'callback_url': twitter_redirect_uri
                }
            })
            auth_components['oauth_manager'] = True
        except Exception as e:
            auth_components['oauth_manager'] = False
            auth_components['oauth_manager_error'] = str(e)
        
        return jsonify({
            'twitter_auth': {
                'client_id_present': bool(twitter_client_id),
                'client_secret_present': bool(twitter_client_secret),
                'redirect_uri': twitter_redirect_uri
            },
            'blueprints': blueprints,
            'auth_components': auth_components,
            'app_config': {
                'secret_key_present': bool(app.config.get('SECRET_KEY')),
                'session_type': app.config.get('SESSION_TYPE'),
                'template_folder': app.template_folder,
                'debug': app.config.get('DEBUG')
            }
        })
    except Exception as e:
        logger.error(f"Error in debug_auth_config: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Debug endpoint to check session state
@app.route('/debug/session')
def debug_session():
    """Debugging endpoint to check session state"""
    # Only allow in debug mode
    if not ENV_VARS['DEBUG']:
        return jsonify({'error': 'Debug mode not enabled'}), 403
        
    try:
        # Get session details
        session_data = {
            'keys': list(session.keys()),
            'user_id': session.get('user_id'),
            'next': session.get('next'),
            'twitter_code_verifier_exists': 'twitter_code_verifier' in session,
            'twitter_oauth_state_exists': 'twitter_oauth_state' in session,
            'sid': getattr(session, 'sid', None)
        }
        
        return jsonify({
            'session': session_data,
            'headers': {
                'cookie': request.headers.get('Cookie'),
                'user_agent': request.headers.get('User-Agent')
            },
            'request': {
                'path': request.path,
                'url': request.url,
                'method': request.method,
                'remote_addr': request.remote_addr
            }
        })
    except Exception as e:
        logger.error(f"Error in debug_session: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Debug Twitter OAuth endpoint
@app.route('/debug/twitter-oauth')
def debug_twitter_oauth():
    """Debug endpoint for Twitter OAuth configuration"""
    # Only allow in debug mode
    if not ENV_VARS['DEBUG']:
        return jsonify({'error': 'Debug mode not enabled'}), 403
        
    try:
        # Get Twitter OAuth configuration
        twitter_client_id = os.environ.get('TWITTER_CLIENT_ID', '')
        twitter_client_secret = os.environ.get('TWITTER_CLIENT_SECRET', '')
        twitter_redirect_uri = os.environ.get('TWITTER_REDIRECT_URI', '')
        
        # Create OAuth manager and generate URL
        from auth.oauth_final import OAuthManager
        config = {
            'twitter': {
                'client_id': twitter_client_id,
                'client_secret': twitter_client_secret,
                'callback_url': twitter_redirect_uri
            }
        }
        
        oauth_manager = OAuthManager(config)
        auth_url = oauth_manager.get_authorize_url('twitter')
        
        return jsonify({
            'twitter_oauth': {
                'client_id_exists': bool(twitter_client_id),
                'client_id_prefix': twitter_client_id[:5] + '...' if twitter_client_id else '',
                'client_secret_exists': bool(twitter_client_secret),
                'redirect_uri': twitter_redirect_uri,
                'auth_url_generated': bool(auth_url),
                'scopes': ['tweet.read', 'users.read', 'offline.access']
            },
            'session': {
                'keys': list(session.keys()),
                'twitter_code_verifier_exists': 'twitter_code_verifier' in session,
                'twitter_oauth_state_exists': 'twitter_oauth_state' in session
            },
            'app_config': {
                'session_cookie_secure': app.config.get('SESSION_COOKIE_SECURE'),
                'session_cookie_httponly': app.config.get('SESSION_COOKIE_HTTPONLY'),
                'session_cookie_samesite': app.config.get('SESSION_COOKIE_SAMESITE'),
                'session_type': app.config.get('SESSION_TYPE')
            }
        })
    except Exception as e:
        logger.error(f"Error in debug_twitter_oauth: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Debug endpoint for session configuration
@app.route('/debug/session-config')
def debug_session_config():
    """Debug endpoint to check session configuration"""
    if not app.config.get('DEBUG', False):
        return jsonify({"error": "Debug mode disabled"}), 403
        
    # Get cookie settings
    cookie_settings = {
        "SESSION_COOKIE_DOMAIN": app.config.get('SESSION_COOKIE_DOMAIN'),
        "SESSION_COOKIE_SECURE": app.config.get('SESSION_COOKIE_SECURE'),
        "SESSION_COOKIE_HTTPONLY": app.config.get('SESSION_COOKIE_HTTPONLY'),
        "SESSION_COOKIE_SAMESITE": app.config.get('SESSION_COOKIE_SAMESITE'),
        "SESSION_REFRESH_EACH_REQUEST": app.config.get('SESSION_REFRESH_EACH_REQUEST'),
        "PERMANENT_SESSION_LIFETIME": str(app.config.get('PERMANENT_SESSION_LIFETIME')),
        "SECRET_KEY": "Length: " + str(len(app.config.get('SECRET_KEY', ''))) + " chars"
    }
    
    # Check session interface
    session_interface = str(type(app.session_interface).__name__)
    
    # Store a test value in session and check if it persists
    test_key = 'session_test_' + secrets.token_hex(4)
    test_value = secrets.token_hex(8)
    session[test_key] = test_value
    
    # Force session save
    session.modified = True
    
    return jsonify({
        "app_config": cookie_settings,
        "session_interface": session_interface,
        "current_session_keys": list(session.keys()),
        "environment": {
            "RAILWAY_ENVIRONMENT": os.environ.get('RAILWAY_ENVIRONMENT'),
            "RAILWAY_PUBLIC_DOMAIN": os.environ.get('RAILWAY_PUBLIC_DOMAIN')
        },
        "test_session_key": test_key,
        "test_session_value": test_value
    })

# Top-level routes to handle missing auth routes
@app.route('/logout')
def top_level_logout():
    """Redirect to auth blueprint logout route or handle directly if blueprint not available"""
    logger.info("Top-level logout route accessed")
    # Clear the session directly in case the blueprint route fails
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect('/login')

@app.route('/login/twitter')
def top_level_twitter_login():
    """Redirect to Twitter OAuth flow or handle directly if blueprint not available"""
    logger.info("Top-level Twitter login route accessed")
    try:
        # Initialize Twitter OAuth with configuration
        from auth.oauth_final import TwitterOAuth
        
        # Initialize provider with correct config
        provider_config = {
            'client_id': os.environ.get('TWITTER_CLIENT_ID', ''),
            'client_secret': os.environ.get('TWITTER_CLIENT_SECRET', ''),
            'callback_url': os.environ.get('TWITTER_REDIRECT_URI', '')
        }
        
        # Log config (without sensitive info)
        logger.info(f"Twitter client_id present: {bool(provider_config['client_id'])}")
        logger.info(f"Twitter callback_url: {provider_config['callback_url']}")
        
        # Create OAuth manager and get authorization URL
        twitter_oauth = TwitterOAuth(provider_config)
        auth_data = twitter_oauth.get_authorization_url()
        
        if not auth_data or 'url' not in auth_data:
            logger.error("Failed to generate Twitter authorization URL")
            flash("Failed to connect to Twitter", "error")
            return redirect(url_for('index'))
        
        # Store PKCE data in session
        session['oauth_state'] = auth_data.get('state')
        session['code_verifier'] = auth_data.get('code_verifier')
        session['code_challenge'] = auth_data.get('code_challenge')
        
        # Force session to be saved
        session.modified = True
        
        # Redirect to Twitter auth page
        auth_url = auth_data['url']
        logger.info(f"Redirecting to Twitter authorization URL (truncated): {auth_url[:60]}...")
        
        return redirect(auth_url)
    except Exception as e:
        logger.error(f"Twitter login error: {str(e)}")
        logger.error(traceback.format_exc())
        flash("An error occurred connecting to Twitter. Please try again.", "error")
        return redirect(url_for('index'))

# Top-level login route
@app.route('/login')
def login():
    """Render login page"""
    logger.info("Login page requested")
    return render_template('login_final.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
