"""
Multi-user API server code for final environment.
This extends api_server.py with user authentication and multi-user support.
Incorporates improvements from PythonAnywhere implementation.
"""

import os
import sys
import logging
import json
import time
import secrets
import threading
import traceback
import random
import psycopg2
import sqlalchemy
import shutil  # For file operations
from sqlalchemy import text
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from typing import Dict, Any, Optional, List, Tuple
import sqlite3
import psutil
import string
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory, abort, g, flash, current_app
from flask.sessions import SecureCookieSessionInterface
import uuid
import traceback
import shutil
from werkzeug.utils import secure_filename
import glob
import requests
import hashlib
import platform
from sqlalchemy import text, create_engine
from flask_cors import CORS
from multiprocessing import Process
from threading import Thread
from database.multi_user_db.vector_store_final import get_multi_user_vector_store
from auth.user_context import get_current_user, UserContext
from auth.user_context_final import login_required
from database.multi_user_db.search_final_multi_user import BookmarkSearchMultiUser
from database.multi_user_db.db_final import get_db_connection
from flask_session import Session

# Fix path for Railway deployment - Railway root is twitter_bookmark_manager/deployment/final
# We need to navigate up TWO levels from current file to reach repo root 
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print(f"Added repo root to Python path: {repo_root}")

# Also add the parent of the final directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added parent directory to Python path: {parent_dir}")

print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory, abort, g, flash
from flask.sessions import SecureCookieSessionInterface
import uuid
import traceback
import shutil
from werkzeug.utils import secure_filename
import glob
import requests
import hashlib
import platform
from sqlalchemy import text, create_engine
from flask_cors import CORS

# Import user authentication components
from auth.auth_routes_final import auth_bp
from auth.user_api_final import user_api_bp
from auth.user_context_final import UserContextMiddleware, UserContext, with_user_context
from database.multi_user_db.user_model_final import get_user_by_id

# Import database modules
from database.multi_user_db.db_final import (
    get_db_connection, 
    create_tables, 
    cleanup_db_connections, 
    check_engine_health, 
    close_all_sessions, 
    get_engine,
    db_session,
    check_database_status,
    init_database,
    setup_database,
    get_db_url
)
from database.multi_user_db.search_final_multi_user import BookmarkSearchMultiUser
from database.multi_user_db.update_bookmarks_final import (
    final_update_bookmarks,
    rebuild_vector_store,
    find_file_in_possible_paths,
    get_user_directory,
    run_vector_rebuild
)
from database.multi_user_db.vector_store_final import VectorStore

# Set up base directory using environment variables or relative paths
BASE_DIR = os.environ.get('APP_BASE_DIR', '/app')
DATABASE_DIR = os.environ.get('DATABASE_DIR', os.path.join(BASE_DIR, 'database'))
MEDIA_DIR = os.environ.get('MEDIA_DIR', os.path.join(BASE_DIR, 'media'))
UPLOADS_DIR = os.environ.get('UPLOADS_DIR', os.path.join(BASE_DIR, 'uploads'))

# Ensure key directories exist
for directory in [DATABASE_DIR, MEDIA_DIR, UPLOADS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set up logging with absolute paths
LOG_DIR = os.environ.get('LOG_DIR', os.path.join(BASE_DIR, 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists
LOG_FILE = os.path.join(LOG_DIR, 'api_server_multi_user.log')

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger('api_server_multi_user')
logger.info(f"Starting multi-user API server with PythonAnywhere improvements... Log file: {LOG_FILE}")

# Create Flask app - with robust template path resolution
template_paths = [
    os.path.join(os.getcwd(), 'twitter_bookmark_manager/deployment/final/web_final/templates'),
    os.path.join(BASE_DIR, 'twitter_bookmark_manager/deployment/final/web_final/templates'),
    os.path.join(BASE_DIR, 'web_final/templates'),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web_final/templates')),
]

# Find the first existing template path
template_folder = None
for path in template_paths:
    if os.path.exists(path):
        template_folder = path
        logger.info(f"Found templates at: {path}")
        break

if not template_folder:
    logger.warning(f"Could not find templates in any of: {template_paths}")
    # Use the default path as fallback
    template_folder = os.path.join(os.getcwd(), 'twitter_bookmark_manager/deployment/final/web_final/templates')
    logger.info(f"Using default template path: {template_folder}")

# Create static folder path
static_folder = os.path.join(os.path.dirname(template_folder), 'static')
os.makedirs(static_folder, exist_ok=True)

# Create Flask app with the resolved template path
app = Flask(__name__, 
            template_folder=template_folder,
            static_folder=static_folder)

# Log template information for debugging
logger.info(f"Flask app created with template_folder: {app.template_folder}")
logger.info(f"Flask app created with static_folder: {app.static_folder}")

# Enable error catching
app.config['PROPAGATE_EXCEPTIONS'] = False
app.config['TRAP_HTTP_EXCEPTIONS'] = True
app.config['TRAP_BAD_REQUEST_ERRORS'] = True
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['JSON_AS_ASCII'] = False
app.config['get_db_connection'] = get_db_connection  # Add back the missing database connection function

# Configure CORS
CORS(app)

# Set up session handling
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(BASE_DIR, 'flask_session')
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_NAME'] = 'twitter_bookmark_session'
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

# Set up upload folder configuration
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads'))
app.config['DATABASE_DIR'] = os.environ.get('DATABASE_DIR', os.path.join(BASE_DIR, 'database'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
logger.info(f"Upload folder configured at: {app.config['UPLOAD_FOLDER']}")
logger.info(f"Database directory configured at: {app.config['DATABASE_DIR']}")

# Initialize session
try:
    Session(app)
    logger.info("Flask session initialized with Flask-Session")
except Exception as e:
    logger.warning(f"Could not initialize Flask-Session: {e}")
    logger.info("Continuing with Flask's built-in session management")
    # Continue with Flask's built-in session

# Add a before request handler to log session information
@app.before_request
def log_session_info():
    if request.path.startswith('/static/'):
        return  # Skip for static files
    
    user_id = session.get('user_id')
    logger.info(f"Request: {request.path} - Session: {session.sid if hasattr(session, 'sid') else 'No SID'} - User ID: {user_id}")
    
    # Log when a session is created
    if not hasattr(g, '_session_accessed'):
        g._session_accessed = True
        if not user_id:
            logger.info(f"Session without user_id accessed: {request.path}")
        else:
            logger.info(f"Session with user_id={user_id} accessed: {request.path}")

# Add a before request handler to verify authentication
@app.before_request
def check_user_authentication():
    # Skip check for static files and auth routes
    if (request.path.startswith('/static/') or 
        request.path.startswith('/auth/') or 
        request.path == '/login' or 
        request.path.startswith('/oauth/callback/')):
        return
    
    # Get user from context or session
    user_id = session.get('user_id')
    user = None
    
    if user_id:
        # Try to get user from database
        try:
            conn = get_db_connection()
            from database.multi_user_db.user_model_final import get_user_by_id
            user = get_user_by_id(conn, user_id)
            
            # If user not found but user_id exists in session, clear session
            if not user:
                logger.warning(f"User ID {user_id} from session not found in database")
                session.pop('user_id', None)
                
                # Check if this is an AJAX request
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'authenticated': False,
                        'error': 'User not authenticated. Please log out and log in again.'
                    }), 401
        except Exception as e:
            logger.error(f"Error checking user authentication: {e}")
    
    # Store user in g for this request
    g.user = user
    g.authenticated = user is not None

# Global session status tracking
session_status: Dict[str, Dict[str, Any]] = {}

# Path for the SQLite database to store session status
STATUS_DB_PATH = os.path.join(os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', 'data'), 'session_status.db')

# Initialize status database
def init_status_db():
    """Initialize the SQLite database for storing session status"""
    try:
        conn = sqlite3.connect(STATUS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_status (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            timestamp TEXT NOT NULL,
            is_complete BOOLEAN DEFAULT 0,
            success BOOLEAN DEFAULT 0,
            data TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"Session status database initialized at {STATUS_DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing session status database: {str(e)}")
        logger.error(traceback.format_exc())

# Initialize the database at startup
init_status_db()

# Function to save session status to the database
def save_session_status(session_id, status_data):
    """
    Save session status data to a JSON file.
    
    Args:
        session_id: Unique identifier for the session
        status_data: Dictionary with status data to save
    """
    try:
        # Get user ID from status data or current user
        user_id = status_data.get('user_id')
        if not user_id:
            user = UserContext.get_current_user()
            if user:
                user_id = user.id
                status_data['user_id'] = user_id
            
        if not user_id:
            logger.warning(f"No user ID found for session {session_id}")
            return False
            
        # Create user directory for status files
        from database.multi_user_db.update_bookmarks_final import get_user_directory
        user_dir = get_user_directory(user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Create status file
        status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
            
        logger.debug(f"Saved status for session {session_id}: {status_data.get('status', 'unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving session status: {str(e)}")
        return False

# Function to get session status from the database
def get_session_status(session_id):
    """Get session status from the SQLite database"""
    try:
        conn = sqlite3.connect(STATUS_DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM session_status WHERE session_id = ?
        ''', (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Convert row to dict
            status = dict(row)
            
            # Parse JSON data field
            try:
                status['data'] = json.loads(status['data'])
            except:
                status['data'] = {}
                
            # Convert SQLite booleans (0/1) to Python booleans
            status['is_complete'] = bool(status['is_complete'])
            status['success'] = bool(status['success'])
            
            return status
        else:
            return None
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Cleanup old sessions periodically
def cleanup_old_sessions():
    """Remove sessions older than 24 hours from the database"""
    try:
        conn = sqlite3.connect(STATUS_DB_PATH)
        cursor = conn.cursor()
        
        # Get timestamp for 24 hours ago
        cutoff_time = (datetime.now() - timedelta(hours=24)).isoformat()
        
        cursor.execute('''
        DELETE FROM session_status WHERE created_at < ?
        ''', (cutoff_time,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired session records")
        
        return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning up old sessions: {str(e)}")
        logger.error(traceback.format_exc())
        return 0

# Register cleanup function to run on app shutdown
@app.teardown_appcontext
def shutdown_cleanup(exception=None):
    """Clean up resources when the app shuts down"""
    logger.info("Application context tearing down, cleaning up resources")
    cleanup_db_connections()

# Check database health before each request
@app.before_request
def check_db_health():
    """Check database connection health before handling request"""
    # Skip for static files and non-db routes
    if request.path.startswith('/static/') or request.path.startswith('/favicon.ico'):
        return
        
    try:
        # Only check health on percentage of requests to avoid overhead
        if random.random() < 0.05:  # 5% of requests (reduced from 10%)
            try:
                # Use a simple direct database check instead of the complex engine health check
                conn = get_db_connection()
                if conn:
                    # We got a connection, so the database is up
                    if hasattr(conn, 'close'):
                        conn.close()  # Close the connection properly
                    return
                else:
                    # Force reconnect if we couldn't get a connection
                    logger.warning("Database connection check failed, forcing reconnect")
                    setup_database(force_reconnect=True)
                    logger.info("Forced database reconnection after connection check failure")
                    
            except Exception as e:
                # Log the error but don't fail the request
                logger.error(f"Database health check error: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in health check routine: {e}")
        # Continue processing the request even if health check fails

@app.before_request
def make_session_permanent():
    """Make session permanent and refresh it with each request"""
    # Skip for static files
    if request.path.startswith('/static/'):
        return
        
    # Make session permanent
    session.permanent = True
    
    # Ensure session modified is set for any user with user_id
    # This helps ensure session data persists across requests
    if 'user_id' in session:
        session.modified = True
        # Add a timestamp to help track session activity
        session['last_activity'] = time.time()
        # Log minimal session info for debugging
        logger.debug(f"Session refreshed - Path: {request.path} - User ID: {session.get('user_id')}")

# Register authentication blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(user_api_bp)

# Initialize user context middleware
UserContextMiddleware(app, lambda conn, user_id: get_user_by_id(conn, user_id))

# Debug database initialization function (to be called at startup)
def init_app_debug():
    """Check database at startup and ensure guest user exists"""
    try:
        logger.info("STARTUP DEBUG - Checking database initialization")
        conn = get_db_connection()
        
        # Check table counts
        tables = ['users', 'bookmarks', 'categories', 'bookmark_categories']
        for table in tables:
            cursor = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = cursor.scalar()
            logger.info(f"STARTUP DEBUG - Table {table} has {count} records")
            
        # Ensure guest user exists (user_id = 1)
        cursor = conn.execute(text("SELECT COUNT(*) FROM users WHERE id = 1"))
        if cursor.scalar() == 0:
            logger.info("STARTUP DEBUG - Creating default guest user with ID 1")
            cursor = conn.execute(
                text("INSERT INTO users (id, username, email) VALUES (1, 'guest', 'guest@example.com') ON CONFLICT DO NOTHING")
            )
            
        # Check if we have any bookmarks for user 1
        cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id = 1"))
        count = cursor.scalar()
        logger.info(f"STARTUP DEBUG - User 1 has {count} bookmarks")
        
        # If no bookmarks for user 1, check if there are "orphaned" bookmarks with NULL user_id
        if count == 0:
            cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id IS NULL"))
            null_count = cursor.scalar()
            logger.info(f"STARTUP DEBUG - Found {null_count} bookmarks with NULL user_id")
            
            if null_count > 0:
                # Update these bookmarks to belong to user 1
                cursor = conn.execute(text("UPDATE bookmarks SET user_id = 1 WHERE user_id IS NULL"))
                logger.info(f"STARTUP DEBUG - Updated {null_count} bookmarks to user_id 1")
                
        # Get total bookmarks for user 1
        cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id = :user_id"), {"user_id": 1})
        total_user_1 = cursor.scalar()
        logger.info(f"Total bookmarks for user 1: {total_user_1}")
        
        # Get total bookmarks with no user
        cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id IS NULL"))
        total_no_user = cursor.scalar()
        logger.info(f"Total bookmarks with no user: {total_no_user}")
        
        conn.close()
    except Exception as e:
        logger.error(f"STARTUP DEBUG - Error: {e}")
        logger.error(traceback.format_exc())

# Call the init function right away
init_app_debug()

# Home page
@app.route('/')
def index():
    """Home page with multiple database connection fallbacks"""
    logger.info("Home page requested")
    user = UserContext.get_current_user()
    
    # DEBUG: Log user information
    if user:
        logger.info(f"USER DEBUG - Authenticated user: ID={user.id}, Name={getattr(user, 'username', 'unknown')}")
    else:
        logger.info("USER DEBUG - No authenticated user")
    
    # Choose template based on authentication
    if user:
        template = 'index_final.html'
    else:
        # Show login page for unauthenticated users
        logger.info("User not authenticated, redirecting to login")
        return redirect(url_for('auth.login'))
    
    # Track if we've tried all database connection methods
    all_methods_tried = False
    categories = []
    latest_tweets = []  # Initialize latest_tweets
    error_message = None
    
    # Method 1: Direct psycopg2 connection
    try:
        logger.info("Trying direct psycopg2 connection")
        db_url = get_db_url()
        
        # Parse the connection string to get connection parameters
        if 'postgresql://' in db_url:
            # Extract connection params from sqlalchemy URL
            conn_parts = db_url.replace('postgresql://', '').split('@')
            user_pass = conn_parts[0].split(':')
            host_port_db = conn_parts[1].split('/')
            host_port = host_port_db[0].split(':')
            
            db_user = user_pass[0]
            db_password = user_pass[1]
            db_host = host_port[0]
            db_port = host_port[1] if len(host_port) > 1 else '5432'
            db_name = host_port_db[1]
            
            logger.info(f"Connecting directly to PostgreSQL at {db_host}:{db_port}/{db_name}")
            
            # Connect directly with psycopg2
            direct_conn = psycopg2.connect(
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
                dbname=db_name,
                connect_timeout=3,
                application_name='twitter_bookmark_manager_direct'
            )
            
            # Set autocommit to avoid transaction issues
            direct_conn.autocommit = True
            
            # Execute a simple query to get categories
            cursor = direct_conn.cursor()
            try:
                # First try with description column
                cursor.execute(f"""
                    SELECT id, name, description 
                    FROM categories 
                    WHERE user_id = %s 
                    ORDER BY name
                """, (user.id,))
            except Exception as e:
                logger.warning(f"Error querying categories with description: {e}")
                # Fallback query without description
                cursor.execute(f"""
                    SELECT id, name, '' as description
                    FROM categories 
                    WHERE user_id = %s 
                    ORDER BY name
                """, (user.id,))
            
            # Fetch categories directly
            categories = []
            for row in cursor.fetchall():
                categories.append({
                    'id': row[0],
                    'name': row[1],
                    'description': row[2]
                })
            
            # Get recent bookmarks
            try:
                cursor.execute("""
                    SELECT 
                        bookmark_id,
                        text,
                        author_name,
                        author_username,
                        created_at
                    FROM bookmarks 
                    WHERE user_id = %s
                    ORDER BY created_at DESC 
                    LIMIT 5
                """, (user.id,))
                
                for row in cursor.fetchall():
                    tweet = {
                        'id': row[0],  # Using bookmark_id as id
                        'text': row[1],
                        'author': row[2],
                        'author_username': row[3],
                        'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[4], 'strftime') else row[4],
                        'categories': []  # Initialize empty categories
                    }
                    latest_tweets.append(tweet)
                
                logger.info(f"Successfully retrieved {len(latest_tweets)} latest bookmarks")
            except Exception as e:
                logger.warning(f"Error getting recent bookmarks: {e}")
            
            cursor.close()
            direct_conn.close()
            
            logger.info(f"Successfully loaded {len(categories)} categories directly for user {user.id}")
        else:
            # Non-PostgreSQL DB, skip to method 2
            raise Exception("Not a PostgreSQL database, trying SQLAlchemy")
    except Exception as e:
        logger.warning(f"Direct psycopg2 connection failed: {e}")
        error_message = str(e)
    
    # Method 2: SQLAlchemy connection if Method 1 failed
    if not categories:
        try:
            logger.info("Trying SQLAlchemy connection")
            # Force reconnect the engine first
            from database.multi_user_db.db_final import setup_database
            setup_database(force_reconnect=True)
            
            # Get a new connection
            conn = get_db_connection()
            try:
                searcher = BookmarkSearchMultiUser(conn, user.id if user else 1)
                latest_tweets = searcher.get_recent_bookmarks(limit=5)
                logger.info(f"Successfully retrieved {len(latest_tweets)} latest bookmarks")
                
                # FALLBACK: If no bookmarks found for this user, try getting system bookmarks
                if len(latest_tweets) == 0 and user and user.id != 1:
                    logger.warning(f"No bookmarks found for user {user.id}, falling back to system bookmarks")
                    
                    # Try with user_id = 1 (system user)
                    searcher = BookmarkSearchMultiUser(conn, 1)
                    latest_tweets = searcher.get_recent_bookmarks(limit=5)
                    logger.info(f"Retrieved {len(latest_tweets)} system bookmarks as fallback")
                    
                    # If still no results, try with a direct query for ANY bookmarks
                    if len(latest_tweets) == 0:
                        logger.warning("No system bookmarks found, trying direct query for any bookmarks")
                        
                        # Direct query for any bookmarks
                        query = """
                        SELECT id, bookmark_id, text, author, created_at, author_id
                        FROM bookmarks
                        ORDER BY created_at DESC
                        LIMIT 5
                        """
                        
                        if isinstance(conn, sqlalchemy.engine.Connection):
                            result = conn.execute(text(query))
                            rows = result.fetchall()
                        else:
                            cursor = conn.cursor()
                            cursor.execute(query)
                            rows = cursor.fetchall()
                            
                        # Format the direct query results
                        for row in rows:
                            bookmark = {
                                'id': row[0],
                                'bookmark_id': row[1],
                                'text': row[2],
                                'author': row[3],
                                'author_username': row[3].replace('@', '') if row[3] else '',
                                'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[4], 'strftime') else row[4],
                                'author_id': row[5]
                            }
                            latest_tweets.append(bookmark)
                            
                        logger.info(f"Direct query found {len(latest_tweets)} bookmarks")
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"SQLAlchemy connection failed: {e}")
            if not error_message:
                error_message = str(e)
    
    # Method 3: Direct SQL connection with different parameters if Method 2 failed
    if not categories:
        try:
            logger.info("Trying alternative direct connection")
            db_url = get_db_url()
            
            if 'postgresql://' in db_url:
                # Extract connection params from sqlalchemy URL (same as Method 1)
                conn_parts = db_url.replace('postgresql://', '').split('@')
                user_pass = conn_parts[0].split(':')
                host_port_db = conn_parts[1].split('/')
                host_port = host_port_db[0].split(':')
                
                db_user = user_pass[0]
                db_password = user_pass[1]
                db_host = host_port[0]
                db_port = host_port[1] if len(host_port) > 1 else '5432'
                db_name = host_port_db[1]
                
                logger.info(f"Trying alternative connection to PostgreSQL")
                
                # Try different connection parameters
                direct_conn = psycopg2.connect(
                    user=db_user,
                    password=db_password,
                    host=db_host,
                    port=db_port,
                    dbname=db_name,
                    connect_timeout=5,  # Longer timeout
                    application_name='twitter_bookmark_manager_last_resort',
                    keepalives=1,
                    keepalives_idle=10,
                    keepalives_interval=2,
                    keepalives_count=3
                )
                
                # Important: Set isolation level to avoid transaction issues
                direct_conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                
                # Execute a simple query to get categories
                cursor = direct_conn.cursor()
                try:
                    # First try with description column
                    cursor.execute(f"""
                        SELECT id, name, description 
                        FROM categories 
                        WHERE user_id = %s 
                        ORDER BY name
                    """, (user.id,))
                except Exception as e:
                    logger.warning(f"Error querying categories with description: {e}")
                    # Fallback query without description
                    cursor.execute(f"""
                        SELECT id, name, '' as description
                        FROM categories 
                        WHERE user_id = %s 
                        ORDER BY name
                    """, (user.id,))
                
                # Fetch categories directly
                categories = []
                for row in cursor.fetchall():
                    categories.append({
                        'id': row[0],
                        'name': row[1],
                        'description': row[2]
                    })
                
                cursor.close()
                direct_conn.close()
                
                logger.info(f"Successfully loaded {len(categories)} categories via alternative connection")
            else:
                # Skip for non-PostgreSQL
                all_methods_tried = True
        except Exception as e:
            logger.warning(f"Alternative direct connection failed: {e}")
            if not error_message:
                error_message = str(e)
            all_methods_tried = True
    
    # At this point, we've tried all database methods
    if not categories:
        all_methods_tried = True
    
    # Check if user is admin
    is_admin = getattr(user, 'is_admin', False)
    
    # If all database methods failed, show a simplified interface with error message
    if all_methods_tried and error_message and not categories:
        logger.error(f"All database connection methods failed. Last error: {error_message}")
        
        # Return a simplified interface
        return render_template(
            template, 
            categories=[],  # Empty categories
            user=user, 
            is_admin=is_admin,
            db_error=True,
            error_message="Database connection issues. Some features may be unavailable."
        )
    
    # Return normal template if we have successfully connected, even if no categories
    return render_template(
        template, 
        categories=categories or [], 
        user=user, 
        is_admin=is_admin,
        db_error=False,
        latest_tweets=latest_tweets,
        showing_results=len(latest_tweets),
        total_results=len(latest_tweets)
    )

def safe_vector_operation(func):
    """
    Decorator to safely handle vector store operations.
    Catches vector store access errors and returns proper JSON responses.
    """
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
                    'retry_after': 300  # 5 minutes
                }), 503
            # Re-raise other runtime errors
            raise
    return wrapper

@app.route('/upload-bookmarks', methods=['POST'])
@login_required
@safe_vector_operation
def upload_bookmarks():
    """
    Handle file upload for bookmarks JSON file
    """
    try:
        # Get user info using UserContext
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user found for upload - UserContext check failed")
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
            
        user_id = user.id
        logger.info(f"Processing upload for user {user_id}")
        
        # Check if file is in the request
        if 'file' not in request.files:
            logger.error(f"No file part in the request for user {user_id}")
            return jsonify({'success': False, 'error': 'No file part in the request'}), 400
            
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            logger.error(f"No file selected for user {user_id}")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Check if the file has a .json extension
        if not file.filename.lower().endswith('.json'):
            logger.error(f"Invalid file type: {file.filename} - must be a .json file")
            return jsonify({'success': False, 'error': 'Only JSON files are allowed'}), 400
            
        logger.info(f"File received: {file.filename} for user {user_id}")
        
        # Ensure upload directory exists - use Railway volume mount path if available
        base_upload_dir = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/app/uploads')
        upload_folder = os.path.join(base_upload_dir, 'uploads')
        user_dir = os.path.join(upload_folder, f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"User directory confirmed: {user_dir}")
        
        # Generate a safe filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(f"bookmarks_{timestamp}.json")
        filepath = os.path.join(user_dir, filename)
        
        # Save the file
        try:
            file.save(filepath)
            file_size = os.path.getsize(filepath)
            logger.info(f"File saved: {filepath} - Size: {file_size} bytes")
            
            # Validate JSON format
            is_valid, error_message, data = validate_json_file(filepath)
            if not is_valid:
                logger.error(f"Invalid JSON format: {error_message}")
                return jsonify({
                    'success': False, 
                    'error': error_message,
                    'file_saved': True,
                    'filepath': filepath
                }), 400
                
            logger.info(f"File validated successfully: {filepath}")
            
            # Return success with session ID for later processing
            session_id = str(uuid.uuid4())
            
            # Count bookmarks for informational purposes
            bookmark_count = 0
            if isinstance(data, list):
                bookmark_count = len(data)
            elif isinstance(data, dict) and 'bookmarks' in data:
                bookmark_count = len(data.get('bookmarks', []))
                
            logger.info(f"Successfully uploaded and validated file with {bookmark_count} bookmarks")
            
            return jsonify({
                'success': True, 
                'message': 'File uploaded successfully',
                'file': filename,
                'session_id': session_id,
                'bookmark_count': bookmark_count
            })
            
        except Exception as save_error:
            logger.error(f"Error processing uploaded file: {str(save_error)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False, 
                'error': f'Error processing file: {str(save_error)}',
                'file_saved': os.path.exists(filepath)
            }), 500
            
    except Exception as e:
        logger.error(f"Error in upload_bookmarks: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

def validate_json_file(filepath):
    """
    Validate that a file contains valid JSON and has a structure suitable for bookmarks.
    
    Args:
        filepath (str): Path to the file to validate
        
    Returns:
        tuple: (is_valid, error_message, data)
    """
    try:
        logger.info(f"Validating JSON file: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False, "File not found", None
            
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            logger.error(f"File is empty: {filepath}")
            return False, "File is empty", None
            
        # Check if file is too large (over 30MB)
        if file_size > 30 * 1024 * 1024:
            logger.error(f"File is too large: {file_size} bytes")
            return False, f"File is too large: {file_size} bytes (max 30MB)", None
            
        # Open and read file - handle encoding issues gracefully
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            # Check if file content is empty
            if not file_content.strip():
                logger.error(f"File content is empty: {filepath}")
                return False, "File content is empty", None
                
            # Try to parse JSON
            try:
                data = json.loads(file_content)
                logger.info(f"Successfully parsed JSON, data type: {type(data).__name__}")
            except json.JSONDecodeError as je:
                line_col = f" at line {je.lineno}, column {je.colno}"
                error_msg = f"Invalid JSON format: {je.msg}{line_col}"
                logger.error(error_msg)
                
                # Try to provide more context about the error
                if je.lineno > 1:
                    lines = file_content.split('\n')
                    if je.lineno <= len(lines):
                        context_line = lines[je.lineno - 1]
                        error_msg += f"\nProblematic line: {context_line[:100]}"
                        
                return False, error_msg, None
        except UnicodeDecodeError as ude:
            logger.error(f"Unicode decode error: {str(ude)}")
            return False, f"File encoding error: {str(ude)} - Please ensure the file is UTF-8 encoded", None
        except IOError as ioe:
            logger.error(f"IO error reading file: {str(ioe)}")
            return False, f"Error reading file: {str(ioe)}", None
        
        # Validate structure based on data type
        if isinstance(data, list):
            if len(data) == 0:
                logger.error("JSON array is empty")
                return False, "JSON array is empty", None
                
            # Log first item for debugging
            if len(data) > 0:
                logger.info(f"First item in array has keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'not a dict'}")
                
            # It's a direct array of bookmarks
            return True, "", data
            
        elif isinstance(data, dict):
            # Log keys for debugging
            logger.info(f"JSON object has top-level keys: {list(data.keys())}")
            
            # Check for common bookmark fields in dictionary
            if 'bookmarks' in data and isinstance(data['bookmarks'], list):
                if len(data['bookmarks']) == 0:
                    logger.error("No bookmarks found in the 'bookmarks' field")
                    return False, "No bookmarks found in the 'bookmarks' field", None
                    
                logger.info(f"Found {len(data['bookmarks'])} bookmarks in 'bookmarks' field")
                return True, "", data
                
            # Other possible structures
            if 'tweet' in data and isinstance(data['tweet'], list):
                if len(data['tweet']) == 0:
                    logger.error("No tweets found in the 'tweet' field")
                    return False, "No tweets found in the 'tweet' field", None
                    
                logger.info(f"Found {len(data['tweet'])} tweets in 'tweet' field")
                return True, "", data
                
            # Check for common bookmark fields directly in the object
            bookmark_fields = ['id_str', 'id', 'tweet_id', 'tweet_url', 'text', 'full_text']
            has_bookmark_fields = any(key in data for key in bookmark_fields)
            
            if has_bookmark_fields:
                # Log which fields were found
                found_fields = [key for key in bookmark_fields if key in data]
                logger.info(f"Found bookmark fields directly in object: {found_fields}")
                
                # Wrap single bookmark in an array
                return True, "", [data]
            
            # If we get here, structure is not recognized
            logger.error(f"Unrecognized JSON structure. Found keys: {list(data.keys())}")
            return False, "Could not identify bookmark data in JSON. Expected 'bookmarks' array or bookmark fields.", None
        else:
            # Not an object or array
            logger.error(f"JSON is not an object or array, found type: {type(data).__name__}")
            return False, f"JSON must be an object or array, found: {type(data).__name__}", None
            
    except Exception as e:
        logger.error(f"Unexpected error validating JSON: {str(e)}")
        logger.error(traceback.format_exc())
        return False, f"Error validating JSON: {str(e)}", None

@app.route('/process-bookmarks', methods=['POST'])
@login_required
@safe_vector_operation
def process_bookmarks():
    """Process bookmarks from a JSON file."""
    try:
        user = UserContext.get_current_user()
        if not user:
            logger.error("User not authenticated")
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
        
        user_id = user.id
        logger.info(f"Processing bookmarks for user {user_id}")
        
        # Track if we have a valid file
        file_to_process = None
        
        # Check if we have a file or a direct JSON payload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                logger.error("No file selected in the request")
                return jsonify({'success': False, 'error': 'No file selected'}), 400
                
            # Check if file is a JSON file
            if not file.filename.lower().endswith('.json'):
                logger.error(f"Invalid file type: {file.filename}")
                return jsonify({'success': False, 'error': 'File must be a .json file'}), 400
                
            # Ensure UPLOAD_FOLDER is configured
            if 'UPLOAD_FOLDER' not in app.config:
                logger.warning("UPLOAD_FOLDER not configured, setting default")
                app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
            # Set uploads directory
            uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
            os.makedirs(uploads_dir, exist_ok=True)
            logger.info(f"Upload directory created/verified: {uploads_dir}")
            
            # Generate filename and save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bookmarks_{timestamp}.json"
            filepath = os.path.join(uploads_dir, filename)
            
            try:
                file.save(filepath)
                file_size = os.path.getsize(filepath)
                logger.info(f"File saved: {filepath} - Size: {file_size} bytes")
                
                # Validate it's a valid JSON file using our comprehensive validator
                is_valid, error_message, data = validate_json_file(filepath)
                if not is_valid:
                    logger.error(f"Invalid JSON format: {error_message}")
                    return jsonify({
                        'success': False, 
                        'error': error_message,
                        'file_saved': True,
                        'filepath': filepath
                    }), 400
                    
                # Log validation success
                logger.info(f"File validation successful: {filepath}")
                
                # File is valid JSON, set for processing
                file_to_process = filepath
            except Exception as e:
                logger.error(f"Error saving or validating file: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'}), 500
                
        elif request.is_json:
            # Handle direct JSON payload
            try:
                data = request.get_json()
                if not data:
                    logger.error("Empty JSON data provided")
                    return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
                    
                logger.info(f"Received direct JSON payload of type: {type(data).__name__}")
                
                # Save as file for consistent processing
                uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
                os.makedirs(uploads_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"bookmarks_{timestamp}.json"
                filepath = os.path.join(uploads_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"JSON data saved to file: {filepath}")
                
                # Validate the saved file to ensure it has the right structure
                is_valid, error_message, _ = validate_json_file(filepath)
                if not is_valid:
                    logger.error(f"Invalid JSON structure: {error_message}")
                    return jsonify({
                        'success': False, 
                        'error': error_message,
                        'file_saved': True,
                        'filepath': filepath
                    }), 400
                
                file_to_process = filepath
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {str(je)}")
                return jsonify({'success': False, 'error': f'Invalid JSON format: {str(je)}'}), 400
            except Exception as e:
                logger.error(f"Error processing JSON data: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'success': False, 'error': f'Error processing JSON data: {str(e)}'}), 500
        else:
            logger.error("No file or JSON data provided in request")
            return jsonify({'success': False, 'error': 'No file or JSON data provided'}), 400
            
        # If we don't have a file to process, something went wrong
        if not file_to_process:
            logger.error("Failed to prepare file for processing - no file_to_process set")
            return jsonify({'success': False, 'error': 'Failed to prepare file for processing'}), 500
            
        # Ensure DATABASE_DIR is configured
        if 'DATABASE_DIR' not in app.config:
            app.config['DATABASE_DIR'] = os.path.join(BASE_DIR, 'database')
            os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
            logger.info(f"Created missing DATABASE_DIR at {app.config['DATABASE_DIR']}")
            
        # Copy to database directory
        db_dir = os.path.join(app.config['DATABASE_DIR'], f'user_{user_id}')
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Database directory created/verified: {db_dir}")
        
        target_file = os.path.join(db_dir, 'twitter_bookmarks.json')
        try:
            shutil.copy2(file_to_process, target_file)
            logger.info(f"Copied file to database directory: {target_file}")
        except Exception as e:
            logger.error(f"Error copying file to database directory: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False, 
                'error': f'Error copying file: {str(e)}',
                'file_saved': True,
                'filepath': file_to_process
            }), 500
            
        # Process the bookmarks from the JSON file
        try:
            # Use the final update bookmarks function from Railway
            from database.multi_user_db.update_bookmarks_final import final_update_bookmarks
            
            # Generate a session ID for tracking
            session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            logger.info(f"Generated session ID for processing: {session_id}")
            
            # Start processing in the background
            def process_task():
                with app.app_context():
                    try:
                        logger.info(f"Starting background processing for session {session_id}")
                        
                        # Add specific handling for vector store access errors
                        try:
                            result = final_update_bookmarks(
                                session_id=session_id,
                                start_index=0,
                                rebuild_vector=True,  # Always rebuild vectors when processing
                                user_id=user_id
                            )
                            logger.info(f"Background processing completed for session {session_id}: {result}")
                        except RuntimeError as re:
                            if "Storage folder" in str(re) and "already accessed by another instance" in str(re):
                                logger.error(f"Vector store access conflict: {str(re)}")
                                
                                # Save error status with clear message for client
                                error_status = {
                                    'session_id': session_id,
                                    'status': 'error',
                                    'message': "Vector database is busy. Please try again in a few minutes.",
                                    'timestamp': datetime.now().isoformat(),
                                    'user_id': user_id,
                                    'technical_details': str(re)
                                }
                                save_session_status(session_id, error_status)
                            else:
                                # Re-raise if it's not the vector store access error
                                raise
                    except Exception as e:
                        logger.error(f"Error in background processing for session {session_id}: {str(e)}")
                        logger.error(traceback.format_exc())
                        
                        # Save error status for client to retrieve
                        error_status = {
                            'session_id': session_id,
                            'status': 'error',
                            'message': f"Processing error: {str(e)}",
                            'timestamp': datetime.now().isoformat(),
                            'user_id': user_id
                        }
                        save_session_status(session_id, error_status)
            
            # Start background thread for processing
            thread = Thread(target=process_task)
            thread.daemon = True
            thread.start()
            
            # Return success with session ID for tracking
            return jsonify({
                'success': True,
                'message': 'Processing started in background',
                'session_id': session_id,
                'file_processed': os.path.basename(file_to_process)
            })
            
        except Exception as e:
            logger.error(f"Error initiating bookmark processing: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False, 
                'error': f'Error initiating processing: {str(e)}',
                'file_saved': True,
                'filepath': file_to_process
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in process_bookmarks: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Provide more specific error message based on what happened
        error_message = str(e)
        if "UPLOAD_FOLDER" in error_message:
            # If this specific error occurs, give a more helpful error
            if 'UPLOAD_FOLDER' not in app.config:
                app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                logger.info(f"Created missing upload folder at {app.config['UPLOAD_FOLDER']}")
            error_message = "Server configuration issue with upload folder. It has been fixed, please try again."
        elif "DATABASE_DIR" in error_message:
            # If this specific error occurs, give a more helpful error
            if 'DATABASE_DIR' not in app.config:
                app.config['DATABASE_DIR'] = os.path.join(BASE_DIR, 'database')
                os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
                logger.info(f"Created missing database directory at {app.config['DATABASE_DIR']}")
            error_message = "Server configuration issue with database directory. It has been fixed, please try again."
        
        return jsonify({
            'success': False, 
            'error': f"Unexpected error: {error_message}",
            'retry_recommended': True
        }), 500

@app.route('/update-status', methods=['GET'])
def update_status_redirect():
    """Redirect old /update-status endpoint to /api/process-status for backwards compatibility"""
    session_id = request.args.get('session_id')
    logger.info(f"🔄 Redirecting /update-status to /api/process-status for session {session_id}")
    return redirect(url_for('process_status', session_id=session_id))

@app.route('/api/process-status', methods=['GET'])
def process_status():
    """Check status of background processing"""
    user = UserContext.get_current_user()
    user_id = user.id if user else None
    
    # Get session ID from request
    session_id = request.args.get('session_id')
    if not session_id:
        logger.error(f"❌ [STATUS] No session ID provided")
        return jsonify({'error': 'No session ID provided'}), 400
        
    logger.info(f"🔍 [STATUS-{session_id}] Checking processing status for user {user_id}")
    
    # Find status file
    user_dir = get_user_directory(user_id)
    status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
    
    if not os.path.exists(status_file):
        logger.error(f"❌ [STATUS-{session_id}] Status file not found: {status_file}")
        return jsonify({'error': 'Session not found'}), 404
        
    # Read status file
    try:
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            
        # Return status
        return jsonify({
            'success': True,
            'session_id': session_id,
            'status': status_data.get('status', 'unknown'),
            'details': status_data
        })
        
    except Exception as e:
        logger.error(f"❌ [STATUS-{session_id}] Error: {str(e)}")
        return jsonify({'error': 'Error reading status', 'details': str(e)}), 500

@app.route('/api/update-database', methods=['POST'])
@login_required
def update_database():
    """API endpoint to update the database from a JSON file
    This function is improved based on the PythonAnywhere approach
    with complete separation of vector operations from database updates
    """
    try:
        # Get current user
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user context found for request")
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
            
        user_id = user.id
        logger.info(f"Processing database update for user {user_id}")
        
        # Get request JSON data
        data = request.get_json() or {}
        session_id = data.get('session_id', str(uuid.uuid4())[:8])
        start_index = data.get('start_index', 0)
        skip_vector = data.get('skip_vector', True)  # Default to skipping vector operations
        
        # Check if file exists
        user_dir = os.path.join(app.config.get('DATABASE_DIR', '/app/database'), f'user_{user_id}')
        json_file = os.path.join(user_dir, 'twitter_bookmarks.json')
        
        if not os.path.exists(json_file):
            logger.error(f"JSON file not found at {json_file}")
            return jsonify({
                'success': False,
                'error': f'Bookmarks file not found. Please upload a file first.',
                'file_path': json_file
            }), 404
            
        # Log file size for debugging
        file_size = os.path.getsize(json_file)
        logger.info(f"Found JSON file at {json_file} (size: {file_size} bytes)")
        
        # Check if the file is valid JSON (just the basic check)
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not (first_line.startswith('{') or first_line.startswith('[')):
                    return jsonify({
                        'success': False,
                        'error': 'File does not appear to be valid JSON',
                        'first_line': first_line
                    }), 400
        except Exception as json_error:
            logger.error(f"Error checking JSON file: {json_error}")
            return jsonify({
                'success': False,
                'error': f'Error checking JSON file: {str(json_error)}'
            }), 500
            
        # Start the update process in the background
        logger.info(f"Starting background database update process with session_id {session_id}")
        
        # Generate a status file
        status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
        with open(status_file, 'w') as f:
            status = {
                'user_id': user_id,
                'session_id': session_id,
                'status': 'processing',
                'message': 'Database update started',
                'start_time': datetime.now().isoformat(),
                'progress': 0
            }
            json.dump(status, f)
            
        # Launch update process in background thread
        def update_process():
            from datetime import datetime
            import traceback
            
            start_time = datetime.now()
            
            try:
                # Run the database update WITHOUT vector operations
                from database.multi_user_db.update_bookmarks_final import final_update_bookmarks
                
                with app.app_context():
                    try:
                        # Log the start of the update
                        logger.info(f"Starting database update in background - session_id={session_id}")
                        
                        # When rebuild is explicitly requested
                        if rebuild_vector and direct_path:
                            logger.info(f"Starting database update with vector rebuild for session {session_id}")
                            import_result = final_update_bookmarks(
                                session_id=session_id, 
                                start_index=int(start_index),
                                rebuild_vector=True,
                                user_id=user_id,
                                skip_vector=True  # Skip vector operations to avoid errors
                            )
                        # Normal case - don't rebuild
                        else:
                            logger.info(f"Starting database update without vector rebuild for session {session_id}")
                            import_result = final_update_bookmarks(
                                session_id=session_id, 
                                start_index=int(start_index),
                                rebuild_vector=False,
                                user_id=user_id,
                                skip_vector=True  # Skip vector operations to avoid errors
                            )
                        
                        # Update status file with result
                        with open(status_file, 'w') as f:
                            status = {
                                'user_id': user_id,
                                'session_id': session_id,
                                'status': 'completed' if import_result.get('success', False) else 'error',
                                'message': 'Database update completed' if import_result.get('success', False) else 'Database update failed',
                                'result': import_result,
                                'start_time': start_time.isoformat(),
                                'end_time': datetime.now().isoformat(),
                                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                                'progress': 100 if import_result.get('success', False) else 0,
                                'vector_update_needed': not skip_vector  # Indicate vector update is needed as a second step
                            }
                            json.dump(status, f)
                            
                        logger.info(f"Database update completed - session_id={session_id}")
                        
                        # If vector operations are requested and database update was successful,
                        # trigger a separate vector update process
                        if not skip_vector and import_result.get('success', False):
                            try:
                                logger.info(f"Starting vector store update process - session_id={session_id}")
                                # This should be a separate endpoint call in a real implementation
                                # For now, we'll just log that it would happen
                                logger.info(f"Vector store update would be started here - session_id={session_id}")
                            except Exception as vector_error:
                                logger.error(f"Error in vector store update: {vector_error}")
                                logger.error(traceback.format_exc())
                                # Don't fail the whole process if vector update fails
                                
                    except Exception as e:
                        # Log error and update status file
                        logger.error(f"Error in database update process: {e}")
                        logger.error(traceback.format_exc())
                        
                        with open(status_file, 'w') as f:
                            status = {
                                'user_id': user_id,
                                'session_id': session_id,
                                'status': 'error',
                                'message': f'Error updating database: {str(e)}',
                                'error': str(e),
                                'traceback': traceback.format_exc(),
                                'start_time': start_time.isoformat(),
                                'end_time': datetime.now().isoformat(),
                                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                                'progress': 0
                            }
                            json.dump(status, f)
            except Exception as e:
                # Handle outer exceptions
                logger.error(f"Fatal error in update thread: {e}")
                logger.error(traceback.format_exc())
                
                try:
                    with open(status_file, 'w') as f:
                        status = {
                            'user_id': user_id,
                            'session_id': session_id,
                            'status': 'error',
                            'message': f'Fatal error in update process: {str(e)}',
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'start_time': start_time.isoformat(),
                            'end_time': datetime.now().isoformat(),
                            'duration_seconds': (datetime.now() - start_time).total_seconds(),
                            'progress': 0
                        }
                        json.dump(status, f)
                except Exception:
                    pass
        
        # Start the update process in a background thread
        update_thread = Thread(target=update_process)
        update_thread.daemon = True
        update_thread.start()
        
        # Return immediate success response with session ID for tracking
        return jsonify({
            'success': True,
            'message': 'Database update started in background',
            'session_id': session_id,
            'file': os.path.basename(json_file),
            'status_file': os.path.basename(status_file),
            'next_step': {
                'endpoint': '/api/process-status',
                'method': 'GET',
                'params': {'session_id': session_id}
            },
            'vector_rebuild': {
                'needed': not skip_vector,
                'endpoint': '/api/rebuild-vector-store',
                'method': 'POST',
                'params': {'session_id': session_id}
            }
        })
        
    except Exception as e:
        logger.error(f"Error in update_database endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/categories/all', methods=['GET'])
@login_required
def get_all_categories():
    """API endpoint to get all categories for the current user, including those with zero bookmarks"""
    try:
        # Get current user's ID from UserContext
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user context found for request")
            return jsonify({
                'status': 'error',
                'message': 'User not authenticated'
            }), 401

        # Get database connection
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return jsonify({
                'status': 'error',
                'message': 'Database connection error'
            }), 500
        
        try:
            # Create search instance with user context
            searcher = BookmarkSearchMultiUser(conn, user.id)
            
            # Get categories with counts
            categories = searcher.get_categories(user_id=user.id)
            
            # Sort alphabetically by name
            categories.sort(key=lambda x: x['name'])
            
            return jsonify({
                'status': 'success',
                'categories': categories
            })
        
        finally:
            # Ensure connection is properly handled
            if hasattr(conn, 'close') and not isinstance(conn, sqlalchemy.engine.base.Engine):
                conn.close()
    
    except Exception as e:
        logger.error(f"Error getting all categories: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/categories', methods=['GET'])
@login_required
def get_categories():
    """API endpoint to get all categories for the current user - alias for /api/categories/all"""
    try:
        # Get current user's ID from UserContext
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user context found for request")
            return jsonify({
                'status': 'error',
                'message': 'User not authenticated'
            }), 401

        # Get database connection
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return jsonify({
                'status': 'error',
                'message': 'Database connection error'
            }), 500
        
        try:
            # Create search instance with user context
            searcher = BookmarkSearchMultiUser(conn, user.id)
            
            try:
                # Get categories with counts - no parameters needed
                categories = searcher.get_categories()
                
                # Sort alphabetically by name
                categories.sort(key=lambda x: x['name'])
                
                return jsonify({
                    'status': 'success',
                    'categories': categories
                })
            except Exception as e:
                logger.error(f"Error calling get_categories: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'status': 'error',
                    'message': f'Error retrieving categories: {str(e)}',
                    'categories': []
                }), 500
        
        finally:
            # Ensure connection is properly handled
            if hasattr(conn, 'close') and not isinstance(conn, sqlalchemy.engine.base.Engine):
                conn.close()
    
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def check_tweet_content_column():
    """Check if the tweet_content column exists in bookmarks table and add it if missing"""

# Add a catch-all error handler for API routes to ensure they return JSON, not HTML
@app.errorhandler(Exception)
def handle_api_exception(e):
    """
    Handle ALL exceptions to ensure they return JSON instead of HTML error pages
    """
    # Get path for logging
    path = request.path if request else ""
    
    # Log the error with traceback
    logger.error(f"GLOBAL ERROR HANDLER: {path}: {str(e)}")
    logger.error(traceback.format_exc())
    
    # ALWAYS return JSON for ALL routes - never return HTML for errors
    return jsonify({
        'success': False,
        'error': str(e),
        'path': path,
        'type': e.__class__.__name__
    }), 500

# Add specific error handlers for various HTTP errors
@app.errorhandler(400)
def handle_bad_request(e):
    """Handle 400 Bad Request errors"""
    # Get the path to check if this is an API or upload route
    path = request.path if request else ""
    
    # Only apply JSON handling to API routes and specific endpoints
    if path.startswith('/api/') or path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        logger.error(f"Bad request error on {path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Bad Request: ' + str(e),
            'path': path
        }), 400
    
    # For non-API routes, let the default handler manage it
    return e

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 Not Found errors"""
    # Get the path to check if this is an API or upload route
    path = request.path if request else ""
    
    # Only apply JSON handling to API routes and specific endpoints
    if path.startswith('/api/') or path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        logger.error(f"Not found error on {path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Not Found: The requested resource does not exist',
            'path': path
        }), 404
    
    # For non-API routes, let the default handler manage it
    return e

@app.errorhandler(405)
def handle_method_not_allowed(e):
    """Handle 405 Method Not Allowed errors"""
    # Get the path to check if this is an API or upload route
    path = request.path if request else ""
    
    # Only apply JSON handling to API routes and specific endpoints
    if path.startswith('/api/') or path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        logger.error(f"Method not allowed error on {path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Method Not Allowed: The method is not allowed for this endpoint',
            'path': path,
            'allowed_methods': e.valid_methods if hasattr(e, 'valid_methods') else None
        }), 405
    
    # For non-API routes, let the default handler manage it
    return e

@app.errorhandler(500)
def handle_server_error(e):
    """Handle 500 Internal Server Error"""
    # Get the path to check if this is an API or upload route
    path = request.path if request else ""
    
    # Only apply JSON handling to API routes and specific endpoints
    if path.startswith('/api/') or path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        logger.error(f"Server error on {path}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal Server Error: The server encountered an unexpected condition',
            'path': path
        }), 500
    
    # For non-API routes, let the default handler manage it
    return e

@app.after_request
def add_header(response):
    """Add headers to ensure proper caching and security."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Force Content-Type to be application/json for API routes
    if (request.path.startswith('/api/') or 
        request.path in ['/upload-bookmarks', '/process-bookmarks', '/update-database'] or
        hasattr(g, 'is_json_api')):
        
        # Check if the response is an error response (4xx or 5xx)
        is_error = 400 <= response.status_code < 600
        
        # If this is a JSON API route, force Content-Type to be application/json
        # EVEN IF the response was generated by an error handler that set a different type
        if is_error or not response.headers.get('Content-Type') or 'text/html' in response.headers.get('Content-Type', ''):
            response.headers['Content-Type'] = 'application/json'
            
            # If the response body is HTML but this is a JSON API route, convert it to a JSON error message
            if 'text/html' in response.headers.get('Content-Type', '') or '<!DOCTYPE' in response.get_data(as_text=True)[:20]:
                error_message = {
                    'success': False,
                    'error': 'An error occurred on the server',
                    'status_code': response.status_code,
                    'path': request.path
                }
                response.set_data(json.dumps(error_message))
        
    return response

@app.errorhandler(json.JSONDecodeError)
def handle_json_error(e):
    """Handle JSON decode errors"""
    logger.error(f"JSON decode error: {str(e)}")
    return jsonify({
        'success': False,
        'error': f'Invalid JSON format: {str(e)}',
        'path': request.path
    }), 400

# Add a specific handler for vector store errors
@app.errorhandler(RuntimeError)
def handle_runtime_error(e):
    """Handle RuntimeError specifically for vector store access issues"""
    error_str = str(e)
    logger.error(f"Runtime error: {error_str}")
    
    # Check if this is a vector store access error
    if "Storage folder" in error_str and "already accessed by another instance" in error_str:
        return jsonify({
            'success': False,
            'error': "Vector database is busy. Please try again in a few minutes.",
            'path': request.path,
            'technical_details': error_str
        }), 503  # Service Unavailable
    
    # For other runtime errors, return a generic message
    return jsonify({
        'success': False,
        'error': f"Server runtime error occurred",
        'path': request.path
    }), 500

# Add a global error interceptor middleware to ensure JSON responses
@app.before_request
def ensure_json_response():
    """Ensure all API routes return JSON even when errors occur"""
    # Only apply to API routes and specific endpoints
    if request.path.startswith('/api/') or request.path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        # Store a flag in g to indicate this is a JSON API route
        g.is_json_api = True

@app.route('/emergency-upload', methods=['POST'])
@login_required
def emergency_upload():
    """
    Emergency file upload endpoint with minimal dependencies
    Designed to work even when other endpoints have issues
    """
    try:
        # Get current authenticated user
        user = UserContext.get_current_user()
        if not user:
            return app.response_class(
                response=json.dumps({
                    "success": False, 
                    "error": "User not authenticated"
                }),
                status=401,
                mimetype='application/json'
            )
            
        user_id = user.id
        
        # Check if file part exists
        if 'file' not in request.files:
            return app.response_class(
                response=json.dumps({
                    "success": False, 
                    "error": "No file part in the request"
                }),
                status=400,
                mimetype='application/json'
            )
            
        file = request.files['file']
        if file.filename == '':
            return app.response_class(
                response=json.dumps({
                    "success": False, 
                    "error": "No file selected"
                }),
                status=400,
                mimetype='application/json'
            )
            
        # Check if it's a JSON file
        if not file.filename.lower().endswith('.json'):
            return app.response_class(
                response=json.dumps({
                    "success": False, 
                    "error": "File must be a JSON file"
                }),
                status=400,
                mimetype='application/json'
            )
            
        # Create directory for user
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"emergency_upload_{timestamp}.json"
        filepath = os.path.join(upload_dir, filename)
        
        try:
            file.save(filepath)
            file_size = os.path.getsize(filepath)
            logger.info(f"Emergency upload saved file: {filepath} - Size: {file_size} bytes")
            
            # Minimal validation - just check if it at least looks like JSON
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_char = f.read(1).strip()
                    if first_char not in ['{', '[']:
                        return app.response_class(
                            response=json.dumps({
                                "success": False, 
                                "error": "File doesn't appear to be valid JSON",
                                "file_saved": True,
                                "filepath": filepath
                            }),
                            status=400,
                            mimetype='application/json'
                        )
            except Exception as read_error:
                # Log but don't fail if we can't validate
                logger.error(f"Error validating uploaded file: {str(read_error)}")
            
            # Return success directly with application/json content type
            return app.response_class(
                response=json.dumps({
                    "success": True,
                    "message": "File uploaded successfully via emergency upload",
                    "file": filename,
                    "path": filepath
                }),
                status=200,
                mimetype='application/json'
            )
            
        except Exception as save_error:
            logger.error(f"Error in emergency upload saving file: {str(save_error)}")
            return app.response_class(
                response=json.dumps({
                    "success": False,
                    "error": f"Error saving file: {str(save_error)}"
                }),
                status=500,
                mimetype='application/json'
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in emergency upload: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Always return JSON even on unexpected errors
        return app.response_class(
            response=json.dumps({
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }),
            status=500,
            mimetype='application/json'
        )

@app.route('/direct-upload', methods=['POST'])
@login_required
def direct_upload():
    """Handle bookmark JSON file upload without any vector store interaction
    This is a simplified version of the upload process that completely avoids
    vector store operations to ensure reliable file uploads even when vector
    operations might fail.
    """
    # Generate a unique ID for this upload session
    session_id = str(uuid.uuid4())[:8]
    temp_path = None
    
    try:
        logger.info("="*80)
        logger.info(f"🚀 [UPLOAD-{session_id}] Starting direct upload handler at {datetime.now().isoformat()}")
        
        # Get current user ID
        user = UserContext.get_current_user()
        if not user:
            raise ValueError("User not found or not authenticated")
        
        user_id = user.id
        logger.info(f"👤 [UPLOAD-{session_id}] User ID: {user_id}")
        
        # 1. Check if file exists in request
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 1: Checking if file exists in request")
        if 'file' not in request.files:
            logger.error(f"❌ [UPLOAD-{session_id}] No file part in request")
            return jsonify({
                'error': 'No file provided',
                'details': {
                    'request_method': request.method,
                    'has_files': bool(request.files),
                    'form_keys': list(request.form.keys()) if request.form else None
                }
            }), 400
        
        file = request.files['file']
        logger.info(f"✅ [UPLOAD-{session_id}] Received file object: {file}")
        logger.info(f"📄 [UPLOAD-{session_id}] File name: {file.filename}")
        
        # 2. Validate file name
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 2: Validating file name")
        if not file.filename:
            logger.error(f"❌ [UPLOAD-{session_id}] No selected file")
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.endswith('.json'):
            logger.error(f"❌ [UPLOAD-{session_id}] Invalid file type: {file.filename}")
            return jsonify({'error': 'Only JSON files are allowed'}), 400
        
        # 3. Validate JSON content
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 3: Validating JSON content")
        try:
            file_content = file.read()
            file.seek(0)
            json_data = json.loads(file_content)
            logger.info(f"✅ [UPLOAD-{session_id}] JSON validation successful")
            logger.info(f"📊 [UPLOAD-{session_id}] JSON content size: {len(file_content)} bytes")
            
            # Log some basic statistics about the JSON data
            if isinstance(json_data, list):
                logger.info(f"📊 [UPLOAD-{session_id}] JSON contains a list with {len(json_data)} items")
            elif isinstance(json_data, dict):
                logger.info(f"📊 [UPLOAD-{session_id}] JSON contains a dictionary with {len(json_data.keys())} keys")
                if 'bookmarks' in json_data:
                    logger.info(f"📊 [UPLOAD-{session_id}] JSON contains {len(json_data['bookmarks'])} bookmarks")
        except json.JSONDecodeError as e:
            logger.error(f"❌ [UPLOAD-{session_id}] Invalid JSON file: {str(e)}")
            return jsonify({'error': 'Invalid JSON file: ' + str(e)}), 400
        
        # 4. Create user-specific directories
        if file.filename == '':
            return app.response_class(
                response=json.dumps({'success': False, 'error': 'No file selected'}),
                status=400,
                mimetype='application/json'
            )
            
        # Check if it's a JSON file
        if not file.filename.lower().endswith('.json'):
            return app.response_class(
                response=json.dumps({'success': False, 'error': 'File must be a JSON file'}),
                status=400,
                mimetype='application/json'
            )
            
        # Ensure directory exists for this user
        # Ensure UPLOAD_FOLDER is configured
        if 'UPLOAD_FOLDER' not in app.config:
            logger.warning(f"⚠️ [UPLOAD-{session_id}] UPLOAD_FOLDER not configured, setting default")
            app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            logger.info(f"✅ [UPLOAD-{session_id}] Created UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
            
        # Now create user-specific directory
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"📁 [UPLOAD-{session_id}] Created user upload directory: {upload_dir}")
        
        # Create database directory if not exists
        # Also ensure DATABASE_DIR is configured
        if 'DATABASE_DIR' not in app.config:
            logger.warning(f"⚠️ [UPLOAD-{session_id}] DATABASE_DIR not configured, setting default")
            app.config['DATABASE_DIR'] = os.path.join(BASE_DIR, 'database')
            os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
            logger.info(f"✅ [UPLOAD-{session_id}] Created DATABASE_DIR at {app.config['DATABASE_DIR']}")
            
        database_dir = os.path.join(app.config['DATABASE_DIR'], f'user_{user_id}')
        os.makedirs(database_dir, exist_ok=True)
        logger.info(f"📁 [UPLOAD-{session_id}] Created user database directory: {database_dir}")
        
        # Save the file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"twitter_bookmarks_{timestamp}.json"
        upload_path = os.path.join(upload_dir, filename)
        target_path = os.path.join(database_dir, 'twitter_bookmarks.json')
        
        try:
            # Save to uploads folder
            file.save(upload_path)
            logger.info(f"File saved to {upload_path}")
            
            # Do a simple test that the file is valid JSON
            try:
                with open(upload_path, 'r', encoding='utf-8') as f:
                    # Just read the first line to check it starts with valid JSON characters
                    first_chars = f.read(10).strip()
                    if not first_chars.startswith('{') and not first_chars.startswith('['):
                        return app.response_class(
                            response=json.dumps({
                                'success': False, 
                                'error': 'File does not appear to be valid JSON',
                                'first_chars': first_chars,
                                'file_saved': True,
                                'path': upload_path
                            }),
                            status=400,
                            mimetype='application/json'
                        )
            except Exception as json_error:
                logger.error(f"Error checking JSON: {json_error}")
                return app.response_class(
                    response=json.dumps({
                        'success': False, 
                        'error': f'Error checking JSON file: {str(json_error)}',
                        'file_saved': True,
                        'path': upload_path
                    }),
                    status=400,
                    mimetype='application/json'
                )
                
            # Copy to database directory
            try:
                shutil.copy2(upload_path, target_path)
                logger.info(f"File copied to database directory: {target_path}")
            except Exception as copy_error:
                logger.error(f"Error copying file: {copy_error}")
                return app.response_class(
                    response=json.dumps({
                        'success': False, 
                        'error': f'Error copying file to database: {str(copy_error)}',
                        'file_saved': True,
                        'upload_path': upload_path
                    }),
                    status=500,
                    mimetype='application/json'
                )
            
            # Return success
            return app.response_class(
                response=json.dumps({
                    'success': True,
                    'message': 'File uploaded successfully via direct upload endpoint',
                    'file': filename,
                    'upload_path': upload_path,
                    'database_path': target_path,
                    'timestamp': timestamp
                }),
                status=200,
                mimetype='application/json'
            )
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return app.response_class(
                response=json.dumps({'success': False, 'error': str(e)}),
                status=500,
                mimetype='application/json'
            )
            
    except Exception as e:
        logger.error(f"❌ [UPLOAD-{session_id}] Unexpected error in direct upload: {e}")
        logger.error(traceback.format_exc())
        
        # Check if this is a configuration issue and try to fix it automatically
        error_message = str(e)
        if "UPLOAD_FOLDER" in error_message:
            # Fix missing UPLOAD_FOLDER configuration
            if 'UPLOAD_FOLDER' not in app.config:
                app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                logger.info(f"✅ [UPLOAD-{session_id}] Created missing UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
            error_message = "Server configuration issue with upload folder. It has been fixed, please try again."
            
        elif "DATABASE_DIR" in error_message:
            # Fix missing DATABASE_DIR configuration
            if 'DATABASE_DIR' not in app.config:
                app.config['DATABASE_DIR'] = os.path.join(BASE_DIR, 'database')
                os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
                logger.info(f"✅ [UPLOAD-{session_id}] Created missing DATABASE_DIR at {app.config['DATABASE_DIR']}")
            error_message = "Server configuration issue with database folder. It has been fixed, please try again."
        
        # Always return JSON even on unexpected errors
        return app.response_class(
            response=json.dumps({
                "success": False,
                "error": f"Upload error: {error_message}",
                "retry_recommended": True
            }),
            status=500,
            mimetype='application/json'
        )

# Run the app if this file is executed directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port} with debug={debug}")
    
    # Number of attempts to connect to the database
    max_db_attempts = 5
    db_attempt = 0
    db_initialized = False
    
    while db_attempt < max_db_attempts and not db_initialized:
        try:
            # Initialize the database
            init_database()
            logger.info("Database initialized successfully")
            db_initialized = True
        except Exception as e:
            db_attempt += 1
            wait_time = 2 ** db_attempt  # Exponential backoff
            logger.error(f"Database initialization attempt {db_attempt}/{max_db_attempts} failed: {e}")
            
            if db_attempt < max_db_attempts:
                logger.info(f"Retrying database initialization in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to initialize database after {max_db_attempts} attempts")
                # Continue anyway - we'll try again when a request comes in
    
    try:
        # Run the app
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.exception(e) 

def safe_get_vector_store():
    """
    Safely attempt to get a vector store instance, handling all possible errors
    without propagating exceptions to the caller. This completely isolates vector 
    store operations from the rest of the application.
    
    Returns:
        VectorStore or None: The initialized vector store or None if any error occurred
    """
    try:
        # First try to import the module
        logger.info("Attempting to import VectorStoreMultiUser")
        try:
            from database.multi_user_db.vector_store_final import VectorStoreMultiUser
            logger.info("Successfully imported VectorStoreMultiUser")
        except ImportError as e:
            logger.error(f"Import error for VectorStoreMultiUser: {e}")
            logger.error(traceback.format_exc())
            return None
        
        # Then try to initialize it
        logger.info("Attempting to initialize VectorStoreMultiUser")
        try:
            user = UserContext.get_current_user()
            if not user:
                logger.warning("No user context found, cannot initialize vector store")
                return None
                
            user_id = user.id
            vector_store = VectorStoreMultiUser(user_id=user_id)
            logger.info(f"Successfully initialized vector store for user {user_id}")
            return vector_store
        except Exception as e:
            # Check for common errors
            if "Storage folder is locked" in str(e) or "already being accessed" in str(e):
                logger.warning(f"Vector store folder is locked, another instance is accessing it: {e}")
            elif "Permission denied" in str(e):
                logger.error(f"Permission denied when accessing vector store: {e}")
            else:
                logger.error(f"Error initializing vector store: {e}")
                logger.error(traceback.format_exc())
            return None
    except Exception as e:
        # Catch absolutely any error to avoid breaking API requests
        logger.error(f"Unexpected error in safe_get_vector_store: {e}")
        logger.error(traceback.format_exc())
        return None

@app.route('/api/rebuild-vector-store', methods=['POST'])
@login_required
def rebuild_vector_store_endpoint():
    """API endpoint to rebuild the vector store from the database
    This is separated from the database update process to avoid
    vector store errors affecting the database update.
    """
    try:
        # Get current user
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user context found for request")
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
            
        user_id = user.id
        logger.info(f"Starting vector store rebuild for user {user_id}")
        
        # Get request JSON data
        data = request.get_json() or {}
        session_id = data.get('session_id', str(uuid.uuid4())[:8])
        
        # Generate a status file
        user_dir = os.path.join(app.config.get('DATABASE_DIR', '/app/database'), f'user_{user_id}')
        status_file = os.path.join(user_dir, f"vector_rebuild_{session_id}.json")
        
        with open(status_file, 'w') as f:
            status = {
                'user_id': user_id,
                'session_id': session_id,
                'status': 'processing',
                'message': 'Vector store rebuild started',
                'start_time': datetime.now().isoformat(),
                'progress': 0
            }
            json.dump(status, f)
            
        # Launch rebuild process in background thread
        def rebuild_process():
            from datetime import datetime
            import traceback
            
            start_time = datetime.now()
            
            try:
                # Run the vector store rebuild
                from database.multi_user_db.update_bookmarks_final import rebuild_vector_store
                
                with app.app_context():
                    try:
                        # Log the start of the rebuild
                        logger.info(f"Starting vector store rebuild in background - session_id={session_id}")
                        
                        # Run the rebuild process
                        result = rebuild_vector_store(
                            session_id=session_id,
                            user_id=user_id
                        )
                        
                        # Update status file with result
                        with open(status_file, 'w') as f:
                            status = {
                                'user_id': user_id,
                                'session_id': session_id,
                                'status': 'completed' if result.get('success', False) else 'error',
                                'message': 'Vector store rebuild completed' if result.get('success', False) else 'Vector store rebuild failed',
                                'result': result,
                                'start_time': start_time.isoformat(),
                                'end_time': datetime.now().isoformat(),
                                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                                'progress': 100 if result.get('success', False) else 0
                            }
                            json.dump(status, f)
                            
                        logger.info(f"Vector store rebuild completed - session_id={session_id}")
                        
                    except Exception as e:
                        # Log error and update status file
                        logger.error(f"Error in vector store rebuild process: {e}")
                        logger.error(traceback.format_exc())
                        
                        with open(status_file, 'w') as f:
                            status = {
                                'user_id': user_id,
                                'session_id': session_id,
                                'status': 'error',
                                'message': f'Error rebuilding vector store: {str(e)}',
                                'error': str(e),
                                'traceback': traceback.format_exc(),
                                'start_time': start_time.isoformat(),
                                'end_time': datetime.now().isoformat(),
                                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                                'progress': 0
                            }
                            json.dump(status, f)
            except Exception as e:
                # Handle outer exceptions
                logger.error(f"Fatal error in rebuild thread: {e}")
                logger.error(traceback.format_exc())
                
                try:
                    with open(status_file, 'w') as f:
                        status = {
                            'user_id': user_id,
                            'session_id': session_id,
                            'status': 'error',
                            'message': f'Fatal error in rebuild process: {str(e)}',
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'start_time': start_time.isoformat(),
                            'end_time': datetime.now().isoformat(),
                            'duration_seconds': (datetime.now() - start_time).total_seconds(),
                            'progress': 0
                        }
                        json.dump(status, f)
                except Exception:
                    pass
        
        # Start the rebuild process in a background thread
        rebuild_thread = Thread(target=rebuild_process)
        rebuild_thread.daemon = True
        rebuild_thread.start()
        
        # Return immediate success response with session ID for tracking
        return jsonify({
            'success': True,
            'message': 'Vector store rebuild started in background',
            'session_id': session_id,
            'status_file': os.path.basename(status_file),
            'next_step': {
                'endpoint': '/api/process-status',
                'method': 'GET',
                'params': {'session_id': session_id}
            }
        })
        
    except Exception as e:
        logger.error(f"Error in rebuild_vector_store endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/simplest-upload', methods=['POST'])
@login_required
def simplest_upload():
    """Absolute minimum file upload endpoint with zero dependencies on other code.
    This endpoint does one thing only: save the uploaded file to disk.
    No vector store, no validation, no fancy error handling - just save the file.
    """
    try:
        # Get current user ID directly from Flask session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file in request'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': 'Empty file name'}), 400
            
        # Ensure UPLOAD_FOLDER is configured
        if 'UPLOAD_FOLDER' not in app.config:
            app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            logger.info(f"Created missing UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
            
        # Create user upload directory
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
        os.makedirs(user_dir, exist_ok=True)
        
        # Save file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"upload_{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(user_dir, filename)
        
        file.save(filepath)
        
        # Ensure DATABASE_DIR is configured
        if 'DATABASE_DIR' not in app.config:
            app.config['DATABASE_DIR'] = os.path.join(BASE_DIR, 'database')
            os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
            logger.info(f"Created missing DATABASE_DIR at {app.config['DATABASE_DIR']}")
            
        # Copy to standard location
        db_dir = os.path.join(app.config['DATABASE_DIR'], f'user_{user_id}')
        os.makedirs(db_dir, exist_ok=True)
        
        target_path = os.path.join(db_dir, 'twitter_bookmarks.json')
        shutil.copy2(filepath, target_path)
        
        return jsonify({
            'success': True,
            'message': 'File saved successfully',
            'original_path': filepath,
            'target_path': target_path
        })
        
    except Exception as e:
        # More robust error handling
        logger.error(f"Error in simplest_upload: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to fix the error if it's related to configuration
        error_message = str(e)
        if "UPLOAD_FOLDER" in error_message:
            app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            logger.info(f"Created missing UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
            error_message = "Server configuration issue with upload folder. It has been fixed, please try again."
        elif "DATABASE_DIR" in error_message:
            app.config['DATABASE_DIR'] = os.path.join(BASE_DIR, 'database')
            os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
            logger.info(f"Created missing DATABASE_DIR at {app.config['DATABASE_DIR']}")
            error_message = "Server configuration issue with database folder. It has been fixed, please try again."
        
        return jsonify({
            'success': False,
            'error': f"Upload error: {error_message}",
            'retry_recommended': True
        }), 500

# Error handlers - MUST return JSON for all errors
@app.errorhandler(Exception)
def handle_all_errors(e):
    """Catch-all error handler to ensure JSON responses for all errors"""
    logger.error(f"Unhandled error: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        'success': False,
        'error': str(e),
        'error_type': e.__class__.__name__,
        'path': request.path
    }), 500

@app.errorhandler(404)
def handle_not_found(e):
    """Return JSON for 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Resource not found',
        'path': request.path
    }), 404

@app.errorhandler(401)
def handle_unauthorized(e):
    """Return JSON for 401 errors"""
    return jsonify({
        'success': False,
        'error': 'Unauthorized',
        'path': request.path
    }), 401

@app.errorhandler(400)
def handle_bad_request(e):
    """Return JSON for 400 errors"""
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'path': request.path
    }), 400

# Add a route for checking authentication status
@app.route('/check-auth', methods=['GET'])
def check_auth():
    """Simple endpoint to check authentication status without database operations"""
    user_id = session.get('user_id')
    # Log session information for debugging
    logger.info(f"Auth check - Session: {session.sid if hasattr(session, 'sid') else 'No SID'} - User ID: {user_id}")
    
    # Check if user_id exists in session
    if user_id:
        # Don't do database lookup to avoid potential errors
        # Just return success based on session information
        logger.info(f"Auth check success - user_id found in session: {user_id}")
        return jsonify({
            'success': True,
            'authenticated': True,
            'message': 'User is authenticated via session',
            'user_id': user_id
        })
    else:
        logger.warning("Auth check failed - No user_id in session")
        return jsonify({
            'success': False,
            'authenticated': False,
            'error': 'User not authenticated. Please log out and log in again.'
        }), 401

# Global session status tracking