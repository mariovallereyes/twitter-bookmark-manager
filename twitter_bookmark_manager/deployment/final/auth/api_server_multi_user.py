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

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory, abort, g
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
    get_user_directory
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

# Create Flask app
app = Flask(__name__, 
            template_folder='../web_final/templates',
            static_folder='../web_final/static')

# Configure CORS - Allow all origins during deployment
CORS(app, supports_credentials=True)

# Configure app
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', secrets.token_hex(32)),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    get_db_connection=get_db_connection,
    PREFERRED_URL_SCHEME='https',
    MAX_CONTENT_LENGTH=32 * 1024 * 1024,  # 32MB max file size
    UPLOAD_FOLDER=UPLOADS_DIR,
    DATABASE_DIR=DATABASE_DIR  # Add DATABASE_DIR to the configuration
)

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
    """Save session status to the SQLite database"""
    try:
        # Convert Python dict to JSON string
        data_json = json.dumps(status_data.get('data', {}))
        
        conn = sqlite3.connect(STATUS_DB_PATH)
        cursor = conn.cursor()
        
        # Check if session exists
        cursor.execute("SELECT 1 FROM session_status WHERE session_id = ?", (session_id,))
        exists = cursor.fetchone() is not None
        
        now = datetime.now().isoformat()
        
        if exists:
            # Update existing record
            cursor.execute('''
            UPDATE session_status SET
                status = ?,
                message = ?,
                timestamp = ?,
                is_complete = ?,
                success = ?,
                data = ?,
                updated_at = ?
            WHERE session_id = ?
            ''', (
                status_data.get('status', 'unknown'),
                status_data.get('message', ''),
                status_data.get('timestamp', now),
                1 if status_data.get('is_complete', False) else 0,
                1 if status_data.get('success', False) else 0,
                data_json,
                now,
                session_id
            ))
        else:
            # Insert new record
            cursor.execute('''
            INSERT INTO session_status
            (session_id, user_id, status, message, timestamp, is_complete, success, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                status_data.get('user_id', ''),
                status_data.get('status', 'unknown'),
                status_data.get('message', ''),
                status_data.get('timestamp', now),
                1 if status_data.get('is_complete', False) else 0,
                1 if status_data.get('success', False) else 0,
                data_json,
                now,
                now
            ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving session status: {str(e)}")
        logger.error(traceback.format_exc())
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
            # Use retry logic for health check
            max_attempts = 3
            attempt = 0
            last_error = None
            
            while attempt < max_attempts:
                try:
                    health = check_engine_health()
                    if not health['healthy']:
                        logger.warning(f"Database health check failed: {health['message']}")
                        
                        # Force reconnect on unhealthy status
                        setup_database(force_reconnect=True)
                        logger.info("Forced database reconnection after unhealthy status")
                    
                    # If we get here, either health check passed or we reconnected
                    return
                    
                except Exception as e:
                    attempt += 1
                    last_error = e
                    wait_time = 0.5 * (2 ** (attempt - 1))  # Short exponential backoff
                    
                    if attempt < max_attempts:
                        logger.warning(f"Health check attempt {attempt} failed, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Health check failed after {max_attempts} attempts: {e}")
                        # Don't fail the request, just log the issue
    except Exception as e:
        logger.error(f"Error in health check routine: {e}")
        # Continue processing the request even if health check fails

# Set session to be permanent by default
@app.before_request
def make_session_permanent():
    session.permanent = True

# Register authentication blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(user_api_bp)

# Initialize user context middleware
UserContextMiddleware(app, lambda user_id: get_user_by_id(get_db_connection(), user_id))

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
                
        conn.close()
    except Exception as e:
        logger.error(f"STARTUP DEBUG - Error: {e}")
        logger.error(traceback.format_exc())

# Call the init function right away
init_app_debug()

# Define login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = UserContext.get_current_user()
        if user is None:
            # Check if this is an API request based on the URL or Accept header
            is_api_request = (
                request.path.startswith('/api/') or 
                request.headers.get('Accept') == 'application/json' or
                request.headers.get('Content-Type') == 'application/json'
            )
            
            if is_api_request:
                # Return a JSON error for API requests
                return jsonify({
                    'success': False,
                    'authenticated': False,
                    'error': 'User not authenticated'
                }), 401
            else:
                # Redirect to login for browser requests
                return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

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
                categories = searcher.get_categories()
                logger.info(f"Successfully loaded {len(categories)} categories via SQLAlchemy for user {user.id}")
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
    
    # Get latest bookmarks
    latest_tweets = []
    try:
        if not all_methods_tried:
            # Connect to database
            conn = get_db_connection()
            try:
                # Create a searcher instance
                searcher = BookmarkSearchMultiUser(conn, user.id if user else 1)
                
                # Get 5 most recent bookmarks
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
        logger.warning(f"Failed to retrieve latest bookmarks: {e}")
        logger.error(traceback.format_exc())
    
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

@app.route('/upload-bookmarks', methods=['POST'])
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
            
        logger.info(f"File received: {file.filename} for user {user_id}")
        
        # Ensure upload directory exists
        upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
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
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    bookmark_count = len(json_data)
                    logger.info(f"JSON validated - Contains {bookmark_count} entries")
            except json.JSONDecodeError as je:
                logger.error(f"Invalid JSON format: {str(je)}")
                return jsonify({'success': False, 'error': f'Invalid JSON format: {str(je)}'}), 400
            
            # Return success with session ID for later processing
            session_id = str(uuid.uuid4())
            return jsonify({
                'success': True, 
                'message': 'File uploaded successfully',
                'file': filename,
                'session_id': session_id
            })
            
        except Exception as save_error:
            logger.error(f"Error saving file: {str(save_error)}")
            logger.error(traceback.format_exc())
            return jsonify({'success': False, 'error': f'Error saving file: {str(save_error)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in upload_bookmarks: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process-bookmarks', methods=['POST'])
@login_required
def process_bookmarks():
    """Process bookmarks from a JSON file."""
    try:
        user = UserContext.get_current_user()
        if not user:
            logger.error("User not authenticated")
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
        
        user_id = user.id
        logger.info(f"Processing bookmarks for user {user_id}")
        
        # Check if we have a file or a direct JSON payload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
                
            # Save the file temporarily
            uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
            os.makedirs(uploads_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bookmarks_{timestamp}.json"
            filepath = os.path.join(uploads_dir, filename)
            
            file.save(filepath)
            file_size = os.path.getsize(filepath)
            logger.info(f"File saved: {filepath} - Size: {file_size} bytes")
            
        elif request.is_json:
            # Direct JSON payload
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
                
            # Save as a file for consistent processing
            uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
            os.makedirs(uploads_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bookmarks_{timestamp}.json"
            filepath = os.path.join(uploads_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
                
            logger.info(f"JSON data saved to file: {filepath}")
        else:
            return jsonify({'success': False, 'error': 'No file or JSON data provided'}), 400
            
        # Copy the uploaded file to the database directory for processing
        db_dir = os.path.join(app.config['DATABASE_DIR'], f'user_{user_id}')
        os.makedirs(db_dir, exist_ok=True)
        
        target_file = os.path.join(db_dir, 'twitter_bookmarks.json')
        shutil.copy2(filepath, target_file)
        
        logger.info(f"Copied uploaded file to {target_file} for processing")
        
        # Process the JSON file
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                bookmark_data = json.load(f)
                
            # Extracting bookmarks from different possible structures
            bookmarks_list = []
            
            if isinstance(bookmark_data, list):
                # Direct list of bookmarks
                bookmarks_list = bookmark_data
            elif isinstance(bookmark_data, dict):
                # Try different possible structures
                if 'bookmarks' in bookmark_data:
                    bookmarks_list = bookmark_data['bookmarks']
                elif 'tweet' in bookmark_data and isinstance(bookmark_data['tweet'], list):
                    # Handle structure where tweets are directly in a "tweet" field
                    bookmarks_list = bookmark_data['tweet']
                    
            logger.info(f"Found {len(bookmarks_list)} bookmarks to process")
            
            if not bookmarks_list:
                return jsonify({'success': False, 'error': 'No bookmarks found in file'}), 400
                
            # Get database connection
            try:
                db_conn = get_db_connection()
                
                # Check if we got a SQLAlchemy connection or a psycopg2 connection
                is_sqlalchemy_conn = hasattr(db_conn, 'execute') and not hasattr(db_conn, 'cursor')
                
                if is_sqlalchemy_conn:
                    # Using SQLAlchemy connection
                    # Get existing bookmark IDs
                    result = db_conn.execute(
                        text("SELECT bookmark_id FROM bookmarks WHERE user_id = :user_id"),
                        {"user_id": user_id}
                    )
                    existing_bookmark_ids = {row[0] for row in result}
                    
                    # Process each bookmark
                    processed = 0
                    skipped = 0
                    errors = 0
                    
                    for bookmark in bookmarks_list:
                        try:
                            # Handle either direct tweet or nested tweet structure
                            tweet = bookmark.get('tweet', bookmark)
                            
                            # Extract bookmark ID - try various possible fields
                            bookmark_id = None
                            for id_field in ['id_str', 'id', 'tweet_id', 'rest_id']:
                                if id_field in tweet:
                                    bookmark_id = str(tweet[id_field])
                                    break
                                    
                            # New format: extract ID from tweet_url if available
                            if not bookmark_id and 'tweet_url' in tweet:
                                # URLs like https://x.com/username/status/1234567890123456789
                                tweet_url = tweet['tweet_url']
                                if '/status/' in tweet_url:
                                    # Extract the ID part after /status/
                                    parts = tweet_url.split('/status/')
                                    if len(parts) > 1:
                                        # Get everything after /status/ and before any subsequent /
                                        potential_id = parts[1].split('/')[0]
                                        if potential_id.isdigit():
                                            bookmark_id = potential_id
                                            logger.info(f"Extracted ID {bookmark_id} from tweet_url")
                                    
                            if not bookmark_id:
                                logger.warning(f"Could not find ID in bookmark: {tweet}")
                                errors += 1
                                continue
                                
                            # Skip duplicates
                            if bookmark_id in existing_bookmark_ids:
                                logger.info(f"Skipping duplicate bookmark: {bookmark_id}")
                                skipped += 1
                                continue
                                
                            # Extract content
                            tweet_text = tweet.get('full_text', tweet.get('text', ''))
                            
                            # Handle different date formats
                            created_at = tweet.get('created_at', '')
                            
                            # Extract author information
                            author = ''
                            author_id = ''
                            
                            # Try to find author info in various possible locations
                            user_info = tweet.get('user', {})
                            if user_info:
                                author = user_info.get('screen_name', user_info.get('username', ''))
                                author_id = user_info.get('id_str', user_info.get('id', ''))
                                
                            # Store raw data for vector embedding
                            tweet_content = json.dumps(tweet)
                            
                            # Insert using SQLAlchemy connection
                            db_conn.execute(
                                text("""
                                    INSERT INTO bookmarks 
                                    (bookmark_id, user_id, text, tweet_content, created_at, author, author_id, processed) 
                                    VALUES (:bookmark_id, :user_id, :text, :tweet_content, :created_at, :author, :author_id, :processed)
                                    ON CONFLICT (bookmark_id) DO NOTHING
                                """),
                                {
                                    "bookmark_id": bookmark_id,
                                    "user_id": user_id,
                                    "text": tweet_text,
                                    "tweet_content": tweet_content,
                                    "created_at": created_at,
                                    "author": author,
                                    "author_id": author_id,
                                    "processed": False
                                }
                            )
                            
                            # Track the new ID to prevent duplicates in the same batch
                            existing_bookmark_ids.add(bookmark_id)
                            processed += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing bookmark: {str(e)}")
                            errors += 1
                    
                    # Commit transaction for SQLAlchemy
                    db_conn.commit()
                
                else:
                    # Using psycopg2 connection (traditional approach)
                    cursor = db_conn.cursor()
                    
                    # Check for existing bookmarks to avoid duplicates
                    cursor.execute(
                        "SELECT bookmark_id FROM bookmarks WHERE user_id = %s",
                        (user_id,)
                    )
                    existing_bookmark_ids = {row[0] for row in cursor.fetchall()}
                    
                    # Process each bookmark
                    processed = 0
                    skipped = 0
                    errors = 0
                    
                    for bookmark in bookmarks_list:
                        try:
                            # Handle either direct tweet or nested tweet structure
                            tweet = bookmark.get('tweet', bookmark)
                            
                            # Extract bookmark ID - try various possible fields
                            bookmark_id = None
                            for id_field in ['id_str', 'id', 'tweet_id', 'rest_id']:
                                if id_field in tweet:
                                    bookmark_id = str(tweet[id_field])
                                    break
                                    
                            # New format: extract ID from tweet_url if available
                            if not bookmark_id and 'tweet_url' in tweet:
                                # URLs like https://x.com/username/status/1234567890123456789
                                tweet_url = tweet['tweet_url']
                                if '/status/' in tweet_url:
                                    # Extract the ID part after /status/
                                    parts = tweet_url.split('/status/')
                                    if len(parts) > 1:
                                        # Get everything after /status/ and before any subsequent /
                                        potential_id = parts[1].split('/')[0]
                                        if potential_id.isdigit():
                                            bookmark_id = potential_id
                                            logger.info(f"Extracted ID {bookmark_id} from tweet_url")
                                    
                            if not bookmark_id:
                                logger.warning(f"Could not find ID in bookmark: {tweet}")
                                errors += 1
                                continue
                                
                            # Skip duplicates
                            if bookmark_id in existing_bookmark_ids:
                                logger.info(f"Skipping duplicate bookmark: {bookmark_id}")
                                skipped += 1
                                continue
                                
                            # Extract content
                            tweet_text = tweet.get('full_text', tweet.get('text', ''))
                            
                            # Handle different date formats
                            created_at = tweet.get('created_at', '')
                            
                            # Extract author information
                            author = ''
                            author_id = ''
                            
                            # Try to find author info in various possible locations
                            user_info = tweet.get('user', {})
                            if user_info:
                                author = user_info.get('screen_name', user_info.get('username', ''))
                                author_id = user_info.get('id_str', user_info.get('id', ''))
                                
                            # Store raw data for vector embedding
                            tweet_content = json.dumps(tweet)
                            
                            # Insert into database
                            cursor.execute("""
                                INSERT INTO bookmarks 
                                (bookmark_id, user_id, text, tweet_content, created_at, author, author_id, processed) 
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (bookmark_id) DO NOTHING
                            """, (
                                bookmark_id,
                                user_id,
                                tweet_text,
                                tweet_content,
                                created_at,
                                author,
                                author_id,
                                False
                            ))
                            
                            # Track the new ID to prevent duplicates in the same batch
                            existing_bookmark_ids.add(bookmark_id)
                            processed += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing bookmark: {str(e)}")
                            errors += 1
                    
                    # Commit changes for psycopg2
                    db_conn.commit()
                
                logger.info(f"Processed {processed} bookmarks, skipped {skipped}, errors: {errors}")
                
                return jsonify({
                    'success': True,
                    'message': f"Successfully processed {processed} bookmarks, skipped {skipped} duplicates",
                    'processed': processed,
                    'skipped': skipped,
                    'errors': errors
                })
                
            except Exception as db_error:
                logger.error(f"Database error: {str(db_error)}")
                if 'db_conn' in locals() and db_conn:
                    # Check if we have a SQLAlchemy connection (has execute method)
                    if hasattr(db_conn, 'execute') and not hasattr(db_conn, 'cursor'):
                        db_conn.rollback()
                    elif hasattr(db_conn, 'rollback'):  # psycopg2 connection
                        db_conn.rollback()
                return jsonify({'success': False, 'error': f'Database error: {str(db_error)}'}), 500
                
        except json.JSONDecodeError as je:
            logger.error(f"Invalid JSON format: {str(je)}")
            return jsonify({'success': False, 'error': f'Invalid JSON format: {str(je)}'}), 400
            
    except Exception as e:
        logger.error(f"Error in process_bookmarks: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process-status', methods=['GET'])
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

@app.route('/update-database', methods=['POST'])
def update_database():
    """
    Update the database with bookmarks, with option to rebuild vector store
    """
    try:
        # Get user info using UserContext
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user found for database update - UserContext check failed")
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
            
        user_id = user.id
        logger.info(f"Processing database update for user {user_id}")
        
        # Get rebuild flag
        data = request.get_json(silent=True) or {}
        rebuild = data.get('rebuild', True)
        direct_call = data.get('direct_call', False)
        
        if not rebuild:
            logger.info(f"Rebuild flag set to false, skipping for user {user_id}")
            return jsonify({
                'success': True,
                'message': 'No changes made to vector database (rebuild=false)'
            })
            
        # Create session ID for this rebuild
        session_id = str(uuid.uuid4())
        logger.info(f"Vector rebuild session {session_id} started for user {user_id}")
        
        # Call vector rebuild function
        try:
            if direct_call:
                # Use the direct approach from update_bookmarks_final
                logger.info(f"Using direct rebuild call for user {user_id}")
                
                # Import the rebuild function
                from database.multi_user_db.update_bookmarks_final import rebuild_vector_store
                
                # Execute the rebuild function
                result = rebuild_vector_store(user_id=user_id, session_id=session_id)
                
                if result.get('success', False):
                    logger.info(f"Direct vector rebuild successful for user {user_id}")
                    return jsonify({
                        'success': True,
                        'message': result.get('message', 'Vector database rebuilt successfully'),
                        'details': result
                    })
                else:
                    logger.error(f"Direct vector rebuild failed for user {user_id}: {result.get('error', 'Unknown error')}")
                    return jsonify({
                        'success': False,
                        'error': result.get('error', 'Failed to rebuild vector database'),
                        'details': result
                    }), 500
            else:
                # Use the vector store class directly
                logger.info(f"Using vector store class for rebuild (user {user_id})")
                
                # Import and get the vector store instance
                try:
                    from database.multi_user_db.vector_store_final import get_multi_user_vector_store
                    vector_store = get_multi_user_vector_store()
                except ImportError:
                    # Fall back to regular get_vector_store if the multi-user version isn't available
                    try:
                        from database.multi_user_db.vector_store_final import get_vector_store
                        vector_store = get_vector_store()
                    except Exception as import_error:
                        logger.error(f"Failed to import vector store: {str(import_error)}")
                        return jsonify({'success': False, 'error': f'Vector store import failed: {str(import_error)}'}), 500
                
                if not vector_store:
                    logger.error(f"Failed to get vector store for user {user_id}")
                    return jsonify({'success': False, 'error': 'Vector store initialization failed'}), 500
                    
                # Check for tweet_content column
                try:
                    check_tweet_content_column()
                except Exception as column_error:
                    logger.warning(f"Issue with tweet_content column check: {str(column_error)}")
                    # Continue as this is non-critical
                
                # Initiate rebuild with timeout protection
                try:
                    # Set a timeout for the operation (5 minutes)
                    max_time = 300  # seconds
                    start_time = time.time()
                    
                    # Initiate rebuild
                    result = vector_store.rebuild_user_vectors(user_id=user_id)
                    
                    # Check how long it took
                    elapsed_time = time.time() - start_time
                    logger.info(f"Vector rebuild took {elapsed_time:.2f} seconds for user {user_id}")
                    
                    if result.get('success'):
                        logger.info(f"Vector rebuild successful for user {user_id}: {result.get('message', '')}")
                        return jsonify({
                            'success': True,
                            'message': result.get('message', 'Vector database rebuilt successfully'),
                            'details': result,
                            'elapsed_time': elapsed_time
                        })
                    else:
                        logger.error(f"Vector rebuild failed for user {user_id}: {result.get('error', '')}")
                        return jsonify({
                            'success': False,
                            'error': result.get('error', 'Failed to rebuild vector database'),
                            'details': result,
                            'elapsed_time': elapsed_time
                        }), 500
                except Exception as rebuild_inner_error:
                    logger.error(f"Exception during vector rebuild for user {user_id}: {str(rebuild_inner_error)}")
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'success': False,
                        'error': f'Vector rebuild inner error: {str(rebuild_inner_error)}',
                        'details': {
                            'traceback': traceback.format_exc()
                        }
                    }), 500
                
        except Exception as rebuild_error:
            logger.error(f"Vector rebuild error for user {user_id}: {str(rebuild_error)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Vector rebuild error: {str(rebuild_error)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in update_database: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

def check_tweet_content_column():
    """Check if the tweet_content column exists in bookmarks table and add it if missing"""
    try:
        conn = get_db_connection()
        
        # Check if we got a SQLAlchemy connection or a psycopg2 connection
        is_sqlalchemy_conn = hasattr(conn, 'execute') and not hasattr(conn, 'cursor')
        
        if is_sqlalchemy_conn:
            # Using SQLAlchemy connection
            # Check if column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='bookmarks' AND column_name='tweet_content'
            """))
            
            if result.fetchone() is None:
                # Add tweet_content column if it doesn't exist
                logger.info("Adding tweet_content column to bookmarks table")
                conn.execute(text("""
                    ALTER TABLE bookmarks ADD COLUMN IF NOT EXISTS tweet_content TEXT
                """))
                conn.commit()
                logger.info("tweet_content column added successfully")
        else:
            # Using raw psycopg2 connection
            cursor = conn.cursor()
            
            # Check if column exists
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='bookmarks' AND column_name='tweet_content'
            """)
            
            if cursor.fetchone() is None:
                # Add tweet_content column if it doesn't exist
                logger.info("Adding tweet_content column to bookmarks table")
                cursor.execute("""
                    ALTER TABLE bookmarks ADD COLUMN IF NOT EXISTS tweet_content TEXT
                """)
                conn.commit()
                logger.info("tweet_content column added successfully")
        
    except Exception as e:
        logger.error(f"Error checking tweet_content column: {e}")
        logger.error(traceback.format_exc())

@app.route('/search')
def search():
    """Search bookmarks by query or category"""
    logger.info("Search route accessed")
    user = UserContext.get_current_user()
    
    # DEBUG: Log user information
    if user:
        logger.info(f"USER DEBUG - Search route - Authenticated user: ID={user.id}, Name={getattr(user, 'username', 'unknown')}")
    else:
        logger.info("USER DEBUG - Search route - No authenticated user")
    
    if not user:
        logger.info("User not authenticated, redirecting to login")
        return redirect(url_for('auth.login'))
    
    # Get search parameters
    query = request.args.get('q', '')
    user_query = request.args.get('user', '')
    category_filter = request.args.getlist('categories[]')
    
    logger.info(f"Search params: query='{query}', user='{user_query}', categories={category_filter}")
    
    # Get categories for the sidebar
    categories = []
    results = []
    total_results = 0
    error_message = None
    
    try:
        # Connect to database
        conn = get_db_connection()
        try:
            # Create a searcher instance
            searcher = BookmarkSearchMultiUser(conn, user.id)
            
            # Get categories for sidebar
            categories = searcher.get_categories()
            
            # Convert category names to IDs if needed
            category_ids = []
            if category_filter:
                # Find category IDs by name
                for cat_name in category_filter:
                    for cat in categories:
                        if cat['name'] == cat_name:
                            category_ids.append(cat['id'])
                            break
            
            # Perform search
            logger.info(f"Executing search with query: '{query}', categories: {category_ids}")
            results = searcher.search(
                query=query, 
                user=user_query, 
                category_ids=category_ids, 
                limit=100
            )
            logger.info(f"Search returned {len(results)} results")
            
            # FALLBACK: If no bookmarks found for this user, try with system user
            if len(results) == 0 and user.id != 1:
                logger.warning(f"No search results found for user {user.id}, falling back to system bookmarks")
                
                # Try with user_id = 1 (system user)
                system_searcher = BookmarkSearchMultiUser(conn, 1)
                results = system_searcher.search(
                    query=query, 
                    user=user_query, 
                    category_ids=category_ids, 
                    limit=100
                )
                logger.info(f"System user search returned {len(results)} results")
                
                # If still no results, try with a direct query for ANY matching bookmarks
                if len(results) == 0:
                    logger.warning("No system bookmarks found in search, trying direct query")
                    
                    # Build a direct query
                    direct_query = """
                    SELECT id, bookmark_id, text, author, created_at, author_id
                    FROM bookmarks
                    WHERE 1=1
                    """
                    
                    params = []
                    
                    if query:
                        direct_query += " AND text ILIKE %s"
                        params.append(f"%{query}%")
                    
                    if user_query:
                        direct_query += " AND author ILIKE %s"
                        params.append(f"%{user_query}%")
                        
                    direct_query += " ORDER BY created_at DESC LIMIT 100"
                    
                    if isinstance(conn, sqlalchemy.engine.Connection):
                        stmt = text(direct_query)
                        result_proxy = conn.execute(stmt, params)
                        rows = result_proxy.fetchall()
                    else:
                        cursor = conn.cursor()
                        cursor.execute(direct_query, params)
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
                        results.append(bookmark)
                        
                    logger.info(f"Direct search query found {len(results)} bookmarks")
            
            total_results = len(results)
            
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Search error: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e)
    
    # Check if user is admin
    is_admin = getattr(user, 'is_admin', False)
    
    # Render template with results
    return render_template(
        'index_final.html',
        categories=categories,
        results=results,
        query=query,
        user_query=user_query,
        category_filter=category_filter,
        showing_results=len(results),
        total_results=total_results,
        is_recent=False,
        user=user,
        is_admin=is_admin,
        db_error=bool(error_message),
        error_message=error_message
    )

@app.route('/recent')
def recent():
    """Show recent bookmarks"""
    logger.info("Recent bookmarks route accessed")
    user = UserContext.get_current_user()
    
    # DEBUG: Log user information
    if user:
        logger.info(f"USER DEBUG - Recent route - Authenticated user: ID={user.id}, Name={getattr(user, 'username', 'unknown')}")
    else:
        logger.info("USER DEBUG - Recent route - No authenticated user")
    
    if not user:
        logger.info("User not authenticated, redirecting to login")
        return redirect(url_for('auth.login'))
    
    # Get categories for the sidebar
    categories = []
    results = []
    error_message = None
    
    try:
        # Connect to database
        conn = get_db_connection()
        try:
            # Create a searcher instance
            searcher = BookmarkSearchMultiUser(conn, user.id)
            
            # Get categories for sidebar
            categories = searcher.get_categories()
            
            # Get recent bookmarks
            results = searcher.get_recent_bookmarks(limit=100)
            logger.info(f"Recent bookmarks query returned {len(results)} results")
            
            # FALLBACK: If no bookmarks found for this user, try with system user
            if len(results) == 0 and user.id != 1:
                logger.warning(f"No recent bookmarks found for user {user.id}, falling back to system bookmarks")
                
                # Try with user_id = 1 (system user)
                system_searcher = BookmarkSearchMultiUser(conn, 1)
                results = system_searcher.get_recent_bookmarks(limit=100)
                logger.info(f"System user recent bookmarks returned {len(results)} results")
                
                # If still no results, try with a direct query for ANY bookmarks
                if len(results) == 0:
                    logger.warning("No system bookmarks found, trying direct query for any bookmarks")
                    
                    # Direct query for any bookmarks
                    direct_query = """
                    SELECT id, bookmark_id, text, author, created_at, author_id
                    FROM bookmarks
                    ORDER BY created_at DESC
                    LIMIT 100
                    """
                    
                    if isinstance(conn, sqlalchemy.engine.Connection):
                        result_proxy = conn.execute(text(direct_query))
                        rows = result_proxy.fetchall()
                    else:
                        cursor = conn.cursor()
                        cursor.execute(direct_query)
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
                        results.append(bookmark)
                        
                    logger.info(f"Direct query found {len(results)} bookmarks")
            
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Recent bookmarks error: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e)
    
    # Check if user is admin
    is_admin = getattr(user, 'is_admin', False)
    
    # Render template with results
    return render_template(
        'index_final.html',
        categories=categories,
        results=results,
        query='',
        user_query='',
        category_filter=[],
        showing_results=len(results),
        total_results=len(results),
        is_recent=True,
        user=user,
        is_admin=is_admin,
        db_error=bool(error_message),
        error_message=error_message
    )

@app.route('/category/<category_name>')
def category(category_name):
    """Show bookmarks for a specific category"""
    logger.info(f"Category route accessed for: {category_name}")
    user = UserContext.get_current_user()
    
    # DEBUG: Log user information
    if user:
        logger.info(f"USER DEBUG - Category route - Authenticated user: ID={user.id}, Name={getattr(user, 'username', 'unknown')}")
    else:
        logger.info("USER DEBUG - Category route - No authenticated user")
    
    if not user:
        logger.info("User not authenticated, redirecting to login")
        return redirect(url_for('auth.login'))
    
    # Get categories for the sidebar
    categories = []
    results = []
    error_message = None
    category_id = None
    
    try:
        # Connect to database
        conn = get_db_connection()
        try:
            # Create a searcher instance
            searcher = BookmarkSearchMultiUser(conn, user.id)
            
            # Get categories for sidebar
            categories = searcher.get_categories()
            
            # Find category ID by name
            for cat in categories:
                if cat['name'] == category_name:
                    category_id = cat['id']
                    break
            
            if category_id:
                # Perform search by category
                logger.info(f"Searching bookmarks for category: {category_name} (ID: {category_id}")
                results = searcher.search(
                    query='', 
                    user='', 
                    category_ids=[category_id], 
                    limit=100
                )
                logger.info(f"Found {len(results)} bookmarks for category: {category_name}")
                
                # FALLBACK: If no bookmarks found for this user, try with system user
                if len(results) == 0 and user.id != 1:
                    logger.warning(f"No category results found for user {user.id}, falling back to system bookmarks")
                    
                    # Try with system user (user_id = 1)
                    system_searcher = BookmarkSearchMultiUser(conn, 1)
                    
                    # Try to find the category again using system user's categories
                    system_categories = system_searcher.get_categories()
                    system_category_id = None
                    
                    for cat in system_categories:
                        if cat['name'] == category_name:
                            system_category_id = cat['id']
                            break
                    
                    if system_category_id:
                        results = system_searcher.search(
                            query='', 
                            user='', 
                            category_ids=[system_category_id], 
                            limit=100
                        )
                        logger.info(f"System user category search returned {len(results)} results")
                    
                    # If still no results or category not found, try a direct query
                    if len(results) == 0:
                        logger.warning("No system category results, trying direct query by category name")
                        
                        # Direct query by category name
                        direct_query = """
                        SELECT b.id, b.bookmark_id, b.text, b.author, b.created_at, b.author_id
                        FROM bookmarks b
                        JOIN bookmark_categories bc ON b.id = bc.bookmark_id
                        JOIN categories c ON bc.category_id = c.id
                        WHERE c.name = %s
                        ORDER BY b.created_at DESC
                        LIMIT 100
                        """
                        
                        if isinstance(conn, sqlalchemy.engine.Connection):
                            stmt = text(direct_query)
                            result_proxy = conn.execute(stmt, [category_name])
                            rows = result_proxy.fetchall()
                        else:
                            cursor = conn.cursor()
                            cursor.execute(direct_query, (category_name,))
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
                            results.append(bookmark)
                            
                        logger.info(f"Direct category query found {len(results)} bookmarks")
            else:
                logger.warning(f"Category not found: {category_name}")
                
                # Try to find any bookmarks with this category name across all users
                logger.info(f"Trying to find category '{category_name}' for any user")
                
                direct_query = """
                SELECT c.id, c.user_id
                FROM categories c
                WHERE c.name = %s
                LIMIT 1
                """
                
                if isinstance(conn, sqlalchemy.engine.Connection):
                    stmt = text(direct_query)
                    result = conn.execute(stmt, [category_name])
                    row = result.fetchone()
                else:
                    cursor = conn.cursor()
                    cursor.execute(direct_query, (category_name,))
                    row = cursor.fetchone()
                
                if row:
                    any_category_id = row[0]
                    category_user_id = row[1]
                    logger.info(f"Found category '{category_name}' (ID: {any_category_id}) for user {category_user_id}")
                    
                    # Create a searcher for this user
                    alternate_searcher = BookmarkSearchMultiUser(conn, category_user_id)
                    results = alternate_searcher.search(
                        query='', 
                        user='', 
                        category_ids=[any_category_id], 
                        limit=100
                    )
                    logger.info(f"Found {len(results)} bookmarks for category '{category_name}' from user {category_user_id}")
                
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Category view error: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e)
    
    # Check if user is admin
    is_admin = getattr(user, 'is_admin', False)
    
    # Render template with results
    return render_template(
        'index_final.html',
        categories=categories,
        results=results,
        query='',
        user_query='',
        category_filter=[category_name],
        showing_results=len(results),
        total_results=len(results),
        is_recent=False,
        user=user,
        is_admin=is_admin,
        db_error=bool(error_message),
        error_message=error_message
    )

# Add a debug route for direct database inspection
@app.route('/debug/database')
def debug_database():
    """Debug route to directly check database contents"""
    user = UserContext.get_current_user()
    
    # Only allow this for authenticated users
    if not user:
        return jsonify({"error": "Authentication required"}), 401
        
    # Build debug information
    debug_info = {
        "user": {
            "id": user.id if user else None,
            "username": getattr(user, 'username', 'unknown') if user else None
        },
        "database": {}
    }
    
    try:
        # Connect to database
        conn = get_db_connection()
        try:
            # Get table counts
            tables = ['users', 'bookmarks', 'categories', 'bookmark_categories']
            for table in tables:
                cursor = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                debug_info['database'][f'{table}_count'] = cursor.scalar()
                
            # Check bookmarks for this user
            cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id = :user_id"), {"user_id": user.id})
            debug_info['database']['user_bookmarks_count'] = cursor.scalar()
            
            # Get a few sample bookmarks
            cursor = conn.execute(text("SELECT id, bookmark_id, text, author, created_at FROM bookmarks LIMIT 5"))
            samples = []
            for row in cursor:
                samples.append({
                    "id": row[0],
                    "bookmark_id": row[1],
                    "text": row[2][:50] + "..." if row[2] and len(row[2]) > 50 else row[2],
                    "author": row[3],
                    "created_at": str(row[4])
                })
            debug_info['database']['sample_bookmarks'] = samples
            
            # Get user's bookmarks sample
            cursor = conn.execute(
                text("SELECT id, bookmark_id, text, author, created_at FROM bookmarks WHERE user_id = :user_id LIMIT 5"), 
                {"user_id": user.id}
            )
            user_samples = []
            for row in cursor:
                user_samples.append({
                    "id": row[0],
                    "bookmark_id": row[1],
                    "text": row[2][:50] + "..." if row[2] and len(row[2]) > 50 else row[2],
                    "author": row[3],
                    "created_at": str(row[4])
                })
            debug_info['database']['user_sample_bookmarks'] = user_samples
            
        finally:
            conn.close()
            
        return jsonify(debug_info)
    except Exception as e:
        logger.error(f"Debug route error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/update-status', methods=['GET'])
def update_status():
    """
    Get the status of a background update job
    """
    try:
        # Get session ID from query parameters
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Missing session_id parameter'
            }), 400
            
        # Get status from database
        status_data = get_session_status(session_id)
        if not status_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session_id or job expired'
            }), 404
            
        # Get user context - with fallback methods
        user_id = None
        
        # Try different ways to get user_id
        if hasattr(g, 'user_context') and g.user_context and g.user_context.user_id:
            user_id = g.user_context.user_id
        elif 'user_id' in session:
            user_id = session['user_id']
        elif hasattr(g, 'user') and g.user and hasattr(g.user, 'id'):
            user_id = g.user.id
        elif 'user' in session and isinstance(session['user'], dict) and 'id' in session['user']:
            user_id = session['user']['id']
            
        if not user_id:
            logger.error("No user_id found in context, session, or user object")
            return jsonify({
                'success': False,
                'error': 'User not authenticated or user ID not found'
            }), 401
            
        # Security check - only allow access to your own jobs or skip for admin users
        job_user_id = str(status_data.get('user_id', ''))
        if str(user_id) != job_user_id and not is_admin_user(user_id):
            logger.warning(f"Unauthorized access attempt: User {user_id} tried to access job for user {job_user_id}")
            return jsonify({
                'success': False,
                'error': 'Unauthorized access to job status'
            }), 403
            
        # Return status data
        return jsonify({
            'success': True,
            'session_id': session_id,
            'status': status_data
        })
        
    except Exception as e:
        logger.error(f"Error in update_status: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Helper function to check if user is admin        
def is_admin_user(user_id):
    """Check if the given user_id belongs to an admin user"""
    try:
        # Implement your admin check logic here
        # For example, check against a list of admin user IDs or a database field
        admin_users = [1, 2, 3]  # Replace with your admin user IDs
        return int(user_id) in admin_users
    except Exception as e:
        logger.error(f"Error checking admin status: {str(e)}")
        return False

# Add error handlers to capture all exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    # Log the error and stacktrace
    logger.error(f"Unhandled exception: {e}")
    logger.exception(e)
    
    # Return a generic server error response
    return jsonify({
        'error': 'Internal Server Error',
        'message': str(e)
    }), 500

@app.route('/api/user-info', methods=['GET'])
@login_required
def api_user_info():
    """API endpoint to get current user information.
    This is used by the frontend to check if the user is authenticated.
    """
    try:
        # Get the current user from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                'error': 'Not authenticated',
                'authenticated': False
            }), 401

        # Get user details from database
        user = get_user_by_id(get_db_connection(), user_id)
        if not user:
            return jsonify({
                'error': 'User not found',
                'authenticated': False
            }), 404

        # Return user info (excluding sensitive data)
        return jsonify({
            'authenticated': True,
            'user_id': user_id,
            'username': user.username,
            'is_admin': is_admin_user(user_id),
            'profile_picture': user.profile_picture if hasattr(user, 'profile_picture') else None
        })
    except Exception as e:
        logger.error(f"Error in user info API: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'authenticated': False
        }), 500

# Add API endpoint for categories
@app.route('/api/categories', methods=['GET'])
@login_required
def api_categories():
    """API endpoint to get categories for the current user."""
    try:
        # Get the current user from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                'error': 'Not authenticated'
            }), 401

        # Get categories from database
        categories = get_categories_for_user(user_id)
        
        # Return categories as JSON
        return jsonify({
            'success': True,
            'categories': [
                {
                    'name': category.name,
                    'count': category.bookmark_count if hasattr(category, 'bookmark_count') else 0
                }
                for category in categories
            ]
        })
    except Exception as e:
        logger.error(f"Error in categories API: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# Add authentication check endpoint
@app.route('/check-auth', methods=['GET'])
def check_auth():
    """Simple endpoint to check if user is authenticated."""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user_id': session['user_id']
        })
    else:
        return jsonify({
            'authenticated': False
        }), 401

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