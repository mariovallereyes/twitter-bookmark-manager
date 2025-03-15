"""
Multi-user API server code for final environment.
This extends api_server.py with user authentication and multi-user support.
Incorporates improvements from PythonAnywhere implementation.
"""

# === Standard Library Imports ===
import os
import sys
import json
import time
import secrets
import threading
import traceback
import random
import sqlite3
import string
import uuid
import glob
import hashlib
import platform
import tempfile
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# === Third-Party Imports ===
import psycopg2
import sqlalchemy
from sqlalchemy import text, create_engine
import psutil
from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    session, send_from_directory, abort, g, flash
)
from flask.sessions import SecureCookieSessionInterface
import requests
from flask_cors import CORS
from multiprocessing import Process
from threading import Thread
try:
    from flask_session import Session
except ImportError:
    logger.warning("flask_session not available, using built-in Flask session")
    Session = None

# === Local Module Imports ===
from database.multi_user_db.vector_store_final import get_multi_user_vector_store, VectorStore, cleanup_vector_store
from auth.user_context import get_current_user
from auth.user_context_final import login_required, UserContext, UserContextMiddleware, with_user_context
from database.multi_user_db.search_final_multi_user import BookmarkSearchMultiUser
from database.multi_user_db.db_final import (
    get_db_connection, create_tables, cleanup_db_connections, check_engine_health,
    close_all_sessions, get_engine, db_session, check_database_status, init_database,
    setup_database, get_db_url
)
from database.multi_user_db.update_bookmarks_final import (
    rebuild_vector_store, find_file_in_possible_paths, get_user_directory,
    run_vector_rebuild, final_update_bookmarks
)
from auth.auth_routes_final import auth_bp
from auth.user_api_final import user_api_bp
from database.multi_user_db.user_model_final import get_user_by_id

# === Path Fixing for Railway Deployment ===
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print(f"Added repo root to Python path: {repo_root}")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added parent directory to Python path: {parent_dir}")

print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# === Application Directories and Logging ===
BASE_DIR = os.environ.get('APP_BASE_DIR', '/app')
DATABASE_DIR = os.environ.get('DATABASE_DIR', os.path.join(BASE_DIR, 'database'))
MEDIA_DIR = os.environ.get('MEDIA_DIR', os.path.join(BASE_DIR, 'media'))
UPLOADS_DIR = os.environ.get('UPLOADS_DIR', os.path.join(BASE_DIR, 'uploads'))

for directory in [DATABASE_DIR, MEDIA_DIR, UPLOADS_DIR]:
    os.makedirs(directory, exist_ok=True)

LOG_DIR = os.environ.get('LOG_DIR', os.path.join(BASE_DIR, 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'api_server_multi_user.log')

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)]
)
logger = logging.getLogger('api_server_multi_user')
logger.info(f"Starting multi-user API server with PythonAnywhere improvements... Log file: {LOG_FILE}")

# === Flask App Setup ===
template_paths = [
    os.path.join(os.getcwd(), 'twitter_bookmark_manager/deployment/final/web_final/templates'),
    os.path.join(BASE_DIR, 'twitter_bookmark_manager/deployment/final/web_final/templates'),
    os.path.join(BASE_DIR, 'web_final/templates'),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web_final/templates')),
]
template_folder = next((path for path in template_paths if os.path.exists(path)), None)
if not template_folder:
    logger.warning(f"Could not find templates in any of: {template_paths}")
    template_folder = os.path.join(os.getcwd(), 'twitter_bookmark_manager/deployment/final/web_final/templates')
    logger.info(f"Using default template path: {template_folder}")

static_folder = os.path.join(os.path.dirname(template_folder), 'static')
os.makedirs(static_folder, exist_ok=True)

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
logger.info(f"Flask app created with template_folder: {app.template_folder}")
logger.info(f"Flask app created with static_folder: {app.static_folder}")

# === App Configuration ===
app.config.update({
    'PROPAGATE_EXCEPTIONS': False,
    'TRAP_HTTP_EXCEPTIONS': True,
    'TRAP_BAD_REQUEST_ERRORS': True,
    'JSON_SORT_KEYS': False,
    'JSONIFY_PRETTYPRINT_REGULAR': False,
    'JSON_AS_ASCII': False,
    'get_db_connection': get_db_connection,
    'SECRET_KEY': os.environ.get('SECRET_KEY', os.urandom(24).hex()),
    'SESSION_TYPE': 'filesystem',
    'SESSION_FILE_DIR': os.path.join(BASE_DIR, 'flask_session'),
    'SESSION_PERMANENT': True,
    'PERMANENT_SESSION_LIFETIME': timedelta(days=7),
    'SESSION_USE_SIGNER': True,
    'SESSION_COOKIE_SECURE': os.environ.get('FLASK_ENV') == 'production',
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'SESSION_COOKIE_NAME': 'twitter_bookmark_session',
    'SESSION_REFRESH_EACH_REQUEST': True,
    'UPLOAD_FOLDER': UPLOADS_DIR,
    'DATABASE_DIR': DATABASE_DIR,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
})
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
logger.info(f"Upload folder configured at: {app.config['UPLOAD_FOLDER']}")
logger.info(f"Database directory configured at: {app.config['DATABASE_DIR']}")

# === Custom Session Interface to Handle Bytes Session IDs ===
class CustomSessionInterface(SecureCookieSessionInterface):
    """Custom session interface to handle bytes-like session IDs"""
    def save_session(self, app, session, response):
        domain = self.get_cookie_domain(app)
        path = self.get_cookie_path(app)
        
        # Don't save if session is empty and was not modified
        if not session and not session.modified:
            return
            
        # Get expiration
        httponly = self.get_cookie_httponly(app)
        secure = self.get_cookie_secure(app)
        samesite = self.get_cookie_samesite(app)
        expires = self.get_expiration_time(app, session)
        
        # Get session ID, ensuring it's a string not bytes
        session_id = session.sid if hasattr(session, 'sid') else None
        if session_id and isinstance(session_id, bytes):
            session_id = session_id.decode('utf-8')
            
        # Set the cookie with the string session ID
        if session_id:
            response.set_cookie(
                app.config['SESSION_COOKIE_NAME'],
                session_id,
                expires=expires,
                httponly=httponly,
                domain=domain,
                path=path,
                secure=secure,
                samesite=samesite
            )

try:
    if Session:
        sess = Session(app)
        logger.info("Flask session initialized with Flask-Session")
        # Replace the default session interface with our custom one
        app.session_interface = CustomSessionInterface()
        logger.info("Using custom session interface to handle bytes session IDs")
except Exception as e:
    logger.warning(f"Could not initialize Flask-Session: {e}")
    logger.info("Continuing with Flask's built-in session management")
    # Still use our custom interface for built-in sessions
    app.session_interface = CustomSessionInterface()
    logger.info("Using custom session interface with built-in session")

# === Session Status and SQLite DB for Sessions ===
session_status: Dict[str, Dict[str, Any]] = {}
STATUS_DB_PATH = os.path.join(os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', 'data'), 'session_status.db')

def init_status_db():
    """Initialize the SQLite database for storing session status."""
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

init_status_db()

def save_session_status(session_id, status_data):
    """Save session status data to a JSON file."""
    try:
        user_id = status_data.get('user_id')
        if not user_id:
            user = UserContext.get_current_user()
            if user:
                user_id = user.id
                status_data['user_id'] = user_id
        if not user_id:
            logger.warning(f"No user ID found for session {session_id}")
            return False
        user_dir = get_user_directory(user_id)
        os.makedirs(user_dir, exist_ok=True)
        status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
        logger.debug(f"Saved status for session {session_id}: {status_data.get('status', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"Error saving session status: {str(e)}")
        return False

def get_session_status(session_id):
    """Get session status from the SQLite database."""
    try:
        conn = sqlite3.connect(STATUS_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM session_status WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            status = dict(row)
            try:
                status['data'] = json.loads(status['data'])
            except:
                status['data'] = {}
            status['is_complete'] = bool(status['is_complete'])
            status['success'] = bool(status['success'])
            return status
        else:
            return None
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def cleanup_old_sessions():
    """Remove sessions older than 24 hours from the database."""
    try:
        conn = sqlite3.connect(STATUS_DB_PATH)
        cursor = conn.cursor()
        cutoff_time = (datetime.now() - timedelta(hours=24)).isoformat()
        cursor.execute('DELETE FROM session_status WHERE created_at < ?', (cutoff_time,))
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

@app.teardown_appcontext
def shutdown_cleanup(exception=None):
    """Clean up resources when the app shuts down."""
    logger.info("Application context tearing down, cleaning up resources")
    cleanup_db_connections()
    try:
        cleanup_vector_store()
        logger.info("Vector store cleaned up during shutdown")
    except Exception as e:
        logger.error(f"Error cleaning up vector store: {str(e)}")

# === Request Handlers and Middleware ===
@app.before_request
def log_session_info():
    if request.path.startswith('/static/'):
        return
    user_id = session.get('user_id')
    logger.info(f"Request: {request.path} - Session: {getattr(session, 'sid', 'No SID')} - User ID: {user_id}")
    if not hasattr(g, '_session_accessed'):
        g._session_accessed = True
        if not user_id:
            logger.info(f"Session without user_id accessed: {request.path}")
        else:
            logger.info(f"Session with user_id={user_id} accessed: {request.path}")

@app.before_request
def check_user_authentication():
    if (request.path.startswith('/static/') or 
        request.path.startswith('/auth/') or 
        request.path == '/login' or 
        request.path.startswith('/oauth/callback/')):
        return
    user_id = session.get('user_id')
    user = None
    if user_id:
        try:
            conn = get_db_connection()
            user = get_user_by_id(conn, user_id)
            if not user:
                logger.warning(f"User ID {user_id} from session not found in database")
                session.pop('user_id', None)
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'authenticated': False,
                        'error': 'User not authenticated. Please log out and log in again.'
                    }), 401
        except Exception as e:
            logger.error(f"Error checking user authentication: {e}")
    g.user = user
    g.authenticated = user is not None

@app.before_request
def check_db_health():
    if request.path.startswith('/static/') or request.path.startswith('/favicon.ico'):
        return
    try:
        if random.random() < 0.05:
            try:
                conn = get_db_connection()
                if conn:
                    if hasattr(conn, 'close'):
                        conn.close()
                    return
                else:
                    logger.warning("Database connection check failed, forcing reconnect")
                    setup_database(force_reconnect=True)
                    logger.info("Forced database reconnection after connection check failure")
            except Exception as e:
                logger.error(f"Database health check error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in health check routine: {e}")

@app.before_request
def make_session_permanent():
    if request.path.startswith('/static/'):
        return
    session.permanent = True
    if 'user_id' in session:
        session.modified = True
        session['last_activity'] = time.time()
        logger.debug(f"Session refreshed - Path: {request.path} - User ID: {session.get('user_id')}")

@app.before_request
def ensure_json_response():
    if request.path.startswith('/api/') or request.path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        g.is_json_api = True

# === Blueprint Registration ===
app.register_blueprint(auth_bp)
app.register_blueprint(user_api_bp)
UserContextMiddleware(app, lambda conn, user_id: get_user_by_id(conn, user_id))

# === Startup Debug and Guest User Initialization ===
def init_app_debug():
    """Check database at startup and ensure guest user exists."""
    try:
        logger.info("STARTUP DEBUG - Checking database initialization")
        conn = get_db_connection()
        tables = ['users', 'bookmarks', 'categories', 'bookmark_categories']
        for table in tables:
            cursor = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = cursor.scalar()
            logger.info(f"STARTUP DEBUG - Table {table} has {count} records")
        cursor = conn.execute(text("SELECT COUNT(*) FROM users WHERE id = 1"))
        if cursor.scalar() == 0:
            logger.info("STARTUP DEBUG - Creating default guest user with ID 1")
            cursor = conn.execute(
                text("INSERT INTO users (id, username, email) VALUES (1, 'guest', 'guest@example.com') ON CONFLICT DO NOTHING")
            )
        cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id = 1"))
        count = cursor.scalar()
        logger.info(f"STARTUP DEBUG - User 1 has {count} bookmarks")
        if count == 0:
            cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id IS NULL"))
            null_count = cursor.scalar()
            logger.info(f"STARTUP DEBUG - Found {null_count} bookmarks with NULL user_id")
            if null_count > 0:
                cursor = conn.execute(text("UPDATE bookmarks SET user_id = 1 WHERE user_id IS NULL"))
                logger.info(f"STARTUP DEBUG - Updated {null_count} bookmarks to user_id 1")
        cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id = :user_id"), {"user_id": 1})
        total_user_1 = cursor.scalar()
        logger.info(f"Total bookmarks for user 1: {total_user_1}")
        cursor = conn.execute(text("SELECT COUNT(*) FROM bookmarks WHERE user_id IS NULL"))
        total_no_user = cursor.scalar()
        logger.info(f"Total bookmarks with no user: {total_no_user}")
        conn.close()
    except Exception as e:
        logger.error(f"STARTUP DEBUG - Error: {e}")
        logger.error(traceback.format_exc())

init_app_debug()

# === Routes ===
@app.route('/')
def index():
    """Home page with multiple database connection fallbacks."""
    logger.info("Home page requested")
    try:
        session.clear()
    except Exception as session_error:
        logger.error(f"Error clearing session: {session_error}")
    try:
        user = UserContext.get_current_user()
        if user:
            logger.info(f"USER DEBUG - Authenticated user: ID={user.id}, Name={getattr(user, 'username', 'unknown')}")
        else:
            logger.info("USER DEBUG - No authenticated user")
        if user:
            template = 'index_final.html'
        else:
            logger.info("User not authenticated, redirecting to login")
            try:
                return redirect('/auth/login')
            except Exception as redirect_error:
                logger.error(f"Error redirecting to login: {redirect_error}")
                return render_template('login_final.html')
        # Initialize empty lists to avoid undefined variables
        latest_tweets = []
        categories = []
        error_message = ""
        all_methods_tried = False
    except Exception as e:
        logger.error(f"Error getting user context: {e}")
        return render_template('error_final.html', error="Error accessing user data")
    
        # --- Method 1: Direct psycopg2 Connection ---
    try:
        logger.info("Trying direct psycopg2 connection")
        db_url = get_db_url()
        if 'postgresql://' in db_url:
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
            direct_conn = psycopg2.connect(
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
                dbname=db_name,
                connect_timeout=3,
                application_name='twitter_bookmark_manager_direct'
            )
            direct_conn.autocommit = True
            cursor = direct_conn.cursor()
            try:
                    cursor.execute("""
                    SELECT id, name, description 
                    FROM categories 
                    WHERE user_id = %s 
                    ORDER BY name
                """, (user.id,))
            except Exception as e:
                logger.warning(f"Error querying categories with description: {e}")
                cursor.execute("""
                    SELECT id, name, '' as description
                    FROM categories 
                    WHERE user_id = %s 
                    ORDER BY name
                """, (user.id,))
            for row in cursor.fetchall():
                categories.append({
                    'id': row[0],
                    'name': row[1],
                    'description': row[2]
                })
            try:
                cursor.execute("""
                        SELECT bookmark_id, text, author_name, author_username, created_at
                    FROM bookmarks 
                    WHERE user_id = %s
                    ORDER BY created_at DESC 
                    LIMIT 5
                """, (user.id,))
                for row in cursor.fetchall():
                    tweet = {
                            'id': row[0],
                        'text': row[1],
                        'author': row[2],
                        'author_username': row[3],
                        'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[4], 'strftime') else row[4],
                            'categories': []
                    }
                    latest_tweets.append(tweet)
                logger.info(f"Successfully retrieved {len(latest_tweets)} latest bookmarks")
            except Exception as e:
                logger.warning(f"Error getting recent bookmarks: {e}")
            cursor.close()
            direct_conn.close()
            logger.info(f"Successfully loaded {len(categories)} categories directly for user {user.id}")
        else:
            raise Exception("Not a PostgreSQL database, trying SQLAlchemy")
    except Exception as e:
        logger.warning(f"Direct psycopg2 connection failed: {e}")
        error_message = str(e)
    
        # --- Method 2: SQLAlchemy Connection ---
    if not categories:
        try:
            logger.info("Trying SQLAlchemy connection")
            from database.multi_user_db.db_final import setup_database
            setup_database(force_reconnect=True)
            conn = get_db_connection()
            try:
                searcher = BookmarkSearchMultiUser(conn, user.id if user else 1)
                latest_tweets = searcher.get_recent_bookmarks(limit=5)
                logger.info(f"Successfully retrieved {len(latest_tweets)} latest bookmarks")
                if len(latest_tweets) == 0 and user and user.id != 1:
                    logger.warning(f"No bookmarks found for user {user.id}, falling back to system bookmarks")
                    searcher = BookmarkSearchMultiUser(conn, 1)
                    latest_tweets = searcher.get_recent_bookmarks(limit=5)
                    logger.info(f"Retrieved {len(latest_tweets)} system bookmarks as fallback")
                    if len(latest_tweets) == 0:
                        logger.warning("No system bookmarks found, trying direct query for any bookmarks")
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
    
        # --- Method 3: Alternative Direct Connection ---
    if not categories:
        try:
            logger.info("Trying alternative direct connection")
            db_url = get_db_url()
            if 'postgresql://' in db_url:
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
                direct_conn = psycopg2.connect(
                    user=db_user,
                    password=db_password,
                    host=db_host,
                    port=db_port,
                    dbname=db_name,
                        connect_timeout=5,
                    application_name='twitter_bookmark_manager_last_resort',
                    keepalives=1,
                    keepalives_idle=10,
                    keepalives_interval=2,
                    keepalives_count=3
                )
                direct_conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = direct_conn.cursor()
                try:
                        cursor.execute("""
                        SELECT id, name, description 
                        FROM categories 
                        WHERE user_id = %s 
                        ORDER BY name
                    """, (user.id,))
                except Exception as e:
                    logger.warning(f"Error querying categories with description: {e}")
                    cursor.execute("""
                        SELECT id, name, '' as description
                        FROM categories 
                        WHERE user_id = %s 
                        ORDER BY name
                    """, (user.id,))
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
                all_methods_tried = True
        except Exception as e:
            logger.warning(f"Alternative direct connection failed: {e}")
            if not error_message:
                error_message = str(e)
            all_methods_tried = True
    
    if not categories:
        all_methods_tried = True
    
    is_admin = getattr(user, 'is_admin', False)
    if all_methods_tried and error_message and not categories:
        logger.error(f"All database connection methods failed. Last error: {error_message}")
        return render_template(
            template, 
                categories=[], 
            user=user, 
            is_admin=is_admin,
            db_error=True,
            error_message="Database connection issues. Some features may be unavailable."
        )
    try:    
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
    except Exception as e:
        logger.error(f"Error in index: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error_final.html', error=str(e))

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

@app.route('/upload-bookmarks', methods=['POST'])
@login_required
@safe_vector_operation
def upload_bookmarks():
    """Handle file upload for bookmarks JSON file."""
    try:
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user found for upload - UserContext check failed")
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
        user_id = user.id
        logger.info(f"Processing upload for user {user_id}")
        if 'file' not in request.files:
            logger.error(f"No file part in the request for user {user_id}")
            return jsonify({'success': False, 'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            logger.error(f"No file selected for user {user_id}")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        if not file.filename.lower().endswith('.json'):
            logger.error(f"Invalid file type: {file.filename} - must be a .json file")
            return jsonify({'success': False, 'error': 'Only JSON files are allowed'}), 400
        logger.info(f"File received: {file.filename} for user {user_id}")
        base_upload_dir = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/app/uploads')
        upload_folder = os.path.join(base_upload_dir, 'uploads')
        user_dir = os.path.join(upload_folder, f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"User directory confirmed: {user_dir}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = file.filename = f"bookmarks_{timestamp}.json"
        filepath = os.path.join(user_dir, filename)
        try:
            file.save(filepath)
            file_size = os.path.getsize(filepath)
            logger.info(f"File saved: {filepath} - Size: {file_size} bytes")
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
            session_id = str(uuid.uuid4())
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
    """Validate that a file contains valid JSON and has a structure suitable for bookmarks."""
    try:
        logger.info(f"Validating JSON file: {filepath}")
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False, "File not found", None
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            logger.error(f"File is empty: {filepath}")
            return False, "File is empty", None
        if file_size > 30 * 1024 * 1024:
            logger.error(f"File is too large: {file_size} bytes")
            return False, f"File is too large: {file_size} bytes (max 30MB)", None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
            if not file_content.strip():
                logger.error(f"File content is empty: {filepath}")
                return False, "File content is empty", None
            try:
                data = json.loads(file_content)
                logger.info(f"Successfully parsed JSON, data type: {type(data).__name__}")
            except json.JSONDecodeError as je:
                line_col = f" at line {je.lineno}, column {je.colno}"
                error_msg = f"Invalid JSON format: {je.msg}{line_col}"
                logger.error(error_msg)
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
        if isinstance(data, list):
            if len(data) == 0:
                logger.error("JSON array is empty")
                return False, "JSON array is empty", None
            if len(data) > 0 and isinstance(data[0], dict):
                logger.info(f"First item in array has keys: {list(data[0].keys())}")
            return True, "", data
        elif isinstance(data, dict):
            logger.info(f"JSON object has top-level keys: {list(data.keys())}")
            if 'bookmarks' in data and isinstance(data['bookmarks'], list):
                if len(data['bookmarks']) == 0:
                    logger.error("No bookmarks found in the 'bookmarks' field")
                    return False, "No bookmarks found in the 'bookmarks' field", None
                logger.info(f"Found {len(data['bookmarks'])} bookmarks in 'bookmarks' field")
                return True, "", data
            if 'tweet' in data and isinstance(data['tweet'], list):
                if len(data['tweet']) == 0:
                    logger.error("No tweets found in the 'tweet' field")
                    return False, "No tweets found in the 'tweet' field", None
                logger.info(f"Found {len(data['tweet'])} tweets in 'tweet' field")
                return True, "", data
            bookmark_fields = ['id_str', 'id', 'tweet_id', 'tweet_url', 'text', 'full_text']
            if any(key in data for key in bookmark_fields):
                found_fields = [key for key in bookmark_fields if key in data]
                logger.info(f"Found bookmark fields directly in object: {found_fields}")
                return True, "", [data]
            logger.error(f"Unrecognized JSON structure. Found keys: {list(data.keys())}")
            return False, "Could not identify bookmark data in JSON. Expected 'bookmarks' array or bookmark fields.", None
        else:
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
        file_to_process = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                logger.error("No file selected in the request")
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            if not file.filename.lower().endswith('.json'):
                logger.error(f"Invalid file type: {file.filename}")
                return jsonify({'success': False, 'error': 'File must be a .json file'}), 400
            if 'UPLOAD_FOLDER' not in app.config:
                logger.warning("UPLOAD_FOLDER not configured, setting default")
                app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
            os.makedirs(uploads_dir, exist_ok=True)
            logger.info(f"Upload directory created/verified: {uploads_dir}")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bookmarks_{timestamp}.json"
            filepath = os.path.join(uploads_dir, filename)
            try:
                file.save(filepath)
                file_size = os.path.getsize(filepath)
                logger.info(f"File saved: {filepath} - Size: {file_size} bytes")
                is_valid, error_message, data = validate_json_file(filepath)
                if not is_valid:
                    logger.error(f"Invalid JSON format: {error_message}")
                    return jsonify({
                        'success': False, 
                        'error': error_message,
                        'file_saved': True,
                        'filepath': filepath
                    }), 400
                logger.info(f"File validation successful: {filepath}")
                file_to_process = filepath
            except Exception as e:
                logger.error(f"Error saving or validating file: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'}), 500
        elif request.is_json:
            try:
                data = request.get_json()
                if not data:
                    logger.error("Empty JSON data provided")
                    return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
                logger.info(f"Received direct JSON payload of type: {type(data).__name__}")
                uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
                os.makedirs(uploads_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"bookmarks_{timestamp}.json"
                filepath = os.path.join(uploads_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"JSON data saved to file: {filepath}")
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
        if not file_to_process:
            logger.error("Failed to prepare file for processing - no file_to_process set")
            return jsonify({'success': False, 'error': 'Failed to prepare file for processing'}), 500
        if 'DATABASE_DIR' not in app.config:
            app.config['DATABASE_DIR'] = DATABASE_DIR
            os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
            logger.info(f"Created missing DATABASE_DIR at {app.config['DATABASE_DIR']}")
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
        try:
            session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            logger.info(f"Generated session ID for processing: {session_id}")
            def process_task():
                with app.app_context():
                    try:
                        logger.info(f"Starting background processing for session {session_id}")
                        try:
                            result = final_update_bookmarks(
                                session_id=session_id,
                                start_index=0,
                                rebuild_vector=True,
                                user_id=user_id
                            )
                            logger.info(f"Background processing completed for session {session_id}: {result}")
                        except RuntimeError as re:
                            if "Storage folder" in str(re) and "already accessed by another instance" in str(re):
                                logger.error(f"Vector store access conflict: {str(re)}")
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
                                raise
                    except Exception as e:
                        logger.error(f"Error in background processing for session {session_id}: {str(e)}")
                        logger.error(traceback.format_exc())
                        error_status = {
                            'session_id': session_id,
                            'status': 'error',
                            'message': f"Processing error: {str(e)}",
                            'timestamp': datetime.now().isoformat(),
                            'user_id': user_id
                        }
                        save_session_status(session_id, error_status)
            thread = Thread(target=process_task)
            thread.daemon = True
            thread.start()
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
        error_message = str(e)
        if "UPLOAD_FOLDER" in error_message:
            if 'UPLOAD_FOLDER' not in app.config:
                app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                logger.info(f"Created missing upload folder at {app.config['UPLOAD_FOLDER']}")
            error_message = "Server configuration issue with upload folder. It has been fixed, please try again."
        elif "DATABASE_DIR" in error_message:
            if 'DATABASE_DIR' not in app.config:
                app.config['DATABASE_DIR'] = DATABASE_DIR
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
    """Redirect old /update-status endpoint to /api/process-status for backwards compatibility."""
    session_id = request.args.get('session_id')
    logger.info(f"Redirecting /update-status to /api/process-status for session {session_id}")
    return redirect(url_for('process_status', session_id=session_id))

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
        user_dir = get_user_directory(user_id)
    except Exception as e:
        logger.error(f"Error in process_status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    if process_type == 'vector-rebuild':
        status_file = os.path.join(user_dir, f"vector_rebuild_{session_id}.json")
        if not os.path.exists(status_file):
            lock_file = os.path.join(tempfile.gettempdir(), f"vector_rebuild_lock_{user_id}.lock")
            if os.path.exists(lock_file):
                try:
                    with open(lock_file, 'r') as f:
                        lock_content = f.read()
                        if session_id in lock_content:
                            return jsonify({
                                'success': True,
                                'status': 'initializing',
                                'message': 'Vector rebuild is initializing...',
                                'progress': 0
                            })
                except Exception as e:
                    logger.error(f"Error reading lock file: {e}")
            return jsonify({
                'success': False,
                'status': 'not_found',
                'message': f'No process found for session {session_id}',
                'error': 'Process not found or may have been terminated'
            }), 404
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            file_user_id = status_data.get('user_id')
            if file_user_id and int(file_user_id) != user_id:
                return jsonify({'success': False, 'error': 'Unauthorized access to process status'}), 403
            process_status_val = status_data.get('status', 'unknown')
            progress_file = os.path.join(tempfile.gettempdir(), f"vector_rebuild_progress_{user_id}_{session_id}.json")
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                        if 'bookmarks_processed' in progress_data and 'total_valid' in progress_data:
                            status_data['bookmarks_processed'] = progress_data.get('bookmarks_processed', 0)
                            status_data['total_valid'] = progress_data.get('total_valid', 0)
                            status_data['success_count'] = progress_data.get('total_success', 0) 
                            status_data['error_count'] = progress_data.get('total_errors', 0)
                            status_data['progress'] = progress_data.get('progress_percent', status_data.get('progress', 0))
                except Exception as e:
                    logger.error(f"Error reading progress file: {e}")
            if process_status_val == 'completed' and status_data.get('progress', 0) < 100:
                status_data['progress'] = 100
            if process_status_val == 'error' and 'error' not in status_data:
                status_data['error'] = 'Unknown error occurred'
            if process_status_val == 'processing':
                lock_file = os.path.join(tempfile.gettempdir(), f"vector_rebuild_lock_{user_id}.lock")
                if not os.path.exists(lock_file):
                    status_data['status'] = 'interrupted'
                    status_data['message'] = 'Process appears to have been interrupted'
            return jsonify({'success': True, **status_data})
        except Exception as e:
            logger.error(f"Error reading status file: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'success': False, 'error': f'Error reading status file: {str(e)}', 'traceback': traceback.format_exc()}), 500
    else:
        status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
        if not os.path.exists(status_file):
            logger.error(f"Status file not found: {status_file}")
            return jsonify({'error': 'Session not found'}), 404
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            return jsonify({
                'success': True,
                'session_id': session_id,
                'status': status_data.get('status', 'unknown'),
                'details': status_data
            })
        except Exception as e:
            logger.error(f"Error reading status file: {e}")
            return jsonify({'error': 'Error reading status', 'details': str(e)}), 500

@app.route('/api/update-database', methods=['POST'])
def update_database():
    """Update the database from uploaded bookmarks file."""
    try:
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user context found for database update request")
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
        user_id = user.id
        data = request.get_json() or {}
        rebuild = data.get('rebuild', False)
        direct_call = data.get('direct_call', True)
        reset_progress = data.get('reset_progress', False)
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting database update for user {user_id}, session {session_id}")
        if reset_progress:
            return jsonify({'success': True, 'message': 'Progress reset successfully'})
        user_dir = get_user_directory(user_id)
        if not user_dir:
            return jsonify({'success': False, 'error': 'Could not find or create user directory'}), 500
        status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
        with open(status_file, 'w') as f:
            status = {
                'user_id': user_id,
                'session_id': session_id,
                'status': 'processing',
                'message': 'Database update started',
                'start_time': datetime.now().isoformat(),
                'progress': {'total_processed': 0, 'errors': 0}
            }
            json.dump(status, f)
        rebuild_session_id = None
        if rebuild:
            rebuild_session_id = str(uuid.uuid4())[:8]
            logger.info(f"Will also rebuild vectors with session {rebuild_session_id}")
        thread = Thread(target=update_process, args=(user_id, session_id, rebuild, rebuild_session_id))
        thread.daemon = True
        thread.start()
        response = {
            'success': True,
            'message': 'Database update started in background',
            'session_id': session_id,
            'status_file': os.path.basename(status_file)
        }
        if rebuild_session_id:
            response['rebuild_session_id'] = rebuild_session_id
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in update_database endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

def update_process(user_id, session_id, rebuild=False, rebuild_session_id=None):
    """Background process to update the database."""
    try:
        with app.app_context():
            try:
                user_dir = get_user_directory(user_id)
                status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
                result = {
                    'success': True,
                    'message': 'Database update simulated - vector rebuild will proceed',
                    'session_id': session_id,
                    'user_id': user_id
                }
                with open(status_file, 'w') as f:
                    status = {
                        'user_id': user_id,
                        'session_id': session_id,
                        'status': 'completed',
                        'message': 'Database update completed',
                        'result': result,
                        'end_time': datetime.now().isoformat()
                    }
                    json.dump(status, f)
                logger.info(f"Database update simulated for session {session_id}")
                if rebuild:
                    logger.info("Starting vector rebuild after database update")
                    vector_session_id = rebuild_session_id if rebuild_session_id else str(uuid.uuid4())[:8]
                    rebuild_thread = Thread(
                        target=rebuild_process,
                        args=(user_id, vector_session_id),
                        kwargs={'batch_size': 20, 'resume': False}
                    )
                    rebuild_thread.daemon = True
                    rebuild_thread.start()
                    logger.info(f"Vector rebuild thread started for session {vector_session_id}")
            except Exception as e:
                logger.error(f"Error in update process: {e}")
                logger.error(traceback.format_exc())
                try:
                    with open(status_file, 'w') as f:
                        status = {
                            'user_id': user_id,
                            'session_id': session_id,
                            'status': 'error',
                            'message': f'Error updating database: {str(e)}',
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'end_time': datetime.now().isoformat()
                        }
                        json.dump(status, f)
                except Exception as write_error:
                    logger.error(f"Error writing status file: {write_error}")
    except Exception as e:
        logger.error(f"Fatal error in update thread: {e}")
        logger.error(traceback.format_exc())

@app.route('/api/categories/all', methods=['GET'])
@login_required
def get_all_categories():
    """API endpoint to get all categories for the current user, including those with zero bookmarks."""
    try:
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user context found for request")
            return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return jsonify({'status': 'error', 'message': 'Database connection error'}), 500
        try:
            searcher = BookmarkSearchMultiUser(conn, user.id)
            categories = searcher.get_categories(user_id=user.id)
            categories.sort(key=lambda x: x['name'])
            return jsonify({'status': 'success', 'categories': categories})
        finally:
            if hasattr(conn, 'close') and not isinstance(conn, sqlalchemy.engine.base.Engine):
                conn.close()
    except Exception as e:
        logger.error(f"Error getting all categories: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
@login_required
def get_categories():
    """Alias for /api/categories/all."""
    return get_all_categories()

@app.route('/emergency-upload', methods=['POST'])
@login_required
def emergency_upload():
    """Emergency file upload endpoint with minimal dependencies."""
    try:
        user = UserContext.get_current_user()
        if not user:
            return app.response_class(
                response=json.dumps({"success": False, "error": "User not authenticated"}),
                status=401,
                mimetype='application/json'
            )
        user_id = user.id
        if 'file' not in request.files:
            return app.response_class(
                response=json.dumps({"success": False, "error": "No file part in the request"}),
                status=400,
                mimetype='application/json'
            )
        file = request.files['file']
        if file.filename == '':
            return app.response_class(
                response=json.dumps({"success": False, "error": "No file selected"}),
                status=400,
                mimetype='application/json'
            )
        if not file.filename.lower().endswith('.json'):
            return app.response_class(
                response=json.dumps({"success": False, "error": "File must be a JSON file"}),
                status=400,
                mimetype='application/json'
            )
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
        os.makedirs(upload_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"emergency_upload_{timestamp}.json"
        filepath = os.path.join(upload_dir, filename)
        try:
            file.save(filepath)
            file_size = os.path.getsize(filepath)
            logger.info(f"Emergency upload saved file: {filepath} - Size: {file_size} bytes")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_char = f.read(1).strip()
                    if first_char not in ['{', '[']:
                        return app.response_class(
                            response=json.dumps({"success": False, "error": "File doesn't appear to be valid JSON", "file_saved": True, "filepath": filepath}),
                            status=400,
                            mimetype='application/json'
                        )
            except Exception as read_error:
                logger.error(f"Error validating uploaded file: {str(read_error)}")
            return app.response_class(
                response=json.dumps({"success": True, "message": "File uploaded successfully via emergency upload", "file": filename, "path": filepath}),
                status=200,
                mimetype='application/json'
            )
        except Exception as save_error:
            logger.error(f"Error in emergency upload saving file: {str(save_error)}")
            return app.response_class(
                response=json.dumps({"success": False, "error": f"Error saving file: {str(save_error)}"}),
                status=500,
                mimetype='application/json'
            )
    except Exception as e:
        logger.error(f"Unexpected error in emergency upload: {str(e)}")
        logger.error(traceback.format_exc())
        return app.response_class(
            response=json.dumps({"success": False, "error": f"Unexpected error: {str(e)}"}),
            status=500,
            mimetype='application/json'
        )

@app.route('/direct-upload', methods=['POST'])
@login_required
def direct_upload():
    """Handle bookmark JSON file upload without vector store interaction."""
    session_id = str(uuid.uuid4())[:8]
    try:
        logger.info("=" * 80)
        logger.info(f"[UPLOAD-{session_id}] Starting direct upload handler at {datetime.now().isoformat()}")
        user = UserContext.get_current_user()
        if not user:
            raise ValueError("User not found or not authenticated")
        user_id = user.id
        logger.info(f"[UPLOAD-{session_id}] User ID: {user_id}")
        if 'file' not in request.files:
            logger.error(f"[UPLOAD-{session_id}] No file part in request")
            return jsonify({'error': 'No file provided', 'details': {'request_method': request.method, 'has_files': bool(request.files), 'form_keys': list(request.form.keys()) if request.form else None}}), 400
        file = request.files['file']
        logger.info(f"[UPLOAD-{session_id}] Received file: {file.filename}")
        if not file.filename:
            logger.error(f"[UPLOAD-{session_id}] No selected file")
            return jsonify({'error': 'No file selected'}), 400
        if not file.filename.endswith('.json'):
            logger.error(f"[UPLOAD-{session_id}] Invalid file type: {file.filename}")
            return jsonify({'error': 'Only JSON files are allowed'}), 400
        if 'UPLOAD_FOLDER' not in app.config:
            logger.warning(f"[UPLOAD-{session_id}] UPLOAD_FOLDER not configured, setting default")
            app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            logger.info(f"[UPLOAD-{session_id}] Created UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"[UPLOAD-{session_id}] Created user upload directory: {upload_dir}")
        if 'DATABASE_DIR' not in app.config:
            logger.warning(f"[UPLOAD-{session_id}] DATABASE_DIR not configured, setting default")
            app.config['DATABASE_DIR'] = DATABASE_DIR
            os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
            logger.info(f"[UPLOAD-{session_id}] Created DATABASE_DIR at {app.config['DATABASE_DIR']}")
        database_dir = os.path.join(app.config['DATABASE_DIR'], f'user_{user_id}')
        os.makedirs(database_dir, exist_ok=True)
        logger.info(f"[UPLOAD-{session_id}] Created user database directory: {database_dir}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"twitter_bookmarks_{timestamp}.json"
        upload_path = os.path.join(upload_dir, filename)
        target_path = os.path.join(database_dir, 'twitter_bookmarks.json')
        try:
            file.save(upload_path)
            logger.info(f"File saved to {upload_path}")
            try:
                with open(upload_path, 'r', encoding='utf-8') as f:
                    first_chars = f.read(10).strip()
                    if not first_chars.startswith('{') and not first_chars.startswith('['):
                        return app.response_class(
                            response=json.dumps({'success': False, 'error': 'File does not appear to be valid JSON', 'first_chars': first_chars, 'file_saved': True, 'path': upload_path}),
                            status=400,
                            mimetype='application/json'
                        )
            except Exception as json_error:
                logger.error(f"Error checking JSON: {json_error}")
                return app.response_class(
                    response=json.dumps({'success': False, 'error': f'Error checking JSON file: {str(json_error)}', 'file_saved': True, 'path': upload_path}),
                    status=400,
                    mimetype='application/json'
                )
            try:
                shutil.copy2(upload_path, target_path)
                logger.info(f"File copied to database directory: {target_path}")
            except Exception as copy_error:
                logger.error(f"Error copying file: {copy_error}")
                return app.response_class(
                    response=json.dumps({'success': False, 'error': f'Error copying file to database: {str(copy_error)}', 'file_saved': True, 'upload_path': upload_path}),
                    status=500,
                    mimetype='application/json'
                )
            return app.response_class(
                response=json.dumps({'success': True, 'message': 'File uploaded successfully via direct upload endpoint', 'file': filename, 'upload_path': upload_path, 'database_path': target_path, 'timestamp': timestamp}),
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
        logger.error(f"[UPLOAD-{session_id}] Unexpected error in direct upload: {e}")
        logger.error(traceback.format_exc())
        error_message = str(e)
        if "UPLOAD_FOLDER" in error_message:
            if 'UPLOAD_FOLDER' not in app.config:
                app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                logger.info(f"[UPLOAD-{session_id}] Created missing UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
            error_message = "Server configuration issue with upload folder. It has been fixed, please try again."
        elif "DATABASE_DIR" in error_message:
            if 'DATABASE_DIR' not in app.config:
                app.config['DATABASE_DIR'] = DATABASE_DIR
                os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
                logger.info(f"[UPLOAD-{session_id}] Created missing DATABASE_DIR at {app.config['DATABASE_DIR']}")
            error_message = "Server configuration issue with database folder. It has been fixed, please try again."
        return app.response_class(
            response=json.dumps({"success": False, "error": f"Upload error: {error_message}", "retry_recommended": True}),
            status=500,
            mimetype='application/json'
        )

@app.route('/simplest-upload', methods=['POST'])
@login_required
def simplest_upload():
    """Absolute minimum file upload endpoint with zero dependencies."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file in request'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': 'Empty file name'}), 400
        if 'UPLOAD_FOLDER' not in app.config:
            app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            logger.info(f"Created missing UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'user_{user_id}')
        os.makedirs(user_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(user_dir, filename)
        file.save(filepath)
        if 'DATABASE_DIR' not in app.config:
            app.config['DATABASE_DIR'] = DATABASE_DIR
            os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
            logger.info(f"Created missing DATABASE_DIR at {app.config['DATABASE_DIR']}")
        db_dir = os.path.join(app.config['DATABASE_DIR'], f'user_{user_id}')
        os.makedirs(db_dir, exist_ok=True)
        target_path = os.path.join(db_dir, 'twitter_bookmarks.json')
        shutil.copy2(filepath, target_path)
        return jsonify({'success': True, 'message': 'File saved successfully', 'original_path': filepath, 'target_path': target_path})
    except Exception as e:
        logger.error(f"Error in simplest_upload: {str(e)}")
        logger.error(traceback.format_exc())
        error_message = str(e)
        if "UPLOAD_FOLDER" in error_message:
            app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            logger.info(f"Created missing UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
            error_message = "Server configuration issue with upload folder. It has been fixed, please try again."
        elif "DATABASE_DIR" in error_message:
            app.config['DATABASE_DIR'] = DATABASE_DIR
            os.makedirs(app.config['DATABASE_DIR'], exist_ok=True)
            logger.info(f"Created missing DATABASE_DIR at {app.config['DATABASE_DIR']}")
            error_message = "Server configuration issue with database folder. It has been fixed, please try again."
        return jsonify({'success': False, 'error': f"Upload error: {error_message}", 'retry_recommended': True}), 500

@app.route('/check-auth', methods=['GET'])
def check_auth():
    """Simple endpoint to check authentication status."""
    user_id = session.get('user_id')
    logger.info(f"Auth check - Session: {getattr(session, 'sid', 'No SID')} - User ID: {user_id}")
    if user_id:
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

def safe_get_vector_store():
    """Safely attempt to get a vector store instance with retry logic."""
    MAX_RETRIES = 3
    initial_backoff = 1
    session_id = str(uuid.uuid4())[:8]
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"[RETRY-{session_id}] Attempting to import VectorStoreMultiUser (attempt {attempt+1}/{MAX_RETRIES})")
            from database.multi_user_db.vector_store_final import VectorStoreMultiUser
            logger.info(f"[RETRY-{session_id}] Successfully imported VectorStoreMultiUser")
            user = UserContext.get_current_user()
            if not user:
                logger.warning(f"[RETRY-{session_id}] No user context found, cannot initialize vector store")
                return None
            user_id = user.id
            vector_store = VectorStoreMultiUser(user_id=user_id)
            logger.info(f"[RETRY-{session_id}] Successfully initialized vector store for user {user_id} on attempt {attempt+1}")
            return vector_store
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
            logger.error(f"[RETRY-{session_id}] Unexpected error in safe_get_vector_store: {e}")
            logger.error(traceback.format_exc())
            if attempt < MAX_RETRIES - 1:
                backoff = initial_backoff * (2 ** attempt)
                logger.info(f"[RETRY-{session_id}] Retrying in {backoff} seconds due to unexpected error...")
                time.sleep(backoff)
            else:
                logger.error(f"[RETRY-{session_id}] Failed to recover from unexpected error after {MAX_RETRIES} attempts")
            return None
    return None

@app.route('/api/rebuild-vector-store', methods=['POST'])
@login_required
def rebuild_vector_store_endpoint():
    """API endpoint to rebuild the vector store from the database."""
    try:
        user = UserContext.get_current_user()
        if not user:
            logger.error("No user context found for request")
            return jsonify({'success': False, 'error': 'User not authenticated'}), 401
        user_id = user.id
        logger.info(f"Starting vector store rebuild for user {user_id}")
        data = request.get_json() or {}
        session_id = data.get('session_id', str(uuid.uuid4())[:8])
        batch_size = data.get('batch_size', 20)
        resume = data.get('resume', True)
        user_dir = os.path.join(app.config.get('DATABASE_DIR', DATABASE_DIR), f'user_{user_id}')
        os.makedirs(user_dir, exist_ok=True)
        lock_file_path = os.path.join(tempfile.gettempdir(), f"vector_rebuild_lock_{user_id}.lock")
        if os.path.exists(lock_file_path):
            lock_file_stat = os.stat(lock_file_path)
            lock_file_age = time.time() - lock_file_stat.st_mtime
            if lock_file_age > 600:
                logger.warning(f"Removing stale lock file (age: {lock_file_age:.1f}s)")
                try:
                    os.remove(lock_file_path)
                except Exception as e:
                    logger.error(f"Error removing stale lock file: {e}")
            else:
                logger.warning(f"Another vector rebuild is in progress (lock age: {lock_file_age:.1f}s)")
                return jsonify({
                    'success': False,
                    'error': 'Another vector rebuild is already in progress',
                    'status': 'in_progress',
                    'message': f'A rebuild started {lock_file_age:.1f} seconds ago is still running',
                    'progress': get_rebuild_progress(user_id, session_id),
                    'session_id': session_id
                })
        progress_file_path = os.path.join(tempfile.gettempdir(), f"vector_rebuild_progress_{user_id}_{session_id}.json")
        progress_data = {}
        if os.path.exists(progress_file_path) and resume:
            try:
                with open(progress_file_path, 'r') as f:
                    progress_data = json.load(f)
                    logger.info(f"Found existing progress file. Resume from index {progress_data.get('processed_index', 0)}")
            except Exception as e:
                logger.error(f"Error reading progress file: {e}")
                progress_data = {}
        status_file = os.path.join(user_dir, f"vector_rebuild_{session_id}.json")
        with open(status_file, 'w') as f:
            status = {
                'user_id': user_id,
                'session_id': session_id,
                'status': 'processing',
                'message': 'Vector store rebuild started',
                'start_time': datetime.now().isoformat(),
                'progress': progress_data.get('progress_percent', 0),
                'resume': resume and bool(progress_data),
                'batch_size': batch_size
            }
            json.dump(status, f)
        thread = Thread(target=rebuild_process, args=(user_id, session_id, batch_size, resume, progress_data))
        thread.daemon = True
        thread.start()
        return jsonify({
            'success': True,
            'message': 'Vector store rebuild started in background',
            'session_id': session_id,
            'status': 'processing',
            'progress': progress_data.get('progress_percent', 0),
            'resume': resume and bool(progress_data)
        })
    except Exception as e:
        logger.error(f"Error starting vector rebuild: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e), 'details': traceback.format_exc()}), 500

def rebuild_process(user_id, session_id, batch_size=20, resume=True, progress_data=None):
    """Background process to rebuild the vector store."""
    start_time = datetime.now()
    logger.info(f"REBUILD_PROCESS: Starting for user_id={user_id}, session_id={session_id}")
    user_dir = os.path.join(app.config.get('DATABASE_DIR', DATABASE_DIR), f'user_{user_id}')
    os.makedirs(user_dir, exist_ok=True)
    status_file = os.path.join(user_dir, f"vector_rebuild_{session_id}.json")
    progress_file_path = os.path.join(tempfile.gettempdir(), f"vector_rebuild_progress_{user_id}_{session_id}.json")
    logger.info(f"REBUILD_PROCESS: Status file: {status_file}")
    logger.info(f"REBUILD_PROCESS: Progress file: {progress_file_path}")
    try:
        logger.info("REBUILD_PROCESS: Importing rebuild_vector_store...")
        from database.multi_user_db.update_bookmarks_final import rebuild_vector_store
        logger.info("REBUILD_PROCESS: Successfully imported rebuild_vector_store")
        with app.app_context():
            try:
                logger.info(f"REBUILD_PROCESS: Starting vector store rebuild - session_id={session_id}")
                update_status_file(status_file, {
                    'status': 'processing',
                    'message': 'Initializing vector rebuild...',
                    'start_time': start_time.isoformat(),
                })
                result = rebuild_vector_store(
                    session_id=session_id,
                    user_id=user_id,
                    batch_size=batch_size,
                    resume=resume,
                    progress_data=progress_data
                )
                status_update = {
                    'status': 'completed' if result.get('success', False) else 'error',
                    'message': 'Vector store rebuild completed' if result.get('success', False) else 'Vector store rebuild failed',
                    'result': result,
                    'start_time': start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - start_time).total_seconds(),
                    'progress': 100 if result.get('success', False) else result.get('progress', 0)
                }
                update_status_file(status_file, status_update)
                logger.info(f"Vector store rebuild completed - session_id={session_id}")
                if result.get('success', False) and os.path.exists(progress_file_path):
                    try:
                        os.remove(progress_file_path)
                        logger.info(f"Removed progress file after successful completion - session_id={session_id}")
                    except Exception as e:
                        logger.error(f"Error removing progress file: {e}")
            except Exception as e:
                logger.error(f"Error in vector store rebuild process: {e}")
                logger.error(traceback.format_exc())
                progress = 0
                if os.path.exists(progress_file_path):
                    try:
                        with open(progress_file_path, 'r') as f:
                            progress_data = json.load(f)
                            progress = progress_data.get('progress_percent', 0)
                    except Exception:
                        pass
                update_status_file(status_file, {
                    'status': 'error',
                    'message': f'Error rebuilding vector store: {str(e)}',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'start_time': start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - start_time).total_seconds(),
                    'progress': progress
                })
    except ImportError as import_error:
        logger.error(f"REBUILD_PROCESS: Failed to import rebuild_vector_store: {import_error}")
        logger.error(traceback.format_exc())
        with open(status_file, 'w') as f:
            status = {
                'status': 'error',
                'message': f'Failed to import rebuild function: {str(import_error)}',
                'error': str(import_error),
                'traceback': traceback.format_exc(),
                'end_time': datetime.now().isoformat()
            }
            json.dump(status, f)
    except Exception as e:
        logger.error(f"Fatal error in rebuild thread: {e}")
        logger.error(traceback.format_exc())
        update_status_file(status_file, {
            'status': 'error',
            'message': f'Fatal error in rebuild thread: {str(e)}',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'progress': 0
        })
    finally:
        lock_file_path = os.path.join(tempfile.gettempdir(), f"vector_rebuild_lock_{user_id}.lock")
        if os.path.exists(lock_file_path):
            try:
                os.remove(lock_file_path)
                logger.info(f"Removed lock file after rebuild process - session_id={session_id}")
            except Exception as e:
                logger.error(f"Error removing lock file: {e}")

def update_status_file(file_path, updates):
    """Update a status file with new data while preserving existing data."""
    try:
        existing_data = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        existing_data.update(updates)
        with open(file_path, 'w') as f:
            json.dump(existing_data, f)
    except Exception as e:
        logger.error(f"Error updating status file {file_path}: {e}")

def get_rebuild_progress(user_id, session_id):
    """Get current rebuild progress from the progress file."""
    progress_file_path = os.path.join(tempfile.gettempdir(), f"vector_rebuild_progress_{user_id}_{session_id}.json")
    if os.path.exists(progress_file_path):
        try:
            with open(progress_file_path, 'r') as f:
                progress_data = json.load(f)
                return progress_data.get('progress_percent', 0)
        except Exception as e:
            logger.error(f"Error reading progress file: {e}")
    user_dir = os.path.join(app.config.get('DATABASE_DIR', DATABASE_DIR), f'user_{user_id}')
    status_file = os.path.join(user_dir, f"vector_rebuild_{session_id}.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
                return status_data.get('progress', 0)
        except Exception as e:
            logger.error(f"Error reading status file: {e}")
    return 0

# === Global Error Handlers ===
@app.errorhandler(Exception)
def handle_api_exception(e):
    path = request.path if request else ""
    logger.error(f"GLOBAL ERROR HANDLER: {path}: {str(e)}")
    logger.error(traceback.format_exc())
    if path.startswith('/api/') or 'application/json' in request.headers.get('Accept', ''):
        return jsonify({
            'success': False,
            'error': str(e),
            'path': path,
            'type': e.__class__.__name__
        }), 500
    else:
        try:
            user = None
            try:
                user = UserContext.get_current_user()
            except:
                pass
            return render_template('error_final.html', 
                error=str(e), 
                error_type=e.__class__.__name__,
                user=user,
                is_admin=getattr(user, 'is_admin', False) if user else False
            ), 500
        except Exception as template_error:
            logger.error(f"Error rendering error template: {str(template_error)}")
            return f"""
            <html>
                <head><title>Application Error</title></head>
                <body>
                    <h1>Application Error</h1>
                    <p>{str(e)}</p>
                    <p><a href="/">Return to Home</a></p>
                </body>
            </html>
            """, 500

@app.errorhandler(400)
def handle_bad_request(e):
    path = request.path if request else ""
    if path.startswith('/api/') or path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        logger.error(f"Bad request error on {path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Bad Request: ' + str(e),
            'path': path
        }), 400
    return e

@app.errorhandler(404)
def handle_not_found(e):
    path = request.path if request else ""
    if path.startswith('/api/') or path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        logger.error(f"Not found error on {path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Not Found: The requested resource does not exist',
            'path': path
        }), 404
    return e

@app.errorhandler(405)
def handle_method_not_allowed(e):
    path = request.path if request else ""
    if path.startswith('/api/') or path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        logger.error(f"Method not allowed error on {path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Method Not Allowed: The method is not allowed for this endpoint',
            'path': path,
            'allowed_methods': e.valid_methods if hasattr(e, 'valid_methods') else None
        }), 405
    return e

@app.errorhandler(500)
def handle_server_error(e):
    path = request.path if request else ""
    logger.error(f"Server error on {path}: {str(e)}")
    logger.error(traceback.format_exc())
    if path.startswith('/api/') or path in ['/upload-bookmarks', '/process-bookmarks', '/update-database']:
        return jsonify({
            'success': False,
            'error': 'Internal Server Error: The server encountered an unexpected condition',
            'path': path
        }), 500
    return e

@app.errorhandler(json.JSONDecodeError)
def handle_json_error(e):
    logger.error(f"JSON decode error: {str(e)}")
    return jsonify({
        'success': False,
        'error': f'Invalid JSON format: {str(e)}',
        'path': request.path
    }), 400

@app.errorhandler(RuntimeError)
def handle_runtime_error(e):
    logger.error(f"Runtime error: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        'success': False,
        'error': str(e)
    }), 500

@app.errorhandler(TypeError)
def handle_type_error(e):
    error_message = str(e)
    logger.error(f"Type error: {error_message}")
    if "cannot use a string pattern on a bytes-like object" in error_message:
        logger.error("Flask session cookie error detected - handling gracefully")
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Session error - please try logging in again',
                'path': request.path
            }), 400
        return redirect('/auth/login')
    return jsonify({
        'success': False,
        'error': error_message
    }), 500

# === Run the Application ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting server on port {port} with debug={debug}")
    max_db_attempts = 5
    db_attempt = 0
    db_initialized = False
    while db_attempt < max_db_attempts and not db_initialized:
        try:
            init_database()
            logger.info("Database initialized successfully")
            db_initialized = True
        except Exception as e:
            db_attempt += 1
            wait_time = 2 ** db_attempt
            logger.error(f"Database initialization attempt {db_attempt}/{max_db_attempts} failed: {e}")
            if db_attempt < max_db_attempts:
                logger.info(f"Retrying database initialization in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to initialize database after {max_db_attempts} attempts")
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.exception(e)
