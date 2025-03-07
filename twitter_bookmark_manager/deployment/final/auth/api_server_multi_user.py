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
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

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
    UPLOAD_FOLDER=UPLOADS_DIR
)

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

# Define login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = UserContext.get_current_user()
        if user is None:
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Home page
@app.route('/')
def index():
    """Home page with multiple database connection fallbacks"""
    logger.info("Home page requested")
    user = UserContext.get_current_user()
    
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
    if all_methods_tried and not categories:
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
    
    # Return normal template if we have categories
    return render_template(
        template, 
        categories=categories, 
        user=user, 
        is_admin=is_admin,
        db_error=False
    )

# Upload bookmarks endpoint
@app.route('/upload-bookmarks', methods=['POST'])
def upload_bookmarks():
    """
    Upload bookmarks JSON file endpoint with improved error handling and duplicate detection
    """
    user = UserContext.get_current_user()
    if not user:
        return jsonify({"error": "Authentication required"}), 401
        
    # Check if a file was uploaded
    if 'bookmarks_file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['bookmarks_file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "Empty file provided"}), 400
        
    # Create a safe filename based on user ID and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    user_id_hash = hashlib.md5(str(user.id).encode()).hexdigest()[:8]
    safe_filename = f"bookmarks_{user_id_hash}_{timestamp}.json"
    
    # Ensure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(file_path)
    logger.info(f"Saved uploaded file to {file_path}")
    
    # Parse the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                bookmarks_data = json.load(f)
                
                # Check if it's a list of bookmarks or has a nested structure
                if isinstance(bookmarks_data, dict) and 'bookmarks' in bookmarks_data:
                    bookmarks_list = bookmarks_data.get('bookmarks', [])
                else:
                    bookmarks_list = bookmarks_data if isinstance(bookmarks_data, list) else []
                    
                if not bookmarks_list:
                    return jsonify({"error": "No bookmarks found in the file"}), 400
                    
                logger.info(f"Found {len(bookmarks_list)} bookmarks in the uploaded file")
                
                # Process bookmarks with robust error handling
                try:
                    # Get database connection
                    db_conn = get_db_connection()
                    
                    # Check if bookmarks already exist to avoid duplicate key errors
                    cursor = db_conn.cursor()
                    cursor.execute(
                        "SELECT bookmark_id FROM bookmarks WHERE user_id = %s",
                        (user.id,)
                    )
                    existing_bookmark_ids = {row[0] for row in cursor.fetchall()}
                    
                    # Process bookmarks - handle each bookmark individually with transaction control
                    processed = 0
                    skipped = 0
                    errors = 0
                    
                    for bookmark in bookmarks_list:
                        try:
                            # Extract bookmark ID
                            bookmark_id = None
                            # Try different fields that might contain the ID
                            for id_field in ['id_str', 'id', 'tweet_id']:
                                if id_field in bookmark:
                                    bookmark_id = str(bookmark[id_field])
                                    break
                                
                            # Skip if we couldn't find an ID or it already exists
                            if not bookmark_id:
                                logger.warning(f"Bookmark missing ID, skipping: {bookmark}")
                                errors += 1
                                continue
                                
                            if bookmark_id in existing_bookmark_ids:
                                logger.info(f"Bookmark {bookmark_id} already exists, skipping")
                                skipped += 1
                                continue
                                
                            # Extract basic fields
                            text = bookmark.get('text', bookmark.get('full_text', ''))
                            created_at = bookmark.get('created_at', '')
                            author = bookmark.get('user', {}).get('screen_name', '')
                            author_id = bookmark.get('user', {}).get('id_str') or bookmark.get('user', {}).get('id', '')
                            
                            # Use ON CONFLICT DO NOTHING to handle any potential duplicates gracefully
                            cursor.execute("""
                                INSERT INTO bookmarks (bookmark_id, user_id, text, created_at, author, author_id, processed)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (bookmark_id) DO NOTHING
                            """, (
                                bookmark_id,
                                user.id,
                                text,
                                created_at,
                                author,
                                author_id,
                                False
                            ))
                            
                            # Add to existing set to prevent future duplicates in the same batch
                            existing_bookmark_ids.add(bookmark_id)
                            processed += 1
                            
                        except Exception as bookmark_error:
                            logger.error(f"Error processing bookmark: {bookmark_error}")
                            errors += 1
                            # Continue with the next bookmark instead of failing the entire batch
                            continue
                    
                    # Commit all changes
                    db_conn.commit()
                    
                    # Store file path in session for the next step
                    session['uploaded_file'] = file_path
                    session['bookmark_count'] = processed
                    
                    return jsonify({
                        "success": True,
                        "message": f"Successfully processed {processed} bookmarks, skipped {skipped} duplicates",
                        "processed": processed,
                        "skipped": skipped,
                        "errors": errors,
                        "next_step": "/process-bookmarks"
                    })
                    
                except Exception as db_error:
                    # Rollback on error
                    if 'db_conn' in locals() and db_conn:
                        db_conn.rollback()
                        
                    logger.error(f"Database error: {db_error}")
                    return jsonify({"error": f"Database error: {str(db_error)}"}), 500
                    
            except json.JSONDecodeError as json_error:
                logger.error(f"Invalid JSON: {json_error}")
                return jsonify({"error": f"Invalid JSON file: {str(json_error)}"}), 400
                
    except Exception as e:
        logger.error(f"File processing error: {e}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/process-bookmarks', methods=['POST'])
def process_bookmarks():
    """
    Process previously uploaded bookmark file in the background.
    This is separated from upload to avoid connection timeout issues.
    """
    user = UserContext.get_current_user()
    user_id = user.id if user else None
    
    # Get session ID from request
    if request.is_json:
        data = request.json
        session_id = data.get('session_id')
    else:
        session_id = request.form.get('session_id')
        
    if not session_id:
        logger.error(f"❌ [PROCESS] No session ID provided")
        return jsonify({'error': 'No session ID provided', 'details': 'Please provide the session_id from upload'}), 400
        
    logger.info(f"🚀 [PROCESS-{session_id}] Starting bookmark processing for user {user_id}")
    
    # Find status file
    user_dir = get_user_directory(user_id)
    status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
    
    if not os.path.exists(status_file):
        logger.error(f"❌ [PROCESS-{session_id}] Status file not found: {status_file}")
        return jsonify({'error': 'Session not found', 'details': 'Upload session not found'}), 404
        
    # Read status file
    try:
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            
        # Check status
        current_status = status_data.get('status')
        if current_status == 'processing':
            logger.info(f"⏳ [PROCESS-{session_id}] Processing already in progress")
            return jsonify({
                'success': True,
                'message': 'Processing already in progress',
                'session_id': session_id,
                'status': current_status
            })
        elif current_status == 'completed':
            logger.info(f"✅ [PROCESS-{session_id}] Processing already completed")
            return jsonify({
                'success': True,
                'message': 'Processing already completed',
                'session_id': session_id,
                'status': current_status,
                'results': status_data.get('results', {})
            })
        elif current_status == 'error':
            logger.info(f"❌ [PROCESS-{session_id}] Previous processing error: {status_data.get('error')}")
            # Allow retry by continuing
            
        # Get file path
        file_path = status_data.get('file_path')
        if not file_path or not os.path.exists(file_path):
            logger.error(f"❌ [PROCESS-{session_id}] File not found: {file_path}")
            return jsonify({'error': 'File not found', 'details': 'Uploaded file not found'}), 404
            
        # Update status to processing
        status_data['status'] = 'processing'
        status_data['processing_start'] = datetime.now().isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
            
        # Start background processing
        def background_process():
            logger.info(f"🔄 [PROCESS-{session_id}] Starting background processing")
            try:
                # Process bookmarks
                result = final_update_bookmarks(
                    user_id=user_id,
                    json_file=file_path,
                    session_id=session_id,
                    status_file=status_file
                )
                
                # Update status file with results
                with open(status_file, 'r') as f:
                    current_status = json.load(f)
                    
                current_status['status'] = 'completed' if result.get('success') else 'error'
                current_status['completed_at'] = datetime.now().isoformat()
                current_status['results'] = result
                
                with open(status_file, 'w') as f:
                    json.dump(current_status, f)
                    
                logger.info(f"✅ [PROCESS-{session_id}] Background processing completed")
                
            except Exception as e:
                logger.error(f"❌ [PROCESS-{session_id}] Background processing error: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Update status file with error
                try:
                    with open(status_file, 'r') as f:
                        current_status = json.load(f)
                        
                    current_status['status'] = 'error'
                    current_status['error'] = str(e)
                    current_status['traceback'] = traceback.format_exc()
                    current_status['error_time'] = datetime.now().isoformat()
                    
                    with open(status_file, 'w') as f:
                        json.dump(current_status, f)
                except Exception as file_error:
                    logger.error(f"❌ [PROCESS-{session_id}] Error updating status file: {str(file_error)}")
        
        # Start processing in background thread
        processing_thread = threading.Thread(
            target=background_process,
            daemon=True,
            name=f"BookmarkProcessor-{session_id}"
        )
        processing_thread.start()
        
        # Return immediately with processing status
        return jsonify({
            'success': True,
            'message': 'Processing started in background',
            'session_id': session_id,
            'status': 'processing',
            'status_file': status_file,
            'check_endpoint': f"/process-status?session_id={session_id}"
        })
        
    except Exception as e:
        logger.error(f"❌ [PROCESS-{session_id}] Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

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
    """Update database endpoint based on PythonAnywhere implementation"""
    session_id = str(uuid.uuid4())[:8]
    user = UserContext.get_current_user()
    user_id = user.id if user else None
    
    logger.info(f"🚀 [UPDATE-{session_id}] Starting database update for user {user_id}")
    
    try:
        # Get parameters
        data = request.json if request.is_json else {}
        start_index = data.get('start_index', 0)
        rebuild_vector = data.get('rebuild_vector', False)
        
        # Determine the target directory and status file
        user_dir = get_user_directory(user_id)
        status_file = os.path.join(user_dir, f"update_status_{session_id}.json")
        
        # Create a new status file for tracking
        status_data = {
            'session_id': session_id,
            'user_id': user_id,
            'status': 'initializing',
            'start_time': datetime.now().isoformat(),
            'params': {
                'start_index': start_index,
                'rebuild_vector': rebuild_vector
            }
        }
        
        # Ensure directory exists
        os.makedirs(user_dir, exist_ok=True)
        
        # Write initial status
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
        
        # Find the most recent bookmark file
        bookmark_files = glob.glob(os.path.join(user_dir, "bookmarks_*.json"))
        if not bookmark_files:
            logger.error(f"❌ [UPDATE-{session_id}] No bookmark files found for user {user_id}")
            return jsonify({
                'success': False,
                'error': 'No bookmark files found',
                'message': 'Please upload a bookmarks file first'
            }), 400
            
        # Sort by modification time (newest first)
        bookmark_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = bookmark_files[0]
        logger.info(f"📂 [UPDATE-{session_id}] Using most recent bookmark file: {latest_file}")
        
        # Start processing in background
        def background_process():
            logger.info(f"🔄 [UPDATE-{session_id}] Starting background processing")
            try:
                # Process bookmarks
                result = final_update_bookmarks(
                    user_id=user_id,
                    json_file=latest_file,
                    session_id=session_id,
                    start_index=start_index,
                    rebuild_vector=rebuild_vector,
                    status_file=status_file
                )
                
                # Update status file with results
                with open(status_file, 'r') as f:
                    current_status = json.load(f)
                    
                current_status['status'] = 'completed' if result.get('success') else 'error'
                current_status['completed_at'] = datetime.now().isoformat()
                current_status['results'] = result
                
                with open(status_file, 'w') as f:
                    json.dump(current_status, f)
                    
                logger.info(f"✅ [UPDATE-{session_id}] Background processing completed")
                
            except Exception as e:
                logger.error(f"❌ [UPDATE-{session_id}] Background processing error: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Update status file with error
                try:
                    with open(status_file, 'r') as f:
                        current_status = json.load(f)
                        
                    current_status['status'] = 'error'
                    current_status['error'] = str(e)
                    current_status['traceback'] = traceback.format_exc()
                    current_status['error_time'] = datetime.now().isoformat()
                    
                    with open(status_file, 'w') as f:
                        json.dump(current_status, f)
                except Exception as file_error:
                    logger.error(f"❌ [UPDATE-{session_id}] Error updating status file: {str(file_error)}")
        
        # Start processing in background thread
        processing_thread = threading.Thread(
            target=background_process,
            daemon=True,
            name=f"UpdateProcessor-{session_id}"
        )
        processing_thread.start()
        
        # Return immediately with processing status
        return jsonify({
            'success': True,
            'message': 'Database update started in background',
            'session_id': session_id,
            'status': 'initializing',
            'status_file': status_file,
            'check_endpoint': f"/update-status?session_id={session_id}"
        })
        
    except Exception as e:
        logger.error(f"❌ [UPDATE-{session_id}] Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

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