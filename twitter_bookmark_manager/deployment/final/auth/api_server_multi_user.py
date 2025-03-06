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
import random

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
    init_database
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
        if random.random() < 0.1:  # 10% of requests
            health = check_engine_health()
            if not health['healthy']:
                logger.warning(f"Database health check failed: {health['message']}")
                # Don't fail the request, just log the issue
    except Exception as e:
        logger.error(f"Error checking database health: {e}")
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
        # Don't pass user_id again, it's already in the searcher instance
        categories = searcher.get_categories()
        
        # Check if user is admin
        is_admin = getattr(user, 'is_admin', False)
        
        return render_template(template, categories=categories, user=user, is_admin=is_admin)
    finally:
        conn.close()

# Upload bookmarks endpoint
@app.route('/upload-bookmarks', methods=['POST'])
def upload_bookmarks():
    """
    Handle bookmark JSON file upload with improved reliability.
    This endpoint ONLY handles the file upload and validation,
    without starting database processing.
    """
    session_id = str(uuid.uuid4())[:8]
    user = UserContext.get_current_user()
    user_id = user.id if user else None
    
    logger.info(f"🚀 [UPLOAD-{session_id}] Starting bookmark upload for user {user_id}")
    
    try:
        # Check if file exists in request
        if 'file' not in request.files:
            logger.error(f"❌ [UPLOAD-{session_id}] No file part in request")
            return jsonify({
                'error': 'No file part', 
                'details': 'Please select a file to upload'
            }), 400
            
        file = request.files['file']
        
        # Validate file name
        if not file.filename:
            logger.error(f"❌ [UPLOAD-{session_id}] No selected file")
            return jsonify({
                'error': 'No file selected', 
                'details': 'Please select a file to upload'
            }), 400
            
        if not file.filename.lower().endswith('.json'):
            logger.error(f"❌ [UPLOAD-{session_id}] Invalid file type: {file.filename}")
            return jsonify({
                'error': 'Invalid file type', 
                'details': 'Only JSON files are allowed'
            }), 400
            
        # Validate file content is valid JSON
        try:
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            json_data = json.loads(file_content)  # Just validate JSON syntax
            logger.info(f"✅ [UPLOAD-{session_id}] JSON validation successful")
        except json.JSONDecodeError as e:
            logger.error(f"❌ [UPLOAD-{session_id}] Invalid JSON file: {str(e)}")
            return jsonify({
                'error': 'Invalid JSON file', 
                'details': f'File is not a valid JSON: {str(e)}'
            }), 400
            
        # Create user directory if it doesn't exist
        user_dir = get_user_directory(user_id)
        if not os.path.exists(user_dir):
            try:
                os.makedirs(user_dir, exist_ok=True)
                logger.info(f"📁 [UPLOAD-{session_id}] Created user directory: {user_dir}")
            except Exception as e:
                logger.error(f"❌ [UPLOAD-{session_id}] Error creating user directory: {str(e)}")
                return jsonify({
                    'error': 'Server error', 
                    'details': 'Could not create user directory'
                }), 500
                
        # Determine file path
        file_name = f"bookmarks_{session_id}.json"
        file_path = os.path.join(user_dir, file_name)
        
        # Save the file
        try:
            file.save(file_path)
            logger.info(f"💾 [UPLOAD-{session_id}] File saved to: {file_path}")
        except Exception as e:
            logger.error(f"❌ [UPLOAD-{session_id}] Error saving file: {str(e)}")
            return jsonify({'error': 'Failed to save file', 'details': str(e)}), 500
            
        # Create status file
        status_file = os.path.join(user_dir, f"upload_status_{session_id}.json")
        status_data = {
            'session_id': session_id,
            'user_id': user_id,
            'filename': file_name,
            'file_path': file_path,
            'status': 'uploaded',
            'timestamp': datetime.now().isoformat(),
            'size_bytes': os.path.getsize(file_path)
        }
        
        try:
            with open(status_file, 'w') as f:
                json.dump(status_data, f)
            logger.info(f"📝 [UPLOAD-{session_id}] Status file created: {status_file}")
        except Exception as e:
            logger.warning(f"⚠️ [UPLOAD-{session_id}] Error creating status file: {str(e)}")
            # Non-critical error, continue
            
        # Return success with session ID for client to use in processing request
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'session_id': session_id,
            'filename': file_name,
            'size_bytes': os.path.getsize(file_path),
            'next_step': 'Call /process-bookmarks with this session_id to start processing'
        })
        
    except Exception as e:
        logger.error(f"❌ [UPLOAD-{session_id}] Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

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
    
    try:
        # Initialize the database
        init_database()
        logger.info("Database initialized successfully")
        
        # Run the app
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.exception(e) 