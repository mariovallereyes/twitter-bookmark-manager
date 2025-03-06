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
import threading
from datetime import datetime, timedelta
from pathlib import Path
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

# Import user context features
from auth.user_context_final import UserContext, with_user_context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('api_server_multi_user')

# Import user authentication components
from auth.auth_routes_final import auth_bp
from auth.user_api_final import user_api_bp
from auth.user_context_final import UserContextMiddleware
from database.multi_user_db.user_model_final import get_user_by_id

# Import database modules
from database.multi_user_db.db_final import get_db_connection, create_tables
from database.multi_user_db.search_final_multi_user import BookmarkSearchMultiUser
from database.multi_user_db.update_bookmarks_final import (
    final_update_bookmarks,
    rebuild_vector_store
)
from database.multi_user_db.vector_store_final import VectorStore

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
        # Don't pass user_id again, it's already in the searcher instance
        categories = searcher.get_categories()
        
        # Check if user is admin
        is_admin = getattr(user, 'is_admin', False)
        
        return render_template(template, categories=categories, user=user, is_admin=is_admin)
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
    
    # Get admin status
    is_admin = getattr(user, 'is_admin', False)
    
    # Get connection and search
    conn = get_db_connection()
    try:
        searcher = BookmarkSearchMultiUser(conn, user.id)
        
        # Convert category IDs to integers
        category_ids = [int(c) for c in categories if c.isdigit()]
        
        # Perform search
        # Don't pass user_id again, it's already in the searcher instance
        results = searcher.search(
            query=query, 
            user=author, 
            category_ids=category_ids
        )
        
        # Get all categories for display
        # Don't pass user_id again, it's already in the searcher instance
        all_categories = searcher.get_categories()
        
        return render_template(
            'search_results_final.html',
            query=query,
            author=author,
            results=results,
            categories=all_categories,
            selected_categories=category_ids,
            user=user,
            is_admin=is_admin
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
    
    # Get admin status
    is_admin = getattr(user, 'is_admin', False)
    
    # Get connection and fetch recent bookmarks
    conn = get_db_connection()
    try:
        searcher = BookmarkSearchMultiUser(conn, user.id)
        # Don't pass user_id again, it's already in the searcher instance
        results = searcher.get_recent()
        
        # Get all categories for display
        # Don't pass user_id again, it's already in the searcher instance
        categories = searcher.get_categories()
        
        return render_template(
            'recent_final.html',
            results=results,
            categories=categories,
            user=user,
            is_admin=is_admin
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
    
    # Get admin status
    is_admin = getattr(user, 'is_admin', False)
    
    # Get connection and fetch categories
    conn = get_db_connection()
    try:
        searcher = BookmarkSearchMultiUser(conn, user.id)
        # Don't pass user_id again, it's already in the searcher instance
        categories = searcher.get_categories()
        
        return render_template(
            'categories_final.html',
            categories=categories,
            user=user,
            is_admin=is_admin
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
        # Don't pass user_id again, it's already in the searcher instance
        categories = searcher.get_categories()
        
        return jsonify(categories)
    finally:
        conn.close()

# Upload bookmarks endpoint
@app.route('/upload-bookmarks', methods=['POST'])
def upload_bookmarks():
    """
    Improved endpoint for uploading bookmarks JSON file.
    """
    try:
        user_id = get_user_id()
        logger.info(f"Upload bookmarks request received for user {user_id}")
        
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.endswith('.json'):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only JSON files are allowed'}), 400
            
        # Create user directory if it doesn't exist
        user_dir = f"user_{user_id}"
        
        # Try multiple potential paths for consistency
        potential_dirs = [
            os.path.join(BASE_DIR, "database", user_dir),
            os.path.join("database", user_dir),
            os.path.join("/app/database", user_dir)
        ]
        
        # Find or create the first valid directory
        database_dir = None
        for dir_path in potential_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                if os.path.exists(dir_path):
                    database_dir = dir_path
                    break
            except Exception as e:
                logger.warning(f"Could not create directory {dir_path}: {e}")
        
        if not database_dir:
            logger.error("Could not create any database directory")
            return jsonify({'error': 'Server error creating user directory'}), 500
        
        # Set paths for backup and file storage
        bookmarks_file = os.path.join(database_dir, 'twitter_bookmarks.json')
        backup_file = os.path.join(database_dir, f'twitter_bookmarks_backup_{int(time.time())}.json')
        
        # Backup existing file if it exists
        if os.path.exists(bookmarks_file):
            try:
                shutil.copy2(bookmarks_file, backup_file)
                logger.info(f"Created backup of existing bookmarks at {backup_file}")
            except Exception as e:
                logger.warning(f"Could not create backup: {e}")
        
        # Save the uploaded file
        file.save(bookmarks_file)
        file_size = os.path.getsize(bookmarks_file)
        logger.info(f"Saved uploaded file ({file_size} bytes) to {bookmarks_file}")
        
        # Validate JSON format
        try:
            with open(bookmarks_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            # Check if it's a valid bookmarks structure (either array or object with bookmarks key)
            if isinstance(json_data, list):
                bookmark_count = len(json_data)
            elif isinstance(json_data, dict) and 'bookmarks' in json_data:
                bookmark_count = len(json_data['bookmarks'])
            else:
                logger.error("Invalid JSON format - not a bookmarks array or object")
                return jsonify({'error': 'Invalid bookmarks format'}), 400
                
            logger.info(f"Validated JSON with {bookmark_count} bookmarks")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            return jsonify({
                'error': 'Invalid JSON file', 
                'details': str(e)
            }), 400
        
        # Reset progress file to ensure fresh start
        progress_file = os.path.join(database_dir, 'update_progress.json')
        if os.path.exists(progress_file):
            try:
                # Create a backup of the progress file
                progress_backup = os.path.join(database_dir, f'update_progress_backup_{int(time.time())}.json')
                shutil.copy2(progress_file, progress_backup)
                # Remove the original progress file
                os.remove(progress_file)
                logger.info("Reset progress file for fresh start")
            except Exception as e:
                logger.warning(f"Could not reset progress file: {e}")
        
        # Return success with file details
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'filename': file.filename,
            'bookmarks_count': bookmark_count,
            'file_path': bookmarks_file,
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Update database endpoint
@app.route('/update-database', methods=['POST', 'GET'])
def update_database():
    """
    Improved endpoint for updating the database with bookmarks.
    Supports sync processing for small databases and async for larger ones.
    """
    try:
        user_id = get_user_id()
        logger.info(f"Update database request received for user {user_id}")
        
        # Get parameters
        if request.method == 'POST':
            data = request.get_json() or {}
            start_index = data.get('start_index', 0)
            session_id = data.get('session_id')
            rebuild_vector = data.get('rebuild_vector', False)
            reset_progress = data.get('reset_progress', False)
        else:  # GET
            start_index = int(request.args.get('start_index', 0))
            session_id = request.args.get('session_id')
            rebuild_vector = request.args.get('rebuild_vector', '').lower() in ('true', 't', '1')
            reset_progress = request.args.get('reset_progress', '').lower() in ('true', 't', '1')
        
        # Generate a session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
            
        logger.info(f"Update parameters: start_index={start_index}, rebuild_vector={rebuild_vector}, reset_progress={reset_progress}")
        
        # If reset_progress is requested, delete the progress file
        if reset_progress:
            user_dir = f"user_{user_id}"
            potential_progress_files = [
                os.path.join(BASE_DIR, "database", user_dir, 'update_progress.json'),
                os.path.join("database", user_dir, 'update_progress.json'),
                os.path.join("/app/database", user_dir, 'update_progress.json')
            ]
            
            # Try to remove all possible progress files
            for progress_file in potential_progress_files:
                if os.path.exists(progress_file):
                    try:
                        # Backup before deleting
                        backup = f"{progress_file}.bak_{int(time.time())}"
                        shutil.copy2(progress_file, backup)
                        os.remove(progress_file)
                        logger.info(f"Reset progress file: {progress_file}")
                    except Exception as e:
                        logger.warning(f"Could not reset progress file {progress_file}: {e}")
            
            # Reset start_index
            start_index = 0
        
        # If start_index is 0, this is a new update, so use a new session ID
        if start_index == 0:
            session_id = str(uuid.uuid4())[:8]
            logger.info(f"Starting new update session: {session_id}")
        
        # Get total bookmarks count for progress tracking
        user_dir = f"user_{user_id}"
        potential_bookmark_files = [
            os.path.join(BASE_DIR, "database", user_dir, 'twitter_bookmarks.json'),
            os.path.join("database", user_dir, 'twitter_bookmarks.json'),
            os.path.join("/app/database", user_dir, 'twitter_bookmarks.json')
        ]
        
        total_bookmarks = 0
        for bookmarks_file in potential_bookmark_files:
            if os.path.exists(bookmarks_file):
                try:
                    with open(bookmarks_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        total_bookmarks = len(data)
                    elif isinstance(data, dict) and 'bookmarks' in data:
                        total_bookmarks = len(data['bookmarks'])
                        
                    if total_bookmarks > 0:
                        break
                except Exception as e:
                    logger.warning(f"Could not determine bookmark count from {bookmarks_file}: {e}")
        
        # Import the update_bookmarks function from the right location
        from twitter_bookmark_manager.deployment.final.database.multi_user_db.update_bookmarks_final import final_update_bookmarks
        
        # Process update
        result = final_update_bookmarks(
            session_id=session_id,
            start_index=start_index,
            rebuild_vector=rebuild_vector,
            user_id=user_id
        )
        
        # If successful, add total bookmarks count for progress calculation
        if result.get('success'):
            result['total_bookmarks'] = total_bookmarks
            if total_bookmarks > 0:
                result['percent_complete'] = min(100, (result.get('processed_this_session', 0) / total_bookmarks) * 100)
            else:
                result['percent_complete'] = 100
        
        # Return result
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Async update database endpoint - resistant to connection timeouts
@app.route('/async-update-database', methods=['POST'])
def async_update_database():
    """
    New endpoint for asynchronous database update using background threads.
    This prevents timeouts during long-running updates.
    """
    try:
        user_id = get_user_id()
        logger.info(f"Async update database request received for user {user_id}")
        
        # Get parameters from JSON or form data
        data = request.get_json() or {}
        start_index = data.get('start_index', 0)
        rebuild_vector = data.get('rebuild_vector', False)
        reset_progress = data.get('reset_progress', False)
        
        # Generate a session ID
        session_id = str(uuid.uuid4())[:8]
        
        # If reset_progress is requested, delete the progress file
        if reset_progress:
            user_dir = f"user_{user_id}"
            potential_progress_files = [
                os.path.join(BASE_DIR, "database", user_dir, 'update_progress.json'),
                os.path.join("database", user_dir, 'update_progress.json'),
                os.path.join("/app/database", user_dir, 'update_progress.json')
            ]
            
            # Try to remove all possible progress files
            for progress_file in potential_progress_files:
                if os.path.exists(progress_file):
                    try:
                        # Backup before deleting
                        backup = f"{progress_file}.bak_{int(time.time())}"
                        shutil.copy2(progress_file, backup)
                        os.remove(progress_file)
                        logger.info(f"Reset progress file: {progress_file}")
                    except Exception as e:
                        logger.warning(f"Could not reset progress file {progress_file}: {e}")
            
            # Reset start_index
            start_index = 0
            
        # Import the update_bookmarks function
        from twitter_bookmark_manager.deployment.final.database.multi_user_db.update_bookmarks_final import final_update_bookmarks
            
        # Create a thread to run the update in the background
        def background_update():
            try:
                logger.info(f"Starting background update for session {session_id}")
                final_update_bookmarks(
                    session_id=session_id,
                    start_index=start_index,
                    rebuild_vector=rebuild_vector,
                    user_id=user_id
                )
                logger.info(f"Background update completed for session {session_id}")
            except Exception as e:
                logger.error(f"Error in background update: {e}")
                logger.error(traceback.format_exc())
        
        # Start the background thread
        thread = threading.Thread(target=background_update)
        thread.daemon = True
        thread.start()
        
        # Create initial status file
        user_dir = f"user_{user_id}"
        
        # Find a valid database directory
        database_dir = None
        potential_dirs = [
            os.path.join(BASE_DIR, "database", user_dir),
            os.path.join("database", user_dir),
            os.path.join("/app/database", user_dir)
        ]
        
        for dir_path in potential_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                database_dir = dir_path
                break
                
        if not database_dir:
            # Try to create the first directory
            os.makedirs(potential_dirs[0], exist_ok=True)
            database_dir = potential_dirs[0]
            
        # Create status file for polling
        status_file = os.path.join(database_dir, f'update_status_{session_id}.json')
        status_data = {
            'session_id': session_id,
            'status': 'processing',
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'user_id': user_id,
            'start_index': start_index,
            'rebuild_vector': rebuild_vector,
            'progress': {
                'total_processed': 0,
                'new_count': 0,
                'updated_count': 0,
                'errors': 0
            }
        }
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
            
        # Return session ID for polling
        return jsonify({
            'success': True,
            'message': 'Background update started',
            'session_id': session_id,
            'status_file': status_file
        })
        
    except Exception as e:
        logger.error(f"Error starting async update: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Update status endpoint to check progress
@app.route('/update-status', methods=['GET'])
def update_status():
    """
    Check the status of an async database update.
    """
    try:
        user_id = get_user_id()
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
            
        logger.info(f"Checking status for session {session_id}, user {user_id}")
        
        # Find status file
        user_dir = f"user_{user_id}"
        potential_status_files = [
            os.path.join(BASE_DIR, "database", user_dir, f'update_status_{session_id}.json'),
            os.path.join("database", user_dir, f'update_status_{session_id}.json'),
            os.path.join("/app/database", user_dir, f'update_status_{session_id}.json')
        ]
        
        status_file = None
        for file_path in potential_status_files:
            if os.path.exists(file_path):
                status_file = file_path
                break
                
        if not status_file:
            # Check progress file instead
            potential_progress_files = [
                os.path.join(BASE_DIR, "database", user_dir, 'update_progress.json'),
                os.path.join("database", user_dir, 'update_progress.json'),
                os.path.join("/app/database", user_dir, 'update_progress.json')
            ]
            
            for file_path in potential_progress_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            progress_data = json.load(f)
                            
                        # Check if this is the right session
                        if progress_data.get('session_id') == session_id:
                            # Convert progress to status format
                            return jsonify({
                                'status': 'processing',
                                'session_id': session_id,
                                'progress': progress_data.get('stats', {}),
                                'last_processed_index': progress_data.get('last_processed_index', 0)
                            })
                    except Exception as e:
                        logger.warning(f"Error reading progress file {file_path}: {e}")
            
            # No status file found
            return jsonify({
                'status': 'unknown',
                'session_id': session_id,
                'error': 'Status file not found'
            }), 404
        
        # Read status file
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            
        # Check if process is still running (if status file hasn't been updated recently)
        last_update = datetime.fromisoformat(status_data.get('last_update', ''))
        current_time = datetime.now()
        time_diff = (current_time - last_update).total_seconds()
        
        # If no update in 2 minutes, consider it stalled
        if time_diff > 120 and status_data.get('status') == 'processing':
            status_data['status'] = 'stalled'
            status_data['error'] = f'No update in {time_diff:.1f} seconds'
            
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"Error checking update status: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/admin/monitor', methods=['GET'])
def monitoring_dashboard():
    """Render the monitoring dashboard page"""
    # Only allow admin users
    user = UserContext.get_current_user()
    is_admin = getattr(user, 'is_admin', False)
    
    # Check for admin access
    if not is_admin:
        return render_template('error_final.html', 
                              error_title="Access Denied", 
                              error_message="Admin access required to view this page."), 403
    
    return render_template('monitoring_final.html', is_admin=is_admin)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """404 error handler"""
    # Get user and admin status for rendering the template
    user = UserContext.get_current_user()
    is_admin = getattr(user, 'is_admin', False) if user else False
    return render_template('error_final.html', error='Page not found', is_admin=is_admin), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    logger.error(f"Server error: {error}")
    # Get user and admin status for rendering the template
    user = UserContext.get_current_user()
    is_admin = getattr(user, 'is_admin', False) if user else False
    return render_template('error_final.html', error='Server error', is_admin=is_admin), 500

@app.route('/api/status', methods=['GET'])
def system_status():
    """Return system status information as JSON."""
    # Require authentication to view system status
    user = UserContext.get_current_user()
    if not user:
        return jsonify({
            "status": "error",
            "message": "Authentication required"
        }), 401
    
    # Check if user is admin
    is_admin = getattr(user, 'is_admin', False)
    if not is_admin:
        return jsonify({
            "status": "error",
            "message": "Admin access required"
        }), 403
    
    try:
        # Basic system info without psutil
        system_info = {
            'os': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count() or "Unknown",
            'total_memory': "Not available without psutil",
            'available_memory': "Not available without psutil"
        }
        
        # Simplified memory usage without psutil
        memory_usage = {
            'current_mb': 0,
            'peak_mb': 0
        }
        
        # Check for active processes
        active_processes = []
        
        # Check for active update processes by looking for progress files
        import os
        import glob
        import json
        from datetime import datetime
        
        # Path to log directory
        log_dir = os.path.join(app.config.get('DATABASE_DIR', 'database'), 'logs')
        
        # Look for update progress files
        update_progress_files = glob.glob(os.path.join(log_dir, 'update_progress_*.json'))
        for file_path in update_progress_files:
            try:
                with open(file_path, 'r') as f:
                    progress = json.load(f)
                    
                # Extract session ID from filename
                import re
                match = re.search(r'update_progress_([^\.]+)\.json', os.path.basename(file_path))
                session_id = match.group(1) if match else "unknown"
                
                # Calculate age of progress file
                last_update = progress.get('last_update', '')
                if last_update:
                    try:
                        last_update_time = datetime.fromisoformat(last_update)
                        age_seconds = (datetime.now() - last_update_time).total_seconds()
                        age = f"{int(age_seconds / 60)} minutes ago" if age_seconds > 60 else f"{int(age_seconds)} seconds ago"
                    except:
                        age = "unknown"
                else:
                    age = "unknown"
                
                # Prepare process info
                process_info = {
                    'type': 'database_update',
                    'session_id': session_id,
                    'progress': progress.get('last_processed_index', 0),
                    'total': progress.get('total', 0),
                    'success_count': progress.get('stats', {}).get('new_count', 0) + progress.get('stats', {}).get('updated_count', 0),
                    'error_count': progress.get('stats', {}).get('errors', 0),
                    'last_update': last_update,
                    'age': age,
                    'user_id': progress.get('user_id')
                }
                
                active_processes.append(process_info)
            except Exception as e:
                app.logger.error(f"Error reading progress file {file_path}: {e}")
        
        # Look for vector rebuild progress files
        rebuild_progress_files = glob.glob(os.path.join(log_dir, 'rebuild_progress_*.json'))
        for file_path in rebuild_progress_files:
            try:
                with open(file_path, 'r') as f:
                    progress = json.load(f)
                    
                # Extract session ID from filename
                import re
                match = re.search(r'rebuild_progress_([^\.]+)\.json', os.path.basename(file_path))
                session_id = match.group(1) if match else "unknown"
                
                # Calculate age of progress file
                last_update = progress.get('last_update', '')
                if last_update:
                    try:
                        last_update_time = datetime.fromisoformat(last_update)
                        age_seconds = (datetime.now() - last_update_time).total_seconds()
                        age = f"{int(age_seconds / 60)} minutes ago" if age_seconds > 60 else f"{int(age_seconds)} seconds ago"
                    except:
                        age = "unknown"
                else:
                    age = "unknown"
                
                # Prepare process info
                process_info = {
                    'type': 'vector_rebuild',
                    'session_id': session_id,
                    'last_processed_index': progress.get('last_processed_index', 0),
                    'success_count': progress.get('success_count', 0),
                    'error_count': progress.get('error_count', 0),
                    'last_update': last_update,
                    'age': age,
                    'user_id': progress.get('user_id')
                }
                
                active_processes.append(process_info)
            except Exception as e:
                app.logger.error(f"Error reading rebuild progress file {file_path}: {e}")
        
        # Get system info
        import platform
        import psutil
        
        system_info = {
            'os': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'total_memory': f"{psutil.virtual_memory().total / (1024*1024*1024):.1f} GB",
            'available_memory': f"{psutil.virtual_memory().available / (1024*1024*1024):.1f} GB",
            'memory_percent': f"{psutil.virtual_memory().percent}%"
        }
        
        # Build response
        response = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'memory_usage': memory_usage,
            'active_processes': active_processes,
            'system_info': system_info
        }
        
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error getting system status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/get-system-paths-info', methods=['GET'])
def get_system_paths_info():
    """Return information about system paths and environment."""
    # Require authentication to view system paths
    user = UserContext.get_current_user()
    if not user:
        return jsonify({
            "status": "error",
            "message": "Authentication required"
        }), 401
    
    # Check if user is admin
    is_admin = getattr(user, 'is_admin', False)
    if not is_admin:
        return jsonify({
            "status": "error",
            "message": "Admin access required"
        }), 403
    
    try:
        # Get system environment variables
        env_vars = dict(os.environ)
        
        # Filter out sensitive information
        sensitive_keys = ['SECRET', 'KEY', 'PASSWORD', 'TOKEN', 'CREDENTIAL']
        for key in list(env_vars.keys()):
            for pattern in sensitive_keys:
                if pattern.lower() in key.lower():
                    env_vars[key] = "***REDACTED***"
        
        # Get system paths
        system_paths = {
            'cwd': os.getcwd(),
            'python_path': sys.path,
            'os_temp': os.path.join(os.sep, 'tmp'),
            'user_temp': os.path.expanduser('~/tmp')
        }
        
        # Basic system info without psutil
        system_info = {
            'os': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count() or "Unknown",
            'total_memory': "Not available without psutil",
            'available_memory': "Not available without psutil"
        }
        
        # Get disk space information
        try:
            import shutil
            disk_info = {}
            
            # Get disk usage for current directory
            disk_usage = shutil.disk_usage(os.getcwd())
            disk_info['current_directory'] = {
                'path': os.getcwd(),
                'total_gb': f"{disk_usage.total / (1024*1024*1024):.1f} GB",
                'used_gb': f"{disk_usage.used / (1024*1024*1024):.1f} GB",
                'free_gb': f"{disk_usage.free / (1024*1024*1024):.1f} GB",
                'percent_used': f"{(disk_usage.used / disk_usage.total) * 100:.1f}%"
            }
        except Exception as e:
            disk_info = {'error': str(e)}

        return jsonify({
            "status": "ok",
            "system_info": system_info,
            "system_paths": system_paths,
            "disk_info": disk_info,
            "environment": env_vars
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/api/diagnostics', methods=['GET'])
@with_user_context
def path_diagnostics():
    """Endpoint to help diagnose path issues in production"""
    # Only allow admins to access this endpoint
    user = UserContext.get_current_user()
    if not user or not getattr(user, 'is_admin', False):
        return jsonify({"error": "Unauthorized"}), 401
        
    return jsonify(get_system_paths_info())

@app.route('/api/db-health', methods=['GET'])
def db_health_check():
    """
    Simple database health check that tests the connection
    and returns the status. This can be used by monitoring services.
    """
    try:
        from database.multi_user_db.db_final import _engine
        if _engine is None:
            raise ValueError("Database engine not initialized")
            
        # Test direct connection
        with _engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            connection_ok = result.scalar() == 1
            
        # Get pool status
        pool_status = {
            "overflow": _engine.pool.overflow(),
            "checkedin": _engine.pool.checkedin(),
            "checkedout": _engine.pool.checkedout(),
            "size": _engine.pool.size()
        }
            
        return jsonify({
            "status": "healthy" if connection_ok else "degraded",
            "timestamp": datetime.now().isoformat(),
            "connection_ok": connection_ok,
            "pool_status": pool_status
        }), 200 if connection_ok else 503
        
    except Exception as e:
        app.logger.error(f"Database health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 503

@app.route('/api/db-schema', methods=['GET'])
def db_schema_diagnostics():
    """Check the database schema."""
    try:
        # Establish a connection to the database
        from sqlalchemy import create_engine, text
        from database.multi_user_db.db_final import get_db_url, _engine
        
        # Get schema information
        engine = _engine
        with engine.connect() as connection:
            # Query schema information for the bookmarks table
            result = connection.execute(
                text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'bookmarks'")
            )
            
            columns = [{"column_name": row[0], "data_type": row[1]} for row in result]
            
            # Get connection pool statistics if available
            pool_stats = {}
            if hasattr(engine, 'pool'):
                pool_stats = {
                    "overflow": engine.pool.overflow(),
                    "checkedin": engine.pool.checkedin(),
                    "checkedout": engine.pool.checkedout(),
                    "size": engine.pool.size(),
                    "total_connections": engine.pool.checkedin() + engine.pool.checkedout()
                }
            
            # Return results
            return jsonify({
                "status": "ok",
                "database_connection": "established",
                "table_name": "bookmarks",
                "columns": columns,
                "pool_stats": pool_stats,
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        app.logger.error(f"Error getting database schema: {str(e)}")
        return jsonify({
            "status": "error",
            "database_connection": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/admin/database-diagnostics', methods=['GET'])
@with_user_context
def database_diagnostics_page():
    """Admin page for database diagnostics"""
    user = UserContext.get_current_user()
    if not user or not getattr(user, 'is_admin', False):
        return redirect(url_for('auth.login'))
    
    # Get schema information
    try:
        from sqlalchemy import create_engine, text
        from database.multi_user_db.db_final import get_db_url
        
        # Get database URL
        db_url = get_db_url()
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Get bookmarks table schema
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'bookmarks'
                ORDER BY ordinal_position
            """))
            
            columns = [{"column_name": row[0], "data_type": row[1]} for row in result]
            
            # Get sample data
            result = conn.execute(text("""
                SELECT id, user_id, created_at 
                FROM bookmarks
                LIMIT 5
            """))
            
            sample_data = [{"id": row[0], "user_id": row[1], "created_at": row[2]} for row in result]
            
            # Test query with the correct column names
            test_query = """
            SELECT id, 
                   text, 
                   created_at, 
                   author_name, 
                   author_username, 
                   media_files, 
                   raw_data, 
                   user_id
            FROM bookmarks
            LIMIT 1
            """
            try:
                result = conn.execute(text(test_query))
                test_result = "Success! Query executed without errors."
            except Exception as e:
                test_result = f"Error: {str(e)}"
                
            return render_template(
                'admin/database_diagnostics.html',
                schema=columns,
                sample_data=sample_data,
                test_result=test_result,
                user=user,
                is_admin=True
            )
    except Exception as e:
        error_message = str(e)
        return render_template(
            'admin/database_diagnostics.html',
            error=error_message,
            user=user,
            is_admin=True
        )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 