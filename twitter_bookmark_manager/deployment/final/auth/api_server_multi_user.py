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
from flask_login import login_required, current_user
import uuid
import traceback
import shutil
from werkzeug.utils import secure_filename

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
        # Don't pass user_id again, it's already in the searcher instance
        results = searcher.get_recent()
        
        # Get all categories for display
        # Don't pass user_id again, it's already in the searcher instance
        categories = searcher.get_categories()
        
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
        # Don't pass user_id again, it's already in the searcher instance
        categories = searcher.get_categories()
        
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
        # Don't pass user_id again, it's already in the searcher instance
        categories = searcher.get_categories()
        
        return jsonify(categories)
    finally:
        conn.close()

# Upload bookmarks endpoint
@app.route('/upload-bookmarks', methods=['POST'])
def upload_bookmarks():
    """Handle bookmark JSON file upload - multi-user aware"""
    # Get current user
    user = UserContext.get_current_user()
    
    # Redirect to login if not authenticated
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    temp_path = None
    # Generate a unique ID for this upload session to track through logs
    session_id = str(uuid.uuid4())[:8]
    try:
        logger.info("="*80)
        logger.info(f"🚀 [UPLOAD-{session_id}] User {user.id} starting upload handler at {datetime.now().isoformat()}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Request method: {request.method}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Request headers: {dict(request.headers)}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Request files: {list(request.files.keys()) if request.files else 'No files'}")
        
        # 1. Check if file exists in request
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 1: Checking if file exists in request")
        if 'file' not in request.files:
            logger.error(f"❌ [UPLOAD-{session_id}] No file part in request")
            logger.info(f"🔍 [UPLOAD-{session_id}] Form data: {request.form}")
            logger.info(f"🔍 [UPLOAD-{session_id}] Raw data: {request.get_data()}")
            return jsonify({
                'error': 'No file provided',
                'details': {
                    'request_method': request.method,
                    'content_type': request.content_type,
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
            return jsonify({'error': 'Invalid JSON file'}), 400
        
        # 4. Create user-specific directories
        # Define user-specific paths
        base_dir = Path(app.root_path).parent.parent
        upload_folder = base_dir / "uploads" / f"user_{user.id}"
        database_dir = base_dir / "database" / f"user_{user.id}"
        history_dir = database_dir / "json_history"
        
        # Create directories if they don't exist
        upload_folder.mkdir(parents=True, exist_ok=True)
        database_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 [UPLOAD-{session_id}] User-specific directories created/confirmed")
        logger.info(f"📁 [UPLOAD-{session_id}] Upload folder: {upload_folder}")
        logger.info(f"📁 [UPLOAD-{session_id}] Database directory: {database_dir}")
        logger.info(f"📁 [UPLOAD-{session_id}] History directory: {history_dir}")
        
        # 5. Save uploaded file to temporary location
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 5: Saving file to temporary location")
        temp_path = upload_folder / f"{session_id}_{secure_filename(file.filename)}"
        logger.info(f"💾 [UPLOAD-{session_id}] Saving file to: {temp_path}")
        
        try:
            # Save the file
            file.save(temp_path)
            logger.info(f"✅ [UPLOAD-{session_id}] File saved successfully to temporary location")
            logger.info(f"🔍 [UPLOAD-{session_id}] File exists: {os.path.exists(temp_path)}")
            logger.info(f"📊 [UPLOAD-{session_id}] File size: {os.path.getsize(temp_path)} bytes")
        except Exception as e:
            logger.error(f"❌ [UPLOAD-{session_id}] Failed to save file: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # 6. Handle backup and file movement
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 6: Creating backup and moving file to final location")
        current_file = database_dir / 'twitter_bookmarks.json'
        logger.info(f"🎯 [UPLOAD-{session_id}] Target file path: {current_file}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Target exists: {os.path.exists(current_file)}")
        
        if os.path.exists(current_file):
            # Use timestamp format that includes hours and minutes
            backup_date = datetime.now().strftime("%Y%m%d_%H%M")
            history_file = history_dir / f'twitter_bookmarks_{backup_date}.json'
            logger.info(f"💾 [UPLOAD-{session_id}] Backup file path: {history_file}")
            
            # Backup current file
            try:
                shutil.copy2(current_file, history_file)
                logger.info(f"✅ [UPLOAD-{session_id}] Backup created successfully")
                logger.info(f"🔍 [UPLOAD-{session_id}] Backup exists: {os.path.exists(history_file)}")
                logger.info(f"📊 [UPLOAD-{session_id}] Backup size: {os.path.getsize(history_file)} bytes")
            except Exception as e:
                logger.error(f"❌ [UPLOAD-{session_id}] Backup failed: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': 'Failed to create backup'}), 500
        
        # 7. Move new file to final location
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 7: Moving file to final location")
        try:
            # Remove existing file if it exists
            if os.path.exists(current_file):
                os.remove(current_file)
                logger.info(f"🗑️ [UPLOAD-{session_id}] Removed existing file")
            
            # Move temp file to final location
            shutil.move(temp_path, current_file)
            logger.info(f"✅ [UPLOAD-{session_id}] File moved to final location: {current_file}")
            logger.info(f"🔍 [UPLOAD-{session_id}] Final file exists: {os.path.exists(current_file)}")
            logger.info(f"📊 [UPLOAD-{session_id}] Final file size: {os.path.getsize(current_file)} bytes")
            
            # 8. Return success response
            logger.info(f"🎉 [UPLOAD-{session_id}] Upload process completed successfully")
            return jsonify({
                'message': 'File uploaded successfully',
                'session_id': session_id,
                'details': {
                    'original_name': file.filename,
                    'final_path': str(current_file),
                    'backup_created': os.path.exists(history_file) if 'history_file' in locals() else False,
                    'user_id': user.id
                }
            })
        except Exception as e:
            logger.error(f"❌ [UPLOAD-{session_id}] Failed to move file: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to move file to final location'}), 500
        
    except Exception as e:
        logger.error(f"❌ [UPLOAD-{session_id}] Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Upload failed',
            'session_id': session_id,
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500
    finally:
        # 9. Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"🧹 [UPLOAD-{session_id}] Temporary file cleaned up")
            except Exception as e:
                logger.error(f"⚠️ [UPLOAD-{session_id}] Failed to cleanup temporary file: {str(e)}")
        logger.info(f"🏁 [UPLOAD-{session_id}] Upload handler completed at {datetime.now().isoformat()}")
        logger.info("="*80)

# Update database endpoint
@app.route('/update-database', methods=['POST'])
def update_database():
    """Update database with new bookmarks - multi-user aware"""
    # Get current user
    user = UserContext.get_current_user()
    
    # Redirect to login if not authenticated
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Generate a unique ID for this update session
    session_id = str(uuid.uuid4())[:8]
    try:
        logger.info("="*80)
        logger.info(f"🚀 [UPDATE-{session_id}] User {user.id} starting database update process at {datetime.now().isoformat()}")
        
        # Get parameters from request
        if request.is_json:
            start_index = request.json.get('start_index', 0)
            force_rebuild = request.json.get('rebuild_vector', False)
            skip_database = request.json.get('skip_database', False)
        else:
            start_index = 0
            force_rebuild = False
            skip_database = False
            
        logger.info(f"📋 [UPDATE-{session_id}] Request parameters: start_index={start_index}, force_rebuild={force_rebuild}, skip_database={skip_database}")
        
        # Define user-specific paths
        from pathlib import Path
        base_dir = Path(app.root_path).parent.parent
        database_dir = base_dir / "database" / f"user_{user.id}"
        progress_file = database_dir / 'update_progress.json'
        
        # Ensure the database directory exists
        database_dir.mkdir(parents=True, exist_ok=True)
        
        # Import necessary modules
        from database.multi_user_db.update_bookmarks_final import (
            final_update_bookmarks,
            rebuild_vector_store
        )
        
        # Handle force rebuild request
        if force_rebuild or skip_database:
            logger.info(f"🔄 [UPDATE-{session_id}] STEP 1: Force rebuild of vector store requested")
            
            # Perform the rebuild with user_id
            rebuild_result = rebuild_vector_store(session_id=session_id, user_id=user.id)
            logger.info(f"✅ [UPDATE-{session_id}] Rebuild completed with result: {rebuild_result}")
            
            return jsonify({
                'success': rebuild_result.get('success', False),
                'message': 'Vector store rebuild completed',
                'rebuild_result': rebuild_result,
                'session_id': session_id,
                'is_complete': True,
                'user_id': user.id
            }), 200
        
        # Check for loop condition if progress file exists
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                # Check if progress belongs to the current user
                if progress.get('user_id') != user.id:
                    logger.warning(f"⚠️ [UPDATE-{session_id}] Progress file belongs to different user. Creating new progress.")
                    progress = {'user_id': user.id, 'last_processed_index': 0}
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f)
                
                # Check for potential infinite loops
                loop_detection = progress.get('loop_detection', {'count': {}, 'indices': [], 'timestamps': []})
                current_index = str(start_index)
                
                # Update loop detection
                if current_index in loop_detection['count']:
                    loop_detection['count'][current_index] += 1
                else:
                    loop_detection['count'][current_index] = 1
                
                loop_detection['indices'].append(current_index)
                loop_detection['timestamps'].append(datetime.now().isoformat())
                
                # Keep only the last 10 records
                if len(loop_detection['indices']) > 10:
                    loop_detection['indices'] = loop_detection['indices'][-10:]
                    loop_detection['timestamps'] = loop_detection['timestamps'][-10:]
                
                # Check if we're in a loop
                is_in_loop = loop_detection['count'][current_index] >= 3
                
                # Update progress file
                progress['loop_detection'] = loop_detection
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
                
                if is_in_loop:
                    logger.warning(f"⚠️ [UPDATE-{session_id}] Update loop detected before starting! Breaking out with forced rebuild")
                    
                    # Force a vector rebuild to break the loop
                    rebuild_result = rebuild_vector_store(session_id=session_id, user_id=user.id)
                    
                    # Reset progress file
                    try:
                        # Backup the file first
                        backup_file = f"{progress_file}.loop_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        shutil.copy2(progress_file, backup_file)
                        logger.info(f"✅ [UPDATE-{session_id}] Created backup of progress file at {backup_file}")
                        
                        # Reset the progress file
                        os.remove(progress_file)
                        logger.info(f"✅ [UPDATE-{session_id}] Removed progress file to break update loop")
                    except Exception as e:
                        logger.error(f"❌ [UPDATE-{session_id}] Error handling progress file during loop recovery: {e}")
                    
                    return jsonify({
                        'success': True,
                        'message': 'Update loop detected and resolved with vector store rebuild',
                        'rebuild_result': rebuild_result,
                        'session_id': session_id,
                        'is_complete': True,
                        'user_id': user.id
                    }), 200
            except Exception as e:
                logger.error(f"❌ [UPDATE-{session_id}] Error checking for update loop: {e}")
        
        # If no start_index provided, check progress file
        if start_index == 0 and progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    # Only use saved index if it belongs to the current user
                    if progress.get('user_id') == user.id:
                        start_index = progress.get('last_processed_index', 0)
                logger.info(f"📊 [UPDATE-{session_id}] Resuming from index {start_index}")
            except Exception as e:
                logger.error(f"❌ [UPDATE-{session_id}] Error reading progress: {e}")
        
        # Begin database update process
        logger.info(f"📋 [UPDATE-{session_id}] STEP 3: Initiating database update with final_update_bookmarks")
        try:
            logger.info(f"⚙️ [UPDATE-{session_id}] Calling final_update_bookmarks(session_id={session_id}, start_index={start_index}, user_id={user.id})")
            
            # Call the update function with user_id
            result = final_update_bookmarks(
                session_id=session_id, 
                start_index=start_index, 
                user_id=user.id
            )
            
            logger.info(f"✅ [UPDATE-{session_id}] final_update_bookmarks completed")
            logger.info(f"📊 [UPDATE-{session_id}] Update result: {result}")
            
            # Add user_id to result
            result['user_id'] = user.id
            
            # Process the result
            if result.get('success', False):
                progress = result.get('progress', {})
                is_complete = result.get('is_complete', False)
                next_index = result.get('next_index', progress.get('processed', start_index))
                
                status_code = 200 if is_complete else 202  # 202 Accepted means processing should continue
                return jsonify(result), status_code
            else:
                error_msg = result.get('error', 'Unknown error during update')
                progress = result.get('progress', {})
                
                logger.error(f"❌ [UPDATE-{session_id}] Database update failed: {error_msg}")
                return jsonify(result), 500
                
        except Exception as e:
            error_msg = f"Error during update: {str(e)}"
            logger.error(f"❌ [UPDATE-{session_id}] {error_msg}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': error_msg,
                'session_id': session_id,
                'traceback': traceback.format_exc(),
                'next_index': start_index,
                'should_continue': False,
                'user_id': user.id
            }), 500
            
    except Exception as e:
        logger.error(f"❌ [UPDATE-{session_id}] Database update error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id,
            'traceback': traceback.format_exc(),
            'should_continue': False,
            'user_id': user.id
        }), 500
    finally:
        logger.info(f"🏁 [UPDATE-{session_id}] Database update process completed at {datetime.now().isoformat()}")
        logger.info("="*80)

# Other API endpoints would be similarly updated with user_id filtering
# Including upload-bookmarks, update-database, etc.

@app.route('/admin/monitor', methods=['GET'])
@login_required
def monitoring_dashboard():
    """Render the monitoring dashboard page"""
    # Only allow admin users or set a flag for when the user is an admin
    user = UserContext.get_current_user()
    is_admin = getattr(user, 'is_admin', False)
    
    # You might want to add an admin check here in a production environment
    # if not is_admin:
    #     flash('Admin access required')
    #     return redirect(url_for('index'))
    
    return render_template('monitoring_final.html')

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

@app.route('/api/status', methods=['GET'])
@login_required
def system_status():
    """Get system status information including memory usage and running processes"""
    try:
        # Import monitoring functions
        from ..database.multi_user_db.update_bookmarks_final import get_memory_usage

        # Get current memory usage
        memory_usage = get_memory_usage()
        
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 