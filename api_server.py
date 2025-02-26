import sys
import os
from pathlib import Path
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import shutil
from flask import Flask, request, jsonify, render_template
import traceback
import logging
from flask_cors import CORS
import uuid

# Set up the working directory to be the project root
PA_BASE_DIR = os.getenv('PA_BASE_DIR', '/home/mariovallereyes/twitter_bookmark_manager')
project_root = Path(PA_BASE_DIR).resolve()

# Configure logging with absolute PythonAnywhere paths
LOG_DIR = os.path.join(PA_BASE_DIR, 'logs')
API_LOG_FILE = os.path.join(PA_BASE_DIR, 'api_server.log')
TEMP_UPLOADS_DIR = os.path.join(PA_BASE_DIR, 'temp_uploads')
DATABASE_DIR = os.path.join(PA_BASE_DIR, 'database')
HISTORY_DIR = os.path.join(DATABASE_DIR, 'json_history')

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Configure logging BEFORE any other operations
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(API_LOG_FILE, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("="*80)
logger.info("Starting API server")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Log file: {API_LOG_FILE}")
logger.info(f"Log directory: {LOG_DIR}")
logger.info(f"Python path: {sys.path}")
logger.info("="*80)

# Set PythonAnywhere environment flag
os.environ['PYTHONANYWHERE_ENVIRONMENT'] = 'true'
logger.info("Set PYTHONANYWHERE_ENVIRONMENT flag")

# Add project directory to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

# Change working directory to project root
os.chdir(project_root)
logger.info(f"Changed working directory to: {os.getcwd()}")

# Create the Flask app first
app = Flask(__name__, 
            template_folder="twitter_bookmark_manager/web/templates",
            static_folder="twitter_bookmark_manager/web/static")

# Configure CORS - Allow all origins during testing
CORS(app)

# Add root route for basic navigation
@app.route('/')
def index():
    """Serve the index.html template with basic bookmark information"""
    try:
        from twitter_bookmark_manager.core.search import BookmarkSearch
        search = BookmarkSearch()
        
        # Get categories for filter options
        categories = search.get_categories()
        
        # Get 5 most recent bookmarks for homepage
        latest_tweets = search.get_all_bookmarks(limit=5)
        
        # Format the latest tweets consistently
        formatted_latest = [{
            'id': str(tweet['id']),
            'text': tweet['text'],
            'author_username': tweet['author'].replace('@', ''),
            'categories': tweet['categories'],
            'created_at': tweet['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        } for tweet in latest_tweets]
        
        return render_template('index.html', 
                              categories=categories,
                              latest_tweets=formatted_latest)
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        logger.error(traceback.format_exc())
        # Fallback to API listing if template rendering fails
        return jsonify({
            "status": "error",
            "message": "Error rendering template",
            "error": str(e),
            "endpoints": [
                {"path": "/upload-bookmarks", "method": "POST", "description": "Upload Twitter bookmarks JSON file"},
                {"path": "/update-database", "method": "POST", "description": "Update database with bookmarks"},
                {"path": "/debug-database", "method": "GET", "description": "Get debug information about the database"},
                {"path": "/api/status", "method": "GET", "description": "Get API status information"}
            ]
        })

@app.route('/chat')
def chat():
    """Serve the chat interface"""
    try:
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Error rendering chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search')
def search():
    """Search bookmarks"""
    try:
        from twitter_bookmark_manager.core.search import BookmarkSearch
        search = BookmarkSearch()
        
        # Get parameters
        query = request.args.get('q', '')
        user = request.args.get('user', '')
        
        # Get categories
        categories = search.get_categories()
        
        # Perform search
        if query:
            results = search.search_bookmarks(query)
        elif user:
            results = search.search_by_user(user)
        else:
            results = []
        
        # Format results
        formatted_results = [{
            'id': str(tweet['id']),
            'text': tweet['text'],
            'author_username': tweet['author'].replace('@', ''),
            'categories': tweet['categories'],
            'created_at': tweet['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        } for tweet in results]
        
        return render_template('index.html',
                              categories=categories,
                              results=formatted_results,
                              query=query,
                              showing_results=len(formatted_results),
                              total_results=len(formatted_results),
                              total_tweets=search.get_total_tweet_count())
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recent')
def recent():
    """Get recent bookmarks"""
    try:
        from twitter_bookmark_manager.core.search import BookmarkSearch
        search = BookmarkSearch()
        
        # Get categories
        categories = search.get_categories()
        
        # Get recent bookmarks
        results = search.get_all_bookmarks(limit=50)
        
        # Format results
        formatted_results = [{
            'id': str(tweet['id']),
            'text': tweet['text'],
            'author_username': tweet['author'].replace('@', ''),
            'categories': tweet['categories'],
            'created_at': tweet['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        } for tweet in results]
        
        return render_template('index.html',
                              categories=categories,
                              results=formatted_results,
                              showing_results=len(formatted_results),
                              total_results=len(formatted_results),
                              total_tweets=search.get_total_tweet_count(),
                              is_recent=True)
    except Exception as e:
        logger.error(f"Error in recent: {e}")
        return jsonify({"error": str(e)}), 500

# PythonAnywhere-specific upload configuration
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER=TEMP_UPLOADS_DIR  # Use the absolute path
)

# Import our PA-specific update function - try using proper path handling
try:
    # Try more explicit import paths first
    pa_update_module = project_root / 'twitter_bookmark_manager' / 'deployment' / 'pythonanywhere' / 'database' / 'update_bookmarks_pa.py'
    
    if pa_update_module.exists():
        logger.debug(f"Found update_bookmarks_pa.py at {pa_update_module}")
        # Import using the Python path
        from twitter_bookmark_manager.deployment.pythonanywhere.database.update_bookmarks_pa import pa_update_bookmarks
    else:
        # Fallback to original import
        from twitter_bookmark_manager.deployment.pythonanywhere.database.update_bookmarks_pa import pa_update_bookmarks
    
    logger.debug("Successfully imported pa_update_bookmarks")
except ImportError as e:
    logger.error(f"Failed to import pa_update_bookmarks: {e}")
    logger.error(traceback.format_exc())
    raise

def ensure_directory_exists(directory):
    """Helper function to create directory and log the operation"""
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")
        logger.debug(f"Directory permissions: {oct(os.stat(directory).st_mode)[-3:]}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/upload-bookmarks', methods=['POST'])
def upload_bookmarks():
    """Handle bookmark JSON file upload"""
    temp_path = None
    # Generate a unique ID for this upload session to track through logs
    session_id = str(uuid.uuid4())[:8]
    try:
        logger.info("="*80)
        logger.info(f"🚀 [UPLOAD-{session_id}] Starting upload handler at {datetime.now().isoformat()}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Request method: {request.method}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Request headers: {dict(request.headers)}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Request files: {list(request.files.keys()) if request.files else 'No files'}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Current directory: {os.getcwd()}")
        
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
        
        # 4. Save uploaded file to temporary location
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 4: Saving file to temporary location")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{secure_filename(file.filename)}")
        logger.info(f"💾 [UPLOAD-{session_id}] Saving file to: {temp_path}")
        
        try:
            # Make sure the upload directory exists
            ensure_directory_exists(app.config['UPLOAD_FOLDER'])
            logger.info(f"📁 [UPLOAD-{session_id}] Upload directory confirmed: {app.config['UPLOAD_FOLDER']}")
            
            # Save the file
            file.save(temp_path)
            logger.info(f"✅ [UPLOAD-{session_id}] File saved successfully to temporary location")
            logger.info(f"🔍 [UPLOAD-{session_id}] File exists: {os.path.exists(temp_path)}")
            logger.info(f"📊 [UPLOAD-{session_id}] File size: {os.path.getsize(temp_path)} bytes")
            
            # Double-check permissions
            file_stats = os.stat(temp_path)
            logger.info(f"🔐 [UPLOAD-{session_id}] File permissions: {oct(file_stats.st_mode)[-3:]}")
        except Exception as e:
            logger.error(f"❌ [UPLOAD-{session_id}] Failed to save file: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # 5. Handle backup and file movement
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 5: Creating backup and moving file to final location")
        current_file = os.path.join(DATABASE_DIR, 'twitter_bookmarks.json')
        logger.info(f"🎯 [UPLOAD-{session_id}] Target file path: {current_file}")
        logger.info(f"🔍 [UPLOAD-{session_id}] Target exists: {os.path.exists(current_file)}")
        
        if os.path.exists(current_file):
            # Use timestamp format that includes hours and minutes
            backup_date = datetime.now().strftime("%Y%m%d_%H%M")
            history_file = os.path.join(HISTORY_DIR, f'twitter_bookmarks_{backup_date}.json')
            logger.info(f"💾 [UPLOAD-{session_id}] Backup file path: {history_file}")
            
            # Backup current file (no need to check if exists since timestamp makes it unique)
            try:
                # Ensure history directory exists
                ensure_directory_exists(HISTORY_DIR)
                logger.info(f"📁 [UPLOAD-{session_id}] History directory confirmed: {HISTORY_DIR}")
                
                # Create backup
                shutil.copy2(current_file, history_file)
                logger.info(f"✅ [UPLOAD-{session_id}] Backup created successfully")
                logger.info(f"🔍 [UPLOAD-{session_id}] Backup exists: {os.path.exists(history_file)}")
                logger.info(f"📊 [UPLOAD-{session_id}] Backup size: {os.path.getsize(history_file)} bytes")
            except Exception as e:
                logger.error(f"❌ [UPLOAD-{session_id}] Backup failed: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': 'Failed to create backup'}), 500
        
        # 6. Move new file to final location
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 6: Moving file to final location")
        try:
            # Ensure database directory exists
            ensure_directory_exists(DATABASE_DIR)
            logger.info(f"📁 [UPLOAD-{session_id}] Database directory confirmed: {DATABASE_DIR}")
            
            # Remove existing file if it exists
            if os.path.exists(current_file):
                os.remove(current_file)
                logger.info(f"🗑️ [UPLOAD-{session_id}] Removed existing file")
            
            # Move temp file to final location
            shutil.move(temp_path, current_file)
            logger.info(f"✅ [UPLOAD-{session_id}] File moved to final location: {current_file}")
            logger.info(f"🔍 [UPLOAD-{session_id}] Final file exists: {os.path.exists(current_file)}")
            logger.info(f"📊 [UPLOAD-{session_id}] Final file size: {os.path.getsize(current_file)} bytes")
            
            # 7. Return success response
            logger.info(f"🎉 [UPLOAD-{session_id}] Upload process completed successfully")
            return jsonify({
                'message': 'File uploaded successfully',
                'session_id': session_id,
                'details': {
                    'original_name': file.filename,
                    'final_path': current_file,
                    'backup_created': os.path.exists(history_file) if 'history_file' in locals() else False
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
        # 8. Cleanup
        logger.info(f"📋 [UPLOAD-{session_id}] STEP 8: Cleanup")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"🧹 [UPLOAD-{session_id}] Temporary file cleaned up")
            except Exception as e:
                logger.error(f"⚠️ [UPLOAD-{session_id}] Failed to cleanup temporary file: {str(e)}")
        logger.info(f"🏁 [UPLOAD-{session_id}] Upload handler completed at {datetime.now().isoformat()}")
        logger.info("="*80)

@app.route('/update-database', methods=['POST'])
def update_database():
    """Update database with new bookmarks"""
    # Generate a unique ID for this update session
    session_id = str(uuid.uuid4())[:8]
    try:
        logger.info("="*80)
        logger.info(f"🚀 [UPDATE-{session_id}] Starting database update process at {datetime.now().isoformat()}")
        
        # Get the start index from request or progress file
        start_index = request.json.get('start_index', 0) if request.is_json else 0
        progress_file = os.path.join(DATABASE_DIR, 'update_progress.json')
        
        # If no start_index provided, check progress file
        if start_index == 0 and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    start_index = progress.get('last_processed_index', 0)
                logger.info(f"📊 [UPDATE-{session_id}] Resuming from index {start_index}")
            except Exception as e:
                logger.error(f"❌ [UPDATE-{session_id}] Error reading progress: {e}")
        
        # 3. Begin database update process
        logger.info(f"📋 [UPDATE-{session_id}] STEP 3: Initiating database update with pa_update_bookmarks")
        try:
            logger.info(f"⚙️ [UPDATE-{session_id}] Calling pa_update_bookmarks(session_id={session_id}, start_index={start_index})")
            result = pa_update_bookmarks(session_id=session_id, start_index=start_index)
            logger.info(f"✅ [UPDATE-{session_id}] pa_update_bookmarks completed")
            logger.info(f"📊 [UPDATE-{session_id}] Update result: {result}")
        except Exception as e:
            error_msg = f"Error calling pa_update_bookmarks: {str(e)}"
            logger.error(f"❌ [UPDATE-{session_id}] {error_msg}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': error_msg,
                'session_id': session_id,
                'traceback': traceback.format_exc(),
                'next_index': start_index,
                'should_continue': False
            }), 500
        
        # 4. Process the result
        logger.info(f"📋 [UPDATE-{session_id}] STEP 4: Processing update result")
        if result.get('success', False):
            progress = result.get('progress', {})
            is_complete = result.get('is_complete', False)
            next_index = result.get('next_index', progress.get('processed', start_index))
            
            response = {
                'success': True,
                'session_id': session_id,
                'progress': progress,
                'is_complete': is_complete,
                'next_index': next_index,
                'message': 'Processing completed successfully' if is_complete else 'Processing paused - please continue',
                'should_continue': not is_complete
            }
            
            status_code = 200 if is_complete else 202  # 202 Accepted means processing should continue
            return jsonify(response), status_code
        else:
            error_msg = result.get('error', 'Unknown error during update')
            progress = result.get('progress', {})
            
            logger.error(f"❌ [UPDATE-{session_id}] Database update failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'session_id': session_id,
                'traceback': result.get('traceback'),
                'progress': progress,
                'next_index': progress.get('last_index', start_index),
                'should_continue': True
            }), 500
            
    except Exception as e:
        logger.error(f"❌ [UPDATE-{session_id}] Database update error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id,
            'traceback': traceback.format_exc(),
            'should_continue': False
        }), 500
    finally:
        logger.info(f"🏁 [UPDATE-{session_id}] Database update process completed at {datetime.now().isoformat()}")
        logger.info("="*80)

@app.route('/debug-database', methods=['GET'])
def debug_database():
    """Debug endpoint to check database connection and view categories"""
    session_id = str(uuid.uuid4())[:8]
    try:
        logger.info(f"🔍 [DEBUG-{session_id}] Starting database debug")
        
        # Try to import the database modules
        try:
            logger.info(f"🔍 [DEBUG-{session_id}] Importing database modules")
            from database.db import get_session
            from database.models import Category, Tweet
            logger.info(f"✅ [DEBUG-{session_id}] Database modules imported successfully")
        except Exception as e:
            logger.error(f"❌ [DEBUG-{session_id}] Failed to import database modules: {str(e)}")
            return jsonify({
                'error': 'Failed to import database modules',
                'details': str(e)
            }), 500
        
        # Try to connect to the database
        try:
            logger.info(f"🔍 [DEBUG-{session_id}] Getting database session")
            with get_session() as session:
                logger.info(f"✅ [DEBUG-{session_id}] Database session created successfully")
                
                # Query categories
                logger.info(f"🔍 [DEBUG-{session_id}] Querying categories")
                categories = session.query(Category).all()
                category_list = [{'id': c.id, 'name': c.name} for c in categories]
                logger.info(f"✅ [DEBUG-{session_id}] Found {len(category_list)} categories")
                
                # Get tweet count
                logger.info(f"🔍 [DEBUG-{session_id}] Counting tweets")
                tweet_count = session.query(Tweet).count()
                logger.info(f"✅ [DEBUG-{session_id}] Found {tweet_count} tweets")
                
                return jsonify({
                    'status': 'success',
                    'database_connection': 'ok',
                    'category_count': len(category_list),
                    'categories': category_list,
                    'tweet_count': tweet_count
                })
        except Exception as e:
            logger.error(f"❌ [DEBUG-{session_id}] Database query error: {str(e)}")
            return jsonify({
                'error': 'Database query error',
                'details': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"❌ [DEBUG-{session_id}] Debug error: {str(e)}")
        return jsonify({
            'error': 'Debug error',
            'details': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint to check database status"""
    logger.info("📊 [API] Checking database status")
    try:
        # Import the check function
        from deployment.pythonanywhere.database.db_pa import check_database_status
        
        # Get database status
        status = check_database_status()
        logger.info(f"📊 [API] Database status: {status['database_connection']}")
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"❌ [API] Error checking database status: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "database_connection": "error",
            "error_message": str(e),
            "from": "api_status endpoint"
        }), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for the chat functionality"""
    session_id = str(uuid.uuid4())[:8]
    try:
        logger.info(f"🤖 [CHAT-{session_id}] Starting chat request")
        
        # Check if request contains proper JSON
        if not request.is_json:
            logger.error(f"❌ [CHAT-{session_id}] Request does not contain valid JSON")
            return jsonify({"error": "Request must contain valid JSON"}), 400
        
        # Get message and context from request
        data = request.json
        message = data.get('message')
        context = data.get('context', {})
        
        if not message:
            logger.error(f"❌ [CHAT-{session_id}] No message provided")
            return jsonify({"error": "No message provided"}), 400
        
        logger.info(f"💬 [CHAT-{session_id}] Message: {message[:50]}...")
        
        # Import chat engine
        try:
            from twitter_bookmark_manager.core.search import BookmarkSearch
            from twitter_bookmark_manager.core.chat.engine import BookmarkChat
            
            search = BookmarkSearch()
            chat_engine = BookmarkChat(search_engine=search)
            
            logger.info(f"✅ [CHAT-{session_id}] Successfully loaded chat engine")
        except Exception as e:
            logger.error(f"❌ [CHAT-{session_id}] Error loading chat engine: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Error loading chat engine",
                "details": str(e)
            }), 500
        
        # Process chat request
        try:
            response, bookmarks, model = chat_engine.chat(message, context.get('history', []))
            logger.info(f"✅ [CHAT-{session_id}] Chat response generated")
            
            # Return response, used bookmarks, and model information
            return jsonify({
                "response": response,
                "bookmarks": bookmarks,
                "model": model
            })
        except Exception as e:
            logger.error(f"❌ [CHAT-{session_id}] Error generating chat response: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Error generating chat response",
                "details": str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"❌ [CHAT-{session_id}] Unexpected error in chat endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Unexpected error",
            "details": str(e)
        }), 500

# Make the app available for WSGI
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    ) 