import sys
import os
from pathlib import Path
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import shutil
from flask import Flask, request, jsonify
import traceback
import logging
from flask_cors import CORS

# Set up the working directory to be the project root
PA_BASE_DIR = '/home/mariovallereyes/twitter_bookmark_manager'
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
logger.info("="*50)
logger.info("Starting API server")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Log file: {API_LOG_FILE}")
logger.info(f"Log directory: {LOG_DIR}")
logger.info(f"Python path: {sys.path}")
logger.info("="*50)

# Add project directory to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

# Change working directory to project root
os.chdir(project_root)
logger.info(f"Changed working directory to: {os.getcwd()}")

# Import the web application
try:
    logger.debug("Attempting to import web_app...")
    from web.server import app as web_app
    logger.debug("Successfully imported web_app")
except Exception as e:
    logger.error(f"Failed to import web_app: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Import our PA-specific update function
try:
    from twitter_bookmark_manager.deployment.pythonanywhere.database.update_bookmarks_pa import pa_update_bookmarks
    logger.debug("Successfully imported pa_update_bookmarks")
except ImportError as e:
    logger.error(f"Failed to import pa_update_bookmarks: {e}")
    logger.error(traceback.format_exc())
    raise

# Create the main app
app = web_app

# Configure CORS - Allow all origins during testing
CORS(app)

# PythonAnywhere-specific upload configuration
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER=TEMP_UPLOADS_DIR  # Use the absolute path
)

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

def upload_bookmarks():
    """Handle bookmark JSON file upload"""
    temp_path = None
    try:
        logger.info("="*50)
        logger.info("Starting upload handler")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request files: {list(request.files.keys()) if request.files else 'No files'}")
        logger.info(f"Current directory: {os.getcwd()}")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            logger.info(f"Form data: {request.form}")
            logger.info(f"Raw data: {request.get_data()}")
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
        logger.info(f"Received file object: {file}")
        logger.info(f"File name: {file.filename}")
        
        if not file.filename:
            logger.error("No selected file")
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.endswith('.json'):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only JSON files are allowed'}), 400
        
        # Test JSON validity
        try:
            file_content = file.read()
            file.seek(0)
            json.loads(file_content)
            logger.info("JSON validation successful")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {str(e)}")
            return jsonify({'error': 'Invalid JSON file'}), 400
        
        # Save uploaded file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        logger.info(f"Saving file to: {temp_path}")
        
        try:
            file.save(temp_path)
            logger.info("File saved successfully")
            logger.info(f"File exists: {os.path.exists(temp_path)}")
            logger.info(f"File size: {os.path.getsize(temp_path)}")
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Handle backup and file movement
        current_file = os.path.join(DATABASE_DIR, 'twitter_bookmarks.json')
        logger.info(f"Target file path: {current_file}")
        logger.info(f"Target exists: {os.path.exists(current_file)}")
        
        if os.path.exists(current_file):
            # Use timestamp format that includes hours and minutes
            backup_date = datetime.now().strftime("%Y%m%d_%H%M")
            history_file = os.path.join(HISTORY_DIR, f'twitter_bookmarks_{backup_date}.json')
            logger.info(f"Backup file path: {history_file}")
            
            # Backup current file (no need to check if exists since timestamp makes it unique)
            try:
                shutil.copy2(current_file, history_file)
                logger.info("Backup created successfully")
            except Exception as e:
                logger.error(f"Backup failed: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': 'Failed to create backup'}), 500
        
        # Move new file
        try:
            if os.path.exists(current_file):
                os.remove(current_file)
                logger.info("Removed existing file")
            shutil.move(temp_path, current_file)
            logger.info("File moved to final location")
            
            return jsonify({
                'message': 'File uploaded successfully',
                'details': {
                    'original_name': file.filename,
                    'final_path': current_file,
                    'backup_created': os.path.exists(history_file) if 'history_file' in locals() else False
                }
            })
        except Exception as e:
            logger.error(f"Failed to move file: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Failed to move file to final location'}), 500
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Upload failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup temporary file: {str(e)}")
        logger.info("="*50)

def update_database():
    """Update database with new bookmarks"""
    try:
        logger.debug("Starting database update process")
        result = pa_update_bookmarks()
        
        if result.get('success', False):
            logger.info("Database update completed successfully")
            return jsonify({
                'message': 'Database updated successfully',
                'details': {
                    'new_bookmarks': result.get('new_bookmarks', 0),
                    'updated_bookmarks': result.get('updated_bookmarks', 0),
                    'errors': result.get('errors', 0)
                }
            })
        else:
            error_msg = result.get('error', 'Unknown error during update')
            logger.error(f"Database update failed: {error_msg}")
            return jsonify({
                'error': error_msg,
                'traceback': result.get('traceback')
            }), 500
            
    except Exception as e:
        logger.error(f"Database update error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Override the original functions
web_app.view_functions['upload_bookmarks'] = upload_bookmarks
web_app.view_functions['update_database'] = update_database

# For WSGI compatibility
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    ) 