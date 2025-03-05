import sys
import os
import logging
from pathlib import Path

# Set up logging
logging_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

logger.info("="*50)
logger.info("Initializing WSGI Application for Railway")

# Add the current directory to Python path
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    logger.info(f"Added {current_dir} to Python path")

# Add parent directory to Python path
parent_dir = current_dir.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
    logger.info(f"Added {parent_dir} to Python path")

# Add grandparent directory to Python path to find twitter_bookmark_manager module
grandparent_dir = parent_dir.parent.resolve()
if str(grandparent_dir) not in sys.path:
    sys.path.insert(0, str(grandparent_dir))
    logger.info(f"Added {grandparent_dir} to Python path")

# Change working directory
os.chdir(current_dir)
logger.info(f"Changed working directory to: {os.getcwd()}")

# Store the import error for later use
import_error = None

# Import the existing Flask app instead of creating a new one
try:
    logger.info("Importing application from auth.api_server_multi_user")
    from auth.api_server_multi_user import app as application
    logger.info("✅ Successfully imported application from api_server_multi_user")
except Exception as e:
    import_error = str(e)
    logger.error(f"Error importing application: {import_error}")
    # Create a fallback application
    from flask import Flask
    application = Flask(__name__, 
                      template_folder='web_final/templates',
                      static_folder='web_final/static')
    
    # Root route for testing
    @application.route('/')
    def index():
        return f"<h1>Twitter Bookmark Manager</h1><p>Error loading application: {import_error}</p>"
    
    logger.error("⚠️ Using fallback application due to import error")

logger.info("="*50) 