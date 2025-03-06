import sys
import os
import logging
from pathlib import Path
import traceback

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

# Log all environment variables (redacting sensitive ones)
logger.info("Environment variables:")
for key, value in os.environ.items():
    # Redact sensitive values
    if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
        logger.info(f"  {key}=******")
    else:
        logger.info(f"  {key}={value}")

# Store the import error for later use
import_error = None

# Import the existing Flask app instead of creating a new one
try:
    # First try importing the database module to test connection
    logger.info("Testing database connection")
    try:
        from database.multi_user_db.db_final import check_database_status, setup_database
        
        # Test database connection
        db_status = check_database_status()
        if db_status['healthy']:
            logger.info("✅ Database connection successful")
        else:
            logger.warning(f"⚠️ Database connection issues: {db_status['message']}")
            # Try to setup database again
            setup_database(force_reconnect=True)
    except Exception as db_error:
        logger.error(f"❌ Database connection failed: {db_error}")
        import_error = f"Database connection error: {str(db_error)}"
        
    # Now import the application
    if not import_error:
        logger.info("Importing application from auth.api_server_multi_user")
        from auth.api_server_multi_user import app as application
        logger.info("✅ Successfully imported application from api_server_multi_user")
except Exception as e:
    error_details = traceback.format_exc()
    import_error = str(e)
    logger.error(f"Error importing application: {import_error}")
    logger.error(f"Error details: {error_details}")
    
    # Create a fallback application
    from flask import Flask, jsonify
    application = Flask(__name__, 
                      template_folder='web_final/templates',
                      static_folder='web_final/static')
    
    # Root route for testing
    @application.route('/')
    def index():
        return f"""
        <h1>Twitter Bookmark Manager</h1>
        <p>Error loading application: {import_error}</p>
        <h2>Error Details:</h2>
        <pre>{error_details}</pre>
        <h2>Troubleshooting:</h2>
        <ul>
            <li>Check database connection settings</li>
            <li>Verify that all environment variables are set correctly</li>
            <li>Ensure that the application code is properly deployed</li>
        </ul>
        """
    
    # Add JSON error endpoint for API clients
    @application.route('/api/status')
    def api_status():
        return jsonify({
            "status": "error",
            "message": f"Application failed to initialize: {import_error}",
            "error_details": error_details
        }), 500
    
    logger.error("⚠️ Using fallback application due to import error")

logger.info("="*50) 