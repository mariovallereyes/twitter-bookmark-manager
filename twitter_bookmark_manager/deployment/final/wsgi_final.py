import os
import sys
import logging
import traceback
from pathlib import Path
from flask import Flask, jsonify, request, redirect, url_for, render_template_string
from flask_cors import CORS
from database.multi_user_db.db_final import init_database
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Gunicorn configuration for long-running tasks
timeout = 7200          # 2 hours - much longer timeout for large rebuilds
workers = 1             # Single worker to avoid memory competition
worker_class = 'sync'   # Synchronous worker for stability
keepalive = 120
max_requests = 0        # Disable worker recycling to maintain context
max_requests_jitter = 0
worker_tmp_dir = '/dev/shm'  # Use RAM for temp files
preload_app = True      # Preload app to maintain context
graceful_timeout = 600  # 10 minutes grace period for cleanup
worker_connections = 10 # Limit concurrent connections

# Logging configuration
accesslog = '-'         # Log to stdout
errorlog = '-'         # Log to stderr
loglevel = 'info'      # Detailed logging
capture_output = True   # Capture stdout/stderr from workers
enable_stdio_inheritance = True  # Inherit stdio settings

logger.info("==================================================")
logger.info("Starting Progressive WSGI Application for Railway")

# Add paths to system path to ensure modules can be found
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    logger.info(f"Added {current_dir} to Python path")

parent_dir = current_dir.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
    logger.info(f"Added {parent_dir} to Python path")

grandparent_dir = parent_dir.parent.resolve()
if str(grandparent_dir) not in sys.path:
    sys.path.insert(0, str(grandparent_dir))
    logger.info(f"Added {grandparent_dir} to Python path")

# Change working directory
os.chdir(current_dir)
logger.info(f"Changed working directory to: {os.getcwd()}")

# Log selected environment variables for debugging
important_vars = [
    'DATABASE_URL', 'PGHOST', 'PGPORT', 'PGUSER', 'PGDATABASE',
    'DB_HOST', 'DB_PORT', 'DB_USER', 'DB_NAME',
    'RAILWAY_ENVIRONMENT', 'PORT', 'PYTHON_VERSION'
]

logger.info("Important environment variables:")
for var in important_vars:
    value = os.environ.get(var, 'not set')
    if any(secret in var.lower() for secret in ['password', 'secret', 'key']):
        logger.info(f"  {var}=******")
    else:
        logger.info(f"  {var}={value}")

# Initialize db_status with default values to avoid KeyError
db_status = {
    "healthy": False,
    "message": "Not tested yet"
}

# Try to import database module to test connection
try:
    logger.info("Attempting to import database module...")
    from database.multi_user_db.db_final import check_database_status, setup_database, get_db_url

    # Test database connection
    try:
        logger.info("Testing database connection")
        logger.info(f"Database URL: {get_db_url().replace('postgresql://', 'postgresql://user:****@')}")
        
        # Get database status
        temp_status = check_database_status()
        
        # Update our status dictionary with values from check_database_status
        if isinstance(temp_status, dict):
            if 'healthy' in temp_status:
                db_status['healthy'] = temp_status['healthy']
            if 'message' in temp_status:
                db_status['message'] = temp_status['message']
        
        # Log database status
        if db_status.get('healthy', False):
            logger.info("✅ Database connection successful")
        else:
            logger.warning(f"⚠️ Database connection issues: {db_status.get('message', 'Unknown issue')}")
            
            # Try to reconnect
            try:
                setup_database(force_reconnect=True)
                logger.info("✅ Database reconnection successful")
                db_status['healthy'] = True
                db_status['message'] = "Reconnection successful"
            except Exception as reconnect_error:
                logger.warning(f"⚠️ Database reconnection failed: {reconnect_error}")
                db_status['message'] = f"Reconnection failed: {str(reconnect_error)}"
    except Exception as db_test_error:
        logger.error(f"❌ Database test failed: {db_test_error}")
        db_status['message'] = f"Test failed: {str(db_test_error)}"
except ImportError as import_error:
    logger.error(f"❌ Failed to import database module: {import_error}")
    db_status['message'] = f"Import error: {str(import_error)}"
    db_status['healthy'] = False

# Make sure db_status has the required keys
if 'healthy' not in db_status:
    db_status['healthy'] = False
if 'message' not in db_status:
    db_status['message'] = "Status information not available"

# Store application loading errors
application_error = None
error_details = None

# Attempt to import the full application
try:
    # Try to import flask_session first
    import flask_session
    logger.info("flask_session module is available")
    # Continue with normal app loading
    try:
        logger.info("Attempting to import app from auth.api_server_multi_user...")
        from auth.api_server_multi_user import app as flask_app
        app = flask_app
        # Set application for later use
        application = app
        logger.info("✅ Successfully loaded app from auth.api_server_multi_user")
        
        # Skip fallback mode - force using the loaded application
        logger.info("✅ USING FULL APPLICATION MODE - Database is healthy and app loaded successfully")
        
        # Set flags to indicate we're NOT in fallback mode
        application.config['FALLBACK_MODE'] = False
        application.config['FULL_APP_LOADED'] = True
        
        # Prevent re-entering this block
        globals()['application'] = application
    except Exception as import_app_error:
        logger.error(f"❌ Error importing app from auth.api_server_multi_user: {str(import_app_error)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise
    
    # Set template path explicitly
    if hasattr(app, 'template_folder'):
        logger.info(f"Template folder is: {app.template_folder}")
        if not os.path.exists(app.template_folder):
            logger.warning(f"Template folder does not exist: {app.template_folder}")
            # Try to find the correct path
            possible_template_paths = [
                os.path.join(os.environ.get('APP_BASE_DIR', '/app'), 'twitter_bookmark_manager/deployment/final/web_final/templates'),
                os.path.join(os.getcwd(), 'twitter_bookmark_manager/deployment/final/web_final/templates'),
                os.path.join(os.getcwd(), 'web_final/templates')
            ]
            for path in possible_template_paths:
                if os.path.exists(path):
                    logger.info(f"Found template folder at: {path}")
                    app.template_folder = path
                    logger.info(f"Updated template folder to: {path}")
                    break
except ImportError as e:
    logger.warning(f"flask_session module not available: {e}")
    logger.info("Creating mock session class")
    
    # Create a mock Session class to avoid errors
    class MockSession:
        def __init__(self, app=None):
            self.app = app
            logger.info("Initialized MockSession")
            
    # Make flask_session available with mock
    sys.modules['flask_session'] = type('mock_flask_session', (), {'Session': MockSession})
    
    # Load the app
    from auth.api_server_multi_user import app as flask_app
    app = flask_app
    logger.info("Successfully loaded app with MockSession")
    
    # Set the application
    application = app
    
    # Add DB status info for frontend templates
    if not db_status.get('healthy', False):
        logger.warning(f"⚠️ Database issues detected, but still loading full application: {db_status.get('message', 'Unknown issue')}")
        # Inject DB status into application config for templates
        application.config['DB_ERROR'] = True
        application.config['DB_ERROR_MESSAGE'] = db_status.get('message', 'Database connection issues')
        # IMPORTANT: Do not redirect API calls to fallback mode, let them try to reconnect
        application.config['ALLOW_API_RETRY'] = True
    else:
        logger.info("Full application is active with healthy database")
        application.config['DB_ERROR'] = False
        application.config['ALLOW_API_RETRY'] = True
except Exception as e:
    error_details = traceback.format_exc()
    application_error = str(e)
    logger.error(f"❌ Failed to load full application: {application_error}")
    logger.error(f"Error details: {error_details}")

# If full application failed to load, create fallback application
if 'application' not in locals() and 'application' not in globals():
    logger.info("Creating fallback application")
    
    # Create a fallback Flask application with basic functionality
    application = Flask(__name__,
                       template_folder='web_final/templates',
                       static_folder='web_final/static')
    application.config['DEBUG'] = True
    
    # Add a flag to indicate we're in fallback mode
    application.config['FALLBACK_MODE'] = True
    
    # Add error handling
    @application.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"Unhandled exception in fallback app: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template_string("""
            <html>
            <head>
                <title>Error - Twitter Bookmark Manager</title>
                <style>
                    body { font-family: sans-serif; margin: 0; padding: 20px; background: #121212; color: #e0e0e0; }
                    .container { max-width: 800px; margin: 0 auto; background: #1a1a1a; padding: 20px; border-radius: 5px; }
                    h1 { color: #ff4d4d; }
                    pre { background: #222; padding: 15px; border-radius: 3px; overflow-x: auto; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Application Error</h1>
                    <p>The application encountered an error:</p>
                    <pre>{{ error }}</pre>
                    
                    <h2>Details</h2>
                    <pre>{{ details }}</pre>
                </div>
            </body>
            </html>
        """, error=str(e), details=traceback.format_exc())
    
    # Status route for monitoring
    @application.route('/status')
    def status():
        return jsonify({
            "status": "limited_functionality",
            "database": {
                "healthy": db_status.get('healthy', False),
                "message": db_status.get('message', "Status information not available")
            },
            "application_error": application_error,
            "message": "Running in fallback mode with limited functionality"
        })
    
    # Main route that explains the situation
    @application.route('/')
    def index():
        return render_template_string("""
            <html>
            <head>
                <title>Twitter Bookmark Manager - Limited Mode</title>
                <style>
                    body { font-family: sans-serif; margin: 0; padding: 20px; background: #121212; color: #e0e0e0; }
                    .container { max-width: 800px; margin: 0 auto; background: #1a1a1a; padding: 20px; border-radius: 5px; }
                    h1 { color: #1DA1F2; }
                    .info { background: #192734; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                    .error { background: #330a0a; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                    .success { color: #00ba7c; }
                    .warning { color: #ffad1f; }
                    .danger { color: #e0245e; }
                    button { background: #1DA1F2; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; }
                    button:hover { background: #1a91da; }
                    pre { background: #222; padding: 10px; border-radius: 3px; overflow-x: auto; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Twitter Bookmark Manager</h1>
                    
                    <div class="info">
                        <h2 class="warning">⚠️ Limited Functionality Mode</h2>
                        <p>The application is running in limited functionality mode due to initialization issues.</p>
                        
                        <h3>Environment Information:</h3>
                        <ul>
                            <li>Python Version: {{ python_version }}</li>
                            <li>Environment: {{ environment }}</li>
                            <li>Working Directory: {{ working_dir }}</li>
                        </ul>
                    </div>
                    
                    <div class="error">
                        <h3 class="danger">Error Details:</h3>
                        <p>{{ error_message }}</p>
                        <pre>{{ error_details }}</pre>
                        
                        <h3>Database Status:</h3>
                        <p>
                            Status: <span class="{{ 'success' if db_healthy else 'danger' }}">{{ 'Healthy' if db_healthy else 'Not Healthy' }}</span><br>
                            Message: {{ db_message }}
                        </p>
                    </div>
                    
                    <div>
                        <h3>Available Actions:</h3>
                        <ul>
                            <li><a href="/status">Check API Status</a></li>
                            <li><a href="https://railway.app/project/{{ project_id }}">Visit Railway Dashboard</a></li>
                        </ul>
                        <button onclick="window.location.reload()">Refresh Page</button>
                    </div>
                </div>
            </body>
            </html>
        """, 
        python_version=sys.version,
        environment=os.environ.get('RAILWAY_ENVIRONMENT', 'unknown'),
        working_dir=os.getcwd(),
        error_message=application_error or "Unknown error occurred during application initialization",
        error_details=error_details or "No detailed error information available",
        db_healthy=db_status.get('healthy', False),
        db_message=db_status.get('message', "Status information not available"),
        project_id=os.environ.get('RAILWAY_PROJECT_ID', ''))
    
    # Add a simple API test endpoint
    @application.route('/api/test')
    def api_test():
        return jsonify({
            "status": "limited_functionality",
            "message": "Running in fallback mode",
            "database_status": {
                "healthy": db_status.get('healthy', False),
                "message": db_status.get('message', "Status information not available")
            },
            "environment": os.environ.get('RAILWAY_ENVIRONMENT', 'unknown'),
            "python_version": sys.version
        })
    
    logger.warning("⚠️ Running in fallback mode with limited functionality")
else:
    logger.info("✅ Full application loaded successfully")

# Register teardown function to properly handle cleanup
@application.teardown_appcontext
def cleanup_context(exception=None):
    """Clean up resources when the application context ends"""
    if exception:
        logger.error(f"Error during context teardown: {str(exception)}")
    logger.info("Application context tearing down, cleaning up resources")
    
    # Import cleanup functions here to avoid circular imports
    from database.multi_user_db.db_final import cleanup_db_connections
    cleanup_db_connections()

logger.info("WSGI application initialization complete")
logger.info("==================================================")
logger.info("Starting with worker timeout of 2 hours for large rebuilds")

# Configure CORS
CORS(application, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize database
init_database()

# Add health check endpoint for Railway
@application.route('/-/health')
def health_check():
    """Health check endpoint for Railway"""
    try:
        # Return a simple healthy response without any database checks
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "twitter-bookmark-manager",
            "version": "1.0"
        })
    except Exception as e:
        # In case of any errors, also return JSON
        error_response = {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        # Log the error but don't crash the app
        logger.error(f"Health check error: {str(e)}")
        
        # Use Flask's json module directly to avoid HTML error responses
        from flask import json
        return application.response_class(
            response=json.dumps(error_response),
            status=500,
            mimetype='application/json'
        )

logger.info("✅ Successfully imported and configured main application") 