"""
WSGI entry point with emergency fallback for Twitter Bookmark Manager.
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from flask import Flask, jsonify, render_template_string

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler('wsgi.log')  # File handler
    ]
)
logger = logging.getLogger('wsgi_final')

# Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment from .env file")
except ImportError:
    logger.warning("python-dotenv not installed, using environment variables as is")

# Log Twitter authentication variables (securely)
logger.info("Twitter authentication configuration:")
logger.info(f"TWITTER_CLIENT_ID present: {bool(os.environ.get('TWITTER_CLIENT_ID'))}")
logger.info(f"TWITTER_CLIENT_SECRET present: {bool(os.environ.get('TWITTER_CLIENT_SECRET'))}")
logger.info(f"TWITTER_REDIRECT_URI: {os.environ.get('TWITTER_REDIRECT_URI', 'Not set')}")

# Create the emergency application first so it's available if needed
emergency_app = Flask(__name__)

@emergency_app.route('/')
def emergency_index():
    """Emergency home page."""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Twitter Bookmark Manager - Fallback Mode</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            .message { background: #f4f4f4; border-left: 4px solid #0066cc; padding: 10px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Twitter Bookmark Manager</h1>
            <div class="message">
                <h2>Fallback Mode Active</h2>
                <p>The application is running in fallback mode due to startup issues.</p>
                <p>Full functionality is temporarily disabled.</p>
            </div>
        </div>
    </body>
    </html>
    """)

@emergency_app.route('/-/health')
def emergency_health():
    """Health check endpoint for Railway."""
    return jsonify({"status": "ok", "mode": "fallback"}), 200

@emergency_app.route('/<path:path>')
def emergency_catch_all(path):
    """Catch all routes in emergency mode."""
    logger.info(f"Emergency catch-all accessed: {path}")
    return emergency_index()

# Try to load the real application with careful error handling
try:
    logger.info("Attempting to load the main application...")
    
    # Check if we need the flask_session module
    try:
        import flask_session
        logger.info("flask_session module is available")
    except ImportError as e:
        logger.warning(f"flask_session module not available: {e}")
        # Create mock session to prevent errors
        class MockSession:
            def __init__(self, app=None):
                self.app = app
        sys.modules['flask_session'] = type('mock_module', (), {'Session': MockSession})
    
    # Check auth modules first to make sure they're available
    try:
        logger.info("Checking auth modules...")
        
        # Check auth_routes_final
        try:
            from auth.auth_routes_final import auth_bp
            logger.info("auth_routes_final module loaded successfully")
        except Exception as auth_routes_error:
            logger.error(f"Error loading auth_routes_final: {auth_routes_error}")
            logger.error(traceback.format_exc())
        
        # Check user_context_final
        try:
            from auth.user_context_final import UserContext, login_required, UserContextMiddleware
            logger.info("user_context_final module loaded successfully")
        except Exception as user_context_error:
            logger.error(f"Error loading user_context_final: {user_context_error}")
            logger.error(traceback.format_exc())
        
        # Check oauth_final
        try:
            from auth.oauth_final import OAuthManager
            logger.info("oauth_final module loaded successfully")
        except Exception as oauth_error:
            logger.error(f"Error loading oauth_final: {oauth_error}")
            logger.error(traceback.format_exc())
            
    except Exception as auth_module_error:
        logger.error(f"Error checking auth modules: {auth_module_error}")
        logger.error(traceback.format_exc())
        # Continue anyway - we'll try to load the main app
    
    # Try to import database module to initialize it
    try:
        from database.multi_user_db.db_final import init_database, get_db_connection
        # Initialize database
        init_database()
        logger.info("Database initialized successfully")
        
        # Check user tables
        try:
            from database.multi_user_db.user_model_final import create_user_table
            conn = get_db_connection()
            create_user_table(conn)
            logger.info("User tables initialized successfully")
        except Exception as user_error:
            logger.error(f"Error initializing user tables: {user_error}")
            logger.error(traceback.format_exc())
            
    except Exception as db_error:
        logger.error(f"Database initialization error: {db_error}")
        logger.error(traceback.format_exc())
        # Continue anyway - database errors shouldn't prevent app loading
    
    # Import the main application with careful error handling
    try:
        from auth.api_server_multi_user import app
        
        # Import CustomSessionInterface if available
        try:
            from auth.api_server_multi_user import CustomSessionInterface
            app.session_interface = CustomSessionInterface()
            logger.info("Custom session interface set")
        except Exception as session_error:
            logger.error(f"Error setting session interface: {session_error}")
        
        # Add Railway health check to the main app
        @app.route('/-/health')
        def railway_health():
            return jsonify({
                "status": "ok",
                "mode": "basic",
                "timestamp": datetime.now().isoformat()
            }), 200
        
        # Use the imported application
        application = app
        logger.info("Successfully loaded main application")
    except Exception as app_error:
        logger.error(f"Error loading main application: {app_error}")
        logger.error(traceback.format_exc())
        # Fall back to emergency app
        raise
        
except Exception as outer_error:
    logger.critical(f"Failed to load main application, using emergency app: {outer_error}")
    # Use the emergency application
    application = emergency_app
    logger.info("Initialized emergency application as fallback")

logger.info("WSGI application initialization complete") 