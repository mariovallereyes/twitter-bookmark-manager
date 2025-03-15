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

# Disable vector store and other potentially problematic features
os.environ['DISABLE_VECTOR_STORE'] = 'true'
os.environ['FLASK_DEBUG'] = 'false'

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
    
    # Try to import database module first to initialize it
    try:
        from database.multi_user_db.db_final import init_database
        # Initialize database
        init_database()
        logger.info("Database initialized successfully")
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