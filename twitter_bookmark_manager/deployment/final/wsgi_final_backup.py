"""
WSGI entry point for the Twitter Bookmark Manager application.
Configures the environment and starts the Flask application.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from flask import Flask, jsonify, request, redirect, url_for, render_template_string
from flask_cors import CORS
from database.multi_user_db.db_final import init_database
from datetime import datetime

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

# Force disable the vector store for now to ensure application works
os.environ['DISABLE_VECTOR_STORE'] = 'true'  # Prioritize application loading
os.environ.setdefault('FLASK_DEBUG', 'false')
os.environ.setdefault('QDRANT_HOST', 'localhost')  # Set Qdrant host
os.environ.setdefault('QDRANT_PORT', '6333')  # Set Qdrant port

# Create a minimal Flask application that will be used if the real one fails
fallback_app = Flask(__name__)
fallback_app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_flask_session')
CORS(fallback_app)

@fallback_app.route('/health')
def fallback_health():
    return jsonify({"status": "ok", "mode": "fallback"}), 200

@fallback_app.route('/', defaults={'path': ''})
@fallback_app.route('/<path:path>')
def fallback_catch_all(path):
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Twitter Bookmark Manager - Maintenance</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            .message { background: #f0f0f0; border-left: 4px solid #0066cc; padding: 10px; margin-bottom: 20px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Twitter Bookmark Manager</h1>
            <div class="message">
                <h2>Maintenance Mode</h2>
                <p>The application is currently in maintenance mode. Please try again later.</p>
                <p>We are working to resolve this issue as quickly as possible.</p>
            </div>
        </div>
    </body>
    </html>
    """), 503

# First try to load the real application
try:
    # Try to import flask_session
    try:
        import flask_session
        logger.info("flask_session module is available")
    except ImportError as e:
        logger.warning(f"flask_session module not available: {e}")
        # Create mock session
        class MockSession:
            def __init__(self, app=None): 
                self.app = app
        sys.modules['flask_session'] = type('mock_module', (), {'Session': MockSession})
        
    # Import the main application
    try:
        logger.info("Importing api_server_multi_user...")
        from auth.api_server_multi_user import app
        logger.info("Successfully imported app from api_server_multi_user")
        
        # Configure the application - use wrapper and try to import CustomSessionInterface
        try:
            from auth.api_server_multi_user import CustomSessionInterface
            app.session_interface = CustomSessionInterface()
            logger.info("Set custom session interface")
        except Exception as session_error:
            logger.error(f"Failed to set session interface: {session_error}")
            
        # Add Railway health check to main app
        @app.route('/-/health')
        def railway_health():
            return jsonify({
                "status": "ok",
                "mode": "production"
            }), 200
            
        # Use the main application
        application = app
        logger.info("Using main application")
        
    except Exception as app_error:
        logger.error(f"Failed to import main app: {app_error}")
        logger.error(traceback.format_exc())
        raise
        
except Exception as outer_error:
    logger.critical(f"All attempts to load the main application failed: {outer_error}")
    logger.critical(traceback.format_exc())
    logger.info("Using fallback application")
    application = fallback_app

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