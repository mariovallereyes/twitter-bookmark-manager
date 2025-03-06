import os
import sys
import logging
import traceback  # Added for error tracing
from flask import Flask, jsonify, request

# Set up simplified logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*50)
logger.info("Starting SUPER BASIC WSGI Application for Railway")

# Create a basic application for testing
application = Flask(__name__)
application.config['DEBUG'] = True  # Force debug mode

# Add error handling for all exceptions
@application.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exception"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return f"""
    <html>
    <head>
        <title>Error - Twitter Bookmark Manager</title>
    </head>
    <body>
        <h1>Error Detected</h1>
        <p>The application encountered an error:</p>
        <pre>{str(e)}</pre>
        <h2>Traceback:</h2>
        <pre>{traceback.format_exc()}</pre>
        <h2>Request Details:</h2>
        <pre>
Path: {request.path}
Method: {request.method}
Headers: {dict(request.headers)}
        </pre>
        <h2>Environment:</h2>
        <pre>
Python Version: {sys.version}
WSGI Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'unknown')}
Working Directory: {os.getcwd()}
Imported Modules: {list(sys.modules.keys())}
        </pre>
    </body>
    </html>
    """

@application.route('/')
def index():
    """Basic test route with NO CSS to avoid formatting issues"""
    return """
    <html>
    <head>
        <title>Twitter Bookmark Manager - Test Page</title>
    </head>
    <body>
        <h1>Twitter Bookmark Manager</h1>
        <div>
            <h2>WSGI server is working!</h2>
            <p>This is a simplified test page with NO CSS to avoid string formatting issues.</p>
            <h3>Environment Information:</h3>
            <ul>
                <li>Python Version: """ + sys.version + """</li>
                <li>Environment: """ + os.environ.get('RAILWAY_ENVIRONMENT', 'unknown') + """</li>
                <li>Working Directory: """ + os.getcwd() + """</li>
            </ul>
            <p><a href="/api/test">Test API Endpoint</a></p>
        </div>
    </body>
    </html>
    """

@application.route('/api/test')
def api_test():
    """Test API endpoint"""
    return jsonify({
        "status": "success",
        "message": "API endpoint is responding",
        "environment": os.environ.get('RAILWAY_ENVIRONMENT', 'unknown')
    })

# Log basic system information
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"System path: {sys.path}")
logger.info(f"Imported modules: {', '.join(list(sys.modules.keys())[:20])}...")

# Print Flask app config for debugging
logger.info(f"Flask app config: {application.config}")

if __name__ == "__main__":
    # Only for local testing
    application.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

logger.info("Super basic WSGI application initialized")
logger.info("="*50) 