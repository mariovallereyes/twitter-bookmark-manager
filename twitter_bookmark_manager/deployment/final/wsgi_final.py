import os
import sys
import logging
from flask import Flask, jsonify

# Set up simplified logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*50)
logger.info("Starting SIMPLIFIED WSGI Application for Railway")

# Create a basic application for testing
application = Flask(__name__)

@application.route('/')
def index():
    """Basic test route to confirm WSGI is working"""
    return """
    <html>
    <head>
        <title>Twitter Bookmark Manager - Test Page</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #1DA1F2; }
            .info { background: #f5f8fa; padding: 15px; border-radius: 5px; }
            .success { color: green; }
        </style>
    </head>
    <body>
        <h1>Twitter Bookmark Manager</h1>
        <div class="info">
            <h2 class="success">✅ Basic WSGI server is working!</h2>
            <p>This is a simplified test page to verify the server can start.</p>
            <h3>Environment Information:</h3>
            <ul>
                <li>Python Version: {}</li>
                <li>Environment: {}</li>
                <li>Working Directory: {}</li>
            </ul>
        </div>
    </body>
    </html>
    """.format(
        sys.version,
        os.environ.get('RAILWAY_ENVIRONMENT', 'unknown'),
        os.getcwd()
    )

@application.route('/api/test')
def api_test():
    """Test API endpoint"""
    return jsonify({
        "status": "success",
        "message": "API endpoint is responding",
        "environment": os.environ.get('RAILWAY_ENVIRONMENT', 'unknown')
    })

if __name__ == "__main__":
    # Only for local testing
    application.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

logger.info("Simplified WSGI application initialized")
logger.info("="*50) 