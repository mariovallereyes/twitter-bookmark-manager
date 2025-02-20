import sys
import os
from pathlib import Path

# Set up the working directory to be the project root
current_dir = Path(__file__).parent.resolve()

# Add both the root and the package directory to Python path
project_root = current_dir
package_dir = current_dir / 'twitter_bookmark_manager'

# Ensure both paths are in sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

# Ensure we're in the correct directory for database access
if os.path.exists(package_dir):
    os.chdir(package_dir)
else:
    # We might be in PythonAnywhere WSGI context
    os.chdir(current_dir)

from flask import Flask
from flask_cors import CORS
import logging

# Import the web application
from web.server import app as web_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Create the main app
app = web_app

# Configure CORS - Allow all origins during testing
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins during testing
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# PythonAnywhere-specific health check
@app.route('/api/pythonanywhere/health')
def pa_health_check():
    """PythonAnywhere-specific health check"""
    try:
        from core.search import BookmarkSearch
        search = BookmarkSearch()
        total_tweets = search.get_total_tweets()
        return {
            "status": "healthy",
            "environment": "pythonanywhere",
            "total_tweets": total_tweets,
            "version": "transitional-deployment",
            "cwd": os.getcwd()  # Add this for debugging
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}, 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    ) 