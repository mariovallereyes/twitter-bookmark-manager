import sys
import os
import logging
from pathlib import Path
from flask import Flask, redirect, url_for

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

# Change working directory
os.chdir(current_dir)
logger.info(f"Changed working directory to: {os.getcwd()}")

# Create a Flask app directly here to avoid import issues
try:
    # Create Flask app
    application = Flask(__name__, 
                      template_folder='web_final/templates',
                      static_folder='web_final/static')
    
    # Root route for testing
    @application.route('/')
    def index():
        return "<h1>Twitter Bookmark Manager</h1><p>The application is starting up. Please check back soon.</p>"
    
    logger.info("✅ Successfully created basic Flask application")
except Exception as e:
    logger.error(f"Error creating application: {e}")
    raise

logger.info("="*50) 