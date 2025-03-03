import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

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

# Import the Flask app
try:
    from auth.api_server_multi_user import app as application
    logger.info("✅ Successfully imported Flask application")
except Exception as e:
    logger.error(f"Error importing application: {e}")
    raise

logger.info("="*50) 