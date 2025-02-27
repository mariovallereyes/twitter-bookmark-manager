#!/usr/bin/env python3
"""
PythonAnywhere Task Script for Category Processing

This script is meant to be scheduled in PythonAnywhere to run at regular intervals
(e.g., every 2-4 hours) to process uncategorized bookmarks.

Usage in PythonAnywhere:
1. Create a scheduled task in the PythonAnywhere dashboard
2. Use the command: python3 /home/yourusername/twitter_bookmark_manager/process_categories_task.py
"""

import os
import sys
import logging
from datetime import datetime
import traceback

# Configure paths for PythonAnywhere
PA_BASE_DIR = os.getenv('PA_BASE_DIR', '/home/mariovallereyes/twitter_bookmark_manager')
project_root = os.path.abspath(PA_BASE_DIR)

# Add project directory to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set PythonAnywhere environment flag
os.environ['PYTHONANYWHERE_ENVIRONMENT'] = 'true'

# Configure logging with absolute PythonAnywhere paths
LOG_DIR = os.path.join(PA_BASE_DIR, 'logs')
TASK_LOG_FILE = os.path.join(LOG_DIR, 'category_task.log')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(TASK_LOG_FILE, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the category processing job"""
    start_time = datetime.now()
    logger.info("="*60)
    logger.info(f"Starting category processing task at {start_time.isoformat()}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")

    try:
        # Import the processor function - use absolute import to avoid path issues
        from twitter_bookmark_manager.deployment.pythonanywhere.database.process_categories_pa import process_categories_background_job
        
        # Run the job
        result = process_categories_background_job()
        
        if result.get('success'):
            logger.info(f"✅ Successfully processed {result['processed']} bookmarks")
            
            # Get current stats
            if 'stats' in result:
                stats = result['stats']
                logger.info(f"📊 Categorization stats:")
                logger.info(f"   - Total bookmarks: {stats['total_bookmarks']}")
                logger.info(f"   - Categorized: {stats['categorized_count']} ({stats['completion_percentage']}%)")
                logger.info(f"   - Uncategorized: {stats['uncategorized_count']}")
                logger.info(f"   - Total categories: {stats['total_categories']}")
        else:
            logger.error(f"❌ Error in processing: {result.get('error', 'Unknown error')}")
    
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error(f"This might indicate the script is not being run from the correct directory.")
        logger.error(f"Try adjusting your PYTHONPATH or moving the script to the correct location.")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"❌ Unhandled exception: {e}")
        logger.error(traceback.format_exc())
    
    # Log execution time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Task finished in {duration:.2f} seconds")
    logger.info("="*60)

if __name__ == "__main__":
    main() 