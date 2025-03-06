#!/usr/bin/env python
"""
Run the fix_categories_schema.py module to repair the database schema
Designed for Railway environment with appropriate error handling
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('run_fix_categories_schema')

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Check if running in Railway
is_railway = 'RAILWAY_PROJECT_ID' in os.environ
if is_railway:
    logger.info("Running in Railway environment")
else:
    logger.info("Running in local environment")

# Log available environment variables for database connection
db_vars = ["DATABASE_URL", "PGUSER", "PGPASSWORD", "PGHOST", "PGDATABASE", "PGPORT"]
logger.info("Available database environment variables:")
for var in db_vars:
    if var in os.environ:
        if "PASSWORD" in var or "URL" in var:
            logger.info(f"  {var}: [REDACTED]")
        else:
            logger.info(f"  {var}: {os.environ.get(var)}")
    else:
        logger.warning(f"  {var}: Not set")

# Import the script
try:
    sys.path.insert(0, current_dir)
    from fix_categories_schema import fix_schema
    
    # Run the schema fix with retry logic
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries} to fix categories schema")
        
        try:
            result = fix_schema()
            
            if result["success"]:
                logger.info("✅ Schema fix completed successfully")
                logger.info(result["message"])
                sys.exit(0)
            else:
                logger.error(f"❌ Schema fix failed: {result.get('error', 'Unknown error')}")
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Maximum retry attempts reached. Schema fix failed.")
                    sys.exit(1)
        
        except Exception as e:
            logger.error(f"❌ Exception during schema fix: {e}")
            
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("Maximum retry attempts reached. Schema fix failed.")
                sys.exit(1)
    
except ImportError as e:
    logger.error(f"Failed to import fix_categories_schema module: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    sys.exit(1) 