"""
Database schema update script for PythonAnywhere deployment.
Ensures that necessary tables and columns exist for category functionality.
"""
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import Column, DateTime, Boolean, text
from sqlalchemy.exc import OperationalError

# Set up PythonAnywhere paths
PA_BASE_DIR = '/home/mariovallereyes/twitter_bookmark_manager'
if PA_BASE_DIR not in sys.path:
    sys.path.insert(0, PA_BASE_DIR)

# Import from main codebase
from twitter_bookmark_manager.config.constants import BOOKMARK_CATEGORIES
from twitter_bookmark_manager.deployment.pythonanywhere.database.db_pa import get_session, get_engine
from twitter_bookmark_manager.database.models import Bookmark, Category, bookmark_categories
from dotenv import load_dotenv

# Load environment variables
env_paths = [
    Path('/home/mariovallereyes/twitter_bookmark_manager/.env.pythonanywhere').resolve(),
    Path(__file__).parents[3] / '.env.pythonanywhere'
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

# Configure logging
LOG_DIR = os.path.join(PA_BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'db_schema_update.log')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_database_schema():
    """Update database schema for category functionality"""
    try:
        logger.info("="*50)
        logger.info("Starting database schema update for categories")
        
        engine = get_engine()
        updates_applied = []
        
        # Ensure tables exist (will not recreate if they already exist)
        from twitter_bookmark_manager.database.models import Base
        Base.metadata.create_all(engine)
        logger.info("✅ Ensured all base tables exist")
        
        # Add categorized_at column to bookmarks if it doesn't exist
        column_exists = False
        with get_session() as session:
            try:
                # Check if column exists
                session.query(Bookmark.categorized_at).limit(1).all()
                logger.info("✅ categorized_at column already exists")
                column_exists = True
            except:
                # Add column using the SQLAlchemy 2.0 compatible syntax
                logger.info("Adding categorized_at column to bookmarks table")
                try:
                    # Create a connection and execute the SQL directly
                    with engine.connect() as conn:
                        conn.execute(text('ALTER TABLE bookmarks ADD COLUMN categorized_at TIMESTAMP'))
                        conn.commit()
                    updates_applied.append("Added categorized_at column to bookmarks table")
                    logger.info("✅ Added categorized_at column")
                except Exception as e:
                    logger.error(f"Error adding column: {e}")
                    raise
        
        # Ensure all standard categories exist
        with get_session() as session:
            for category_info in BOOKMARK_CATEGORIES:
                category_name = category_info['name']
                category = session.query(Category).filter_by(name=category_name).first()
                
                if not category:
                    category = Category(name=category_name)
                    session.add(category)
                    updates_applied.append(f"Added category: {category_name}")
                    logger.info(f"✅ Added missing category: {category_name}")
            
            session.commit()
            logger.info("✅ Ensured all standard categories exist")
        
        # Check for Gemini API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("⚠️ GEMINI_API_KEY not found in environment variables")
            logger.warning("Category processing will fail without this key")
        else:
            logger.info("✅ GEMINI_API_KEY found in environment variables")
        
        # Get current stats
        with get_session() as session:
            total_bookmarks = session.query(Bookmark).count()
            total_categories = session.query(Category).count()
            bookmarks_with_categories = session.query(Bookmark) \
                .join(bookmark_categories) \
                .distinct(Bookmark.id) \
                .count()
                
            logger.info(f"Database stats: {total_bookmarks} bookmarks, {total_categories} categories")
            logger.info(f"Bookmarks with categories: {bookmarks_with_categories}/{total_bookmarks}")
            
        if updates_applied:
            logger.info(f"Applied {len(updates_applied)} schema updates:")
            for update in updates_applied:
                logger.info(f"  - {update}")
        else:
            logger.info("No schema updates were necessary, database is up to date")
            
        return {
            "success": True,
            "updates_applied": updates_applied,
            "message": "Database schema updated successfully for category functionality"
        }
            
    except Exception as e:
        logger.error(f"Error updating database schema: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    result = update_database_schema()
    print(f"Schema update result: {result}") 