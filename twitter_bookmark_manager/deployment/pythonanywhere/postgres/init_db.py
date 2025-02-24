"""
Initialize PostgreSQL database and migrate data from SQLite.
This script is specific to PythonAnywhere deployment and won't affect local development.
"""
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pg_migration_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
PA_BASE_DIR = '/home/mariovallereyes/twitter_bookmark_manager'
project_root = Path(PA_BASE_DIR)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
env_path = project_root / '.env.pythonanywhere'
if env_path.exists():
    load_dotenv(env_path, override=True)
    logger.info(f"Loaded environment from {env_path}")

# Import our modules
from database.models import Base, Bookmark
from deployment.pythonanywhere.postgres.config import get_database_url, VECTOR_STORE_CONFIG
from deployment.pythonanywhere.database.vector_store_pa import VectorStore
from deployment.pythonanywhere.postgres.migrate_schema import migrate_schema

def init_postgres_db():
    """Initialize PostgreSQL database schema"""
    try:
        logger.info("Initializing PostgreSQL database...")
        
        # First run our PostgreSQL-specific schema migration
        migrate_schema()
        logger.info("Schema migration completed")
        
        # Get engine for further operations
        engine = create_engine(get_database_url())
        logger.info("Database engine initialized")
        
        return engine
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def init_vector_store():
    """Initialize vector store"""
    try:
        logger.info("Initializing vector store...")
        vector_store = VectorStore()
        logger.info("Vector store initialized successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

def migrate_from_sqlite():
    """Migrate data from SQLite to PostgreSQL"""
    try:
        logger.info("Starting migration from SQLite to PostgreSQL...")
        
        # Connect to PostgreSQL
        pg_engine = create_engine(get_database_url())
        PgSession = sessionmaker(bind=pg_engine)
        pg_session = PgSession()
        
        # Connect to SQLite (source)
        sqlite_path = project_root / 'database' / 'bookmarks.db'
        if not sqlite_path.exists():
            logger.warning("SQLite database not found, skipping migration")
            return
        
        sqlite_url = f'sqlite:///{sqlite_path}'
        sqlite_engine = create_engine(sqlite_url)
        SqliteSession = sessionmaker(bind=sqlite_engine)
        sqlite_session = SqliteSession()
        
        try:
            # Get all bookmarks from SQLite
            sqlite_bookmarks = sqlite_session.query(Bookmark).all()
            total = len(sqlite_bookmarks)
            logger.info(f"Found {total} bookmarks in SQLite")
            
            # Initialize vector store
            vector_store = init_vector_store()
            
            # Migrate in batches
            batch_size = 50
            for i, bookmark in enumerate(sqlite_bookmarks, 1):
                try:
                    # Create new bookmark in PostgreSQL
                    new_bookmark = Bookmark(
                        id=bookmark.id,
                        text=bookmark.text,
                        author_name=bookmark.author_name,
                        author_username=bookmark.author_username,
                        created_at=bookmark.created_at,
                        raw_data=bookmark.raw_data,
                        categories=bookmark.categories
                    )
                    pg_session.merge(new_bookmark)
                    
                    # Update vector store
                    if bookmark.text:
                        vector_store.add_bookmark(
                            bookmark_id=str(bookmark.id),
                            text=bookmark.text,
                            metadata={
                                'tweet_url': bookmark.raw_data.get('tweet_url', ''),
                                'screen_name': bookmark.author_username or '',
                                'author_name': bookmark.author_name or ''
                            }
                        )
                    
                    # Commit in batches
                    if i % batch_size == 0:
                        pg_session.commit()
                        logger.info(f"Migrated {i}/{total} bookmarks")
                    
                except Exception as e:
                    logger.error(f"Error migrating bookmark {bookmark.id}: {e}")
                    continue
            
            # Final commit
            pg_session.commit()
            logger.info("Migration completed successfully!")
            
        finally:
            sqlite_session.close()
            pg_session.close()
            
    except Exception as e:
        logger.error(f"Migration error: {e}")
        raise

def verify_migration():
    """Verify the migration was successful"""
    try:
        logger.info("Verifying migration...")
        
        # Connect to both databases
        pg_engine = create_engine(get_database_url())
        PgSession = sessionmaker(bind=pg_engine)
        pg_session = PgSession()
        
        sqlite_path = project_root / 'database' / 'bookmarks.db'
        sqlite_url = f'sqlite:///{sqlite_path}'
        sqlite_engine = create_engine(sqlite_url)
        SqliteSession = sessionmaker(bind=sqlite_engine)
        sqlite_session = SqliteSession()
        
        try:
            # Compare record counts
            sqlite_count = sqlite_session.query(Bookmark).count()
            pg_count = pg_session.query(Bookmark).count()
            
            logger.info(f"SQLite records: {sqlite_count}")
            logger.info(f"PostgreSQL records: {pg_count}")
            
            if sqlite_count == pg_count:
                logger.info("✓ Record counts match")
            else:
                logger.warning(f"! Record count mismatch: SQLite={sqlite_count}, PostgreSQL={pg_count}")
            
            # Verify vector store
            vector_store = init_vector_store()
            vector_info = vector_store.get_collection_info()
            vector_count = vector_info.get('vectors_count', 0)
            logger.info(f"Vector store records: {vector_count}")
            
            return {
                'sqlite_count': sqlite_count,
                'pg_count': pg_count,
                'vector_count': vector_count,
                'success': sqlite_count == pg_count
            }
            
        finally:
            sqlite_session.close()
            pg_session.close()
            
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise

if __name__ == "__main__":
    try:
        # Initialize PostgreSQL
        init_postgres_db()
        
        # Initialize vector store
        init_vector_store()
        
        # Migrate data
        migrate_from_sqlite()
        
        # Verify migration
        results = verify_migration()
        
        if results['success']:
            logger.info("Migration completed and verified successfully!")
            sys.exit(0)
        else:
            logger.error("Migration verification failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1) 