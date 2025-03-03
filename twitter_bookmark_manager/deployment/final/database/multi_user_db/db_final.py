"""
PythonAnywhere-specific database module to override the SQLite implementation
with PostgreSQL and avoid ChromaDB imports.
"""
import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from typing import Generator
from dotenv import load_dotenv
from pathlib import Path
import importlib.util
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Global database objects
_engine = None
_session_factory = None
_vector_store = None

def setup_database():
    """Set up PostgreSQL database connection for PythonAnywhere"""
    global _engine, _session_factory, _vector_store
    
    try:
        logger.info("Loading PythonAnywhere database module")
        
        # Load environment variables - try multiple possible locations
        env_paths = [
            Path('/home/mariovallereyes/twitter_bookmark_manager/.env.final').resolve(),
            Path(__file__).parents[3] / '.env.final'
        ]
        
        env_loaded = False
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path, override=True)
                logger.info(f"✅ Loaded environment variables from {env_path}")
                env_loaded = True
                break
        
        if not env_loaded:
            logger.error("❌ No .env.final file found in any expected location")
            
        # Get database connection settings with fallbacks
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_NAME = os.getenv("DB_NAME")
        
        # Log database connection settings (without password)
        logger.info(f"Database settings: USER={DB_USER}, HOST={DB_HOST}, NAME={DB_NAME}")
        
        # Check if we have all required environment variables
        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            missing = []
            if not DB_USER: missing.append("DB_USER")
            if not DB_PASSWORD: missing.append("DB_PASSWORD")
            if not DB_HOST: missing.append("DB_HOST")
            if not DB_NAME: missing.append("DB_NAME")
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Create database engine with PostgreSQL
        logger.info(f"Creating PostgreSQL engine")
        DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:14374/{DB_NAME}?sslmode=prefer"
        _engine = create_engine(DATABASE_URI)
        logger.info(f"✅ Created PostgreSQL engine successfully")
        
        # Create session factory
        _session_factory = scoped_session(sessionmaker(bind=_engine))
        logger.info("✅ Created session factory")
        
        # Import VectorStore
        try:
            # Try to import using relative import first
            try:
                from .vector_store_pa import VectorStore
                logger.info("✅ Successfully imported VectorStore using relative import")
            except ImportError:
                # Fallback to absolute import
                from deployment.final.database.multi_user_db.vector_store_final import VectorStore
                logger.info("✅ Successfully imported VectorStore using absolute import")
            
            # For compatibility with original code expecting ChromaStore
            ChromaStore = VectorStore
            logger.info("✅ Aliased VectorStore as ChromaStore for compatibility")
            
            return {
                'engine': _engine,
                'session_factory': _session_factory,
                'VectorStore': VectorStore,
                'ChromaStore': ChromaStore
            }
        except Exception as e:
            logger.error(f"❌ Failed to import VectorStore: {e}")
            raise
            
    except Exception as e:
        logger.error(f"❌ Error setting up database: {e}")
        logger.error(traceback.format_exc())
        raise

# Initialize database connection on module import
if _engine is None:
    try:
        db_objects = setup_database()
        # These are already set as globals in the setup_database function
    except Exception as e:
        logger.error(f"❌ Failed to initialize database connection: {e}")
        # Don't raise here to avoid breaking imports, let individual functions handle errors

def get_engine():
    """Get SQLAlchemy engine"""
    global _engine
    if _engine is None:
        db_objects = setup_database()  # Try to set up again if not already done
    return _engine

def get_session():
    """Get a new SQLAlchemy session"""
    global _session_factory
    if _session_factory is None:
        db_objects = setup_database()  # Try to set up again if not already done
    return _session_factory()

# Alias for backward compatibility
get_db_session = get_session

def get_vector_store():
    """Get the vector store singleton for embeddings"""
    global _vector_store
    if _vector_store is None:
        from deployment.final.database.multi_user_db.vector_store_final import VectorStore
        _vector_store = VectorStore()
    return _vector_store

@contextmanager
def db_session() -> Generator:
    """Context manager for database sessions"""
    session = get_session()
    try:
        yield session
        session.commit()
        logger.debug("Session committed successfully")
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error, rolling back: {e}")
        raise
    finally:
        session.close()
        logger.debug("Session closed")

def init_database():
    """Verify database connection for PythonAnywhere and initialize vector store"""
    try:
        # Test database connection
        with db_session() as session:
            session.execute(text("SELECT 1"))
        logger.info("✅ PostgreSQL database connection verified")
        
        # Test vector store initialization
        vector_store = get_vector_store()
        logger.info("✅ Vector store initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize database system: {e}")
        logger.error(traceback.format_exc())
        return False

def check_database_status():
    """Check database status and return counts of records"""
    try:
        session = get_session()
        
        # First just test connection
        session.execute(text("SELECT 1"))
        logger.info("✅ Database connection test successful")
        
        # Check categories
        categories_count = session.execute(text("SELECT COUNT(*) FROM categories")).scalar()
        
        # Check bookmarks
        bookmarks_count = session.execute(text("SELECT COUNT(*) FROM bookmarks")).scalar()
        
        # Check bookmark categories
        bookmark_categories_count = session.execute(text("SELECT COUNT(*) FROM bookmark_categories")).scalar()
        
        # Get list of category names
        category_names = session.execute(text("SELECT name FROM categories")).fetchall()
        category_names = [row[0] for row in category_names]
        
        # Close session
        session.close()
        
        return {
            "database_connection": "success",
            "categories_count": categories_count,
            "bookmarks_count": bookmarks_count,
            "bookmark_categories_count": bookmark_categories_count,
            "category_names": category_names,
            "environment_check": {
                "DB_HOST": os.getenv("DB_HOST", "Not set"),
                "DB_NAME": os.getenv("DB_NAME", "Not set"),
                "DB_USER": os.getenv("DB_USER", "Not set"),
                "DB_PASSWORD": "***" if os.getenv("DB_PASSWORD") else "Not set"
            }
        }
    except Exception as e:
        logger.error(f"Error checking database status: {e}")
        logger.error(traceback.format_exc())
        return {
            "database_connection": "error",
            "error_message": str(e),
            "environment_check": {
                "DB_HOST": os.getenv("DB_HOST", "Not set"),
                "DB_NAME": os.getenv("DB_NAME", "Not set"),
                "DB_USER": os.getenv("DB_USER", "Not set"),
                "DB_PASSWORD": "***" if os.getenv("DB_PASSWORD") else "Not set"
            }
        }

def get_db_connection_with_vector_store():
    """
    Get a PostgreSQL database connection with vector store for embeddings.
    Used as a replacement for get_db_connection in the main app.
    """
    conn = get_db_connection()
    
    # Initialize vector store if needed 
    from deployment.final.database.multi_user_db.vector_store_final import VectorStore
    get_vector_store()
    
    return conn 