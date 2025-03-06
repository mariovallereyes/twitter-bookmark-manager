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
import threading
import time
from sqlalchemy.pool import QueuePool

# Configure logging
logger = logging.getLogger(__name__)

# Global database objects
_engine = None
_session_factory = None
_vector_store = None
_connection_monitor = None

def setup_database():
    """Set up the database connection."""
    global _engine, _session_factory, _vector_store
    
    try:
        logger.info("Loading PythonAnywhere database module")
        
        # Load environment variables - try multiple possible locations
        env_paths = [
            Path(os.path.join(os.path.dirname(__file__), ".env.final")).resolve(),
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
            logger.warning("No .env.final file found in any expected location, using environment variables only")
        
        # First check if DATABASE_URL is provided (Railway recommends this approach)
        DATABASE_URL = os.getenv("DATABASE_URL")
        if DATABASE_URL:
            logger.info("Using DATABASE_URL for connection")
            _engine = create_engine(
                DATABASE_URL,
                poolclass=QueuePool,
                pool_size=3,           # Reduced pool size to avoid too many connections
                max_overflow=5,        # Reduced max overflow to prevent connection buildup
                pool_timeout=10,       # Reduced timeout to fail faster
                pool_recycle=300,      # Recycle connections after 5 minutes
                pool_pre_ping=True,    # Verify connections before using them
                connect_args={
                    "connect_timeout": 5,          # Reduced connection timeout to fail faster
                    "application_name": "TwitterBookmarkManager",  # Helps identify connections in pg_stat_activity
                    "keepalives": 1,               # Enable TCP keepalives
                    "keepalives_idle": 30,         # Send keepalive packets after 30 seconds of inactivity
                    "keepalives_interval": 10,     # Resend keepalives every 10 seconds
                    "keepalives_count": 3,         # Consider connection dead after 3 failed keepalives
                    "tcp_user_timeout": 30000      # Abort connection if not established within 30 seconds
                }
            )
        else:
            # Fall back to individual components
            logger.info("DATABASE_URL not found, using individual connection parameters")
            # Get database connection settings with fallbacks
            DB_USER = os.getenv("DB_USER")
            DB_PASSWORD = os.getenv("DB_PASSWORD")
            DB_HOST = os.getenv("DB_HOST")
            DB_NAME = os.getenv("DB_NAME")
            DB_PORT = os.getenv("DB_PORT", "14374")  # Default PostgreSQL port for Railway
            
            # Log database connection settings (without password)
            logger.info(f"Database settings: USER={DB_USER}, HOST={DB_HOST}, NAME={DB_NAME}, PORT={DB_PORT}")
            
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
            
            # Create connection string from individual components
            DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=prefer"
            
            # Create database engine with PostgreSQL and connection pooling
            logger.info(f"Creating PostgreSQL engine with connection pooling")
            _engine = create_engine(
                DATABASE_URI,
                poolclass=QueuePool,
                pool_size=3,
                max_overflow=5,
                pool_timeout=10,
                pool_recycle=300,
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": 5,
                    "application_name": "TwitterBookmarkManager",
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 3,
                    "tcp_user_timeout": 30000
                }
            )
        
        logger.info("✅ Created PostgreSQL engine successfully")
        
        # Test the connection
        with _engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1
            logger.info("✅ Database connection test successful")
        
        # Create session factory
        _session_factory = scoped_session(sessionmaker(bind=_engine))
        logger.info("✅ Created session factory")
        
        # Start connection monitor
        start_connection_monitor()
        
        # Import VectorStore
        try:
            # Try to import using relative import first
            from .vector_store_final import VectorStore
            logger.info("✅ Successfully imported VectorStore")
            
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

def connection_monitor_thread():
    """
    Background thread that periodically checks database connection
    and keeps the connection pool active.
    """
    logger.info("Starting database connection monitor thread")
    while True:
        try:
            with _engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                is_healthy = result.scalar() == 1
                if is_healthy:
                    logger.debug("Database connection monitor: Connection healthy")
                else:
                    logger.warning("Database connection monitor: Connection test returned unexpected result")
        except Exception as e:
            logger.error(f"Database connection monitor: Connection test failed: {e}")
        
        # Sleep for 5 minutes before next check
        time.sleep(300)

def start_connection_monitor():
    """Start the background connection monitoring thread"""
    global _connection_monitor
    
    if _connection_monitor is None or not _connection_monitor.is_alive():
        _connection_monitor = threading.Thread(
            target=connection_monitor_thread,
            daemon=True,
            name="DBConnectionMonitor"
        )
        _connection_monitor.start()
        logger.info("Started database connection monitor thread")
    else:
        logger.debug("Connection monitor thread already running")

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
        from .vector_store_final import VectorStore
        _vector_store = VectorStore()
    return _vector_store

@contextmanager
def db_session() -> Generator:
    """Context manager for database sessions with retry logic and better error handling"""
    session = None
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            session = get_session()
            yield session
            
            # Try to commit changes
            try:
                session.commit()
                logger.debug("Session committed successfully")
                break  # Success, exit the retry loop
            except Exception as commit_error:
                session.rollback()
                logger.warning(f"Commit failed (attempt {retry_count+1}/{max_retries}): {commit_error}")
                last_error = commit_error
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to commit after {max_retries} attempts: {commit_error}")
                    raise
                # Short delay before retry
                time.sleep(0.5 * retry_count)  # Exponential backoff
                
        except Exception as session_error:
            last_error = session_error
            logger.error(f"Session error (attempt {retry_count+1}/{max_retries}): {session_error}")
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to execute session after {max_retries} attempts: {session_error}")
                raise
            # Short delay before retry
            time.sleep(0.5 * retry_count)  # Exponential backoff
            
        finally:
            if session:
                try:
                    session.close()
                    logger.debug("Session closed")
                except Exception as close_error:
                    logger.warning(f"Error closing session: {close_error}")
                    # Don't raise here, as we may have already succeeded with the transaction

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
    """Check database connection status and configuration"""
    try:
        # Testing the database connection
        engine = get_engine()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            connection_ok = result.scalar() == 1
            logger.info("✅ Database connection test: SUCCESS")
            
        return {
            "database_connection": "ok" if connection_ok else "error",
            "database_type": "PostgreSQL",
            "database_url": "Using DATABASE_URL environment variable" if os.getenv("DATABASE_URL") else "Not using DATABASE_URL",
            "connection_string": "Configured" if _engine else "Not configured",
            "environment_variables": {
                "DB_HOST": os.getenv("DB_HOST", "Not set"),
                "DB_NAME": os.getenv("DB_NAME", "Not set"),
                "DB_USER": os.getenv("DB_USER", "Not set"),
                "DB_PASSWORD": "***" if os.getenv("DB_PASSWORD") else "Not set"
            }
        }
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "database_connection": "error",
            "error_message": str(e),
            "environment_variables": {
                "DB_HOST": os.getenv("DB_HOST", "Not set"),
                "DB_NAME": os.getenv("DB_NAME", "Not set"),
                "DB_USER": os.getenv("DB_USER", "Not set"),
                "DB_PASSWORD": "***" if os.getenv("DB_PASSWORD") else "Not set"
            }
        }

def get_db_url():
    """
    Get the database URL for direct connection.
    This is used by diagnostic tools to inspect the database.
    
    Returns:
        str: The database connection URL
    """
    # First check if DATABASE_URL is provided (Railway recommends this approach)
    DATABASE_URL = os.getenv("DATABASE_URL")
    if DATABASE_URL:
        return DATABASE_URL
    else:
        # Fall back to individual components
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_NAME = os.getenv("DB_NAME")
        
        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            missing = []
            if not DB_USER: missing.append("DB_USER")
            if not DB_PASSWORD: missing.append("DB_PASSWORD")
            if not DB_HOST: missing.append("DB_HOST")
            if not DB_NAME: missing.append("DB_NAME")
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:14374/{DB_NAME}?sslmode=prefer"

def get_db_connection():
    """
    Get a PostgreSQL database connection.
    This function is used by api_server_multi_user.py and other modules.
    
    Returns:
        A database session that should be closed when done.
    """
    return get_session()

def get_db_connection_with_vector_store():
    """
    Get a PostgreSQL database connection with vector store for embeddings.
    Used as a replacement for get_db_connection in the main app.
    """
    conn = get_db_connection()
    
    # Initialize vector store if needed 
    from .vector_store_final import VectorStore
    get_vector_store()
    
    return conn 

def create_tables():
    """
    Create all database tables using SQLAlchemy models.
    This is a wrapper around Base.metadata.create_all() that ensures
    all models are imported and available.
    """
    logger.info("Creating all database tables using SQLAlchemy...")
    
    # Import models to ensure they're registered with the Base
    try:
        from database.multi_user_db.models_final import Base
        from database.multi_user_db.user_model_final import (
            User, create_user_table, create_system_user_if_needed, 
            reset_user_id_sequence
        )
        
        # Get the engine
        engine = get_engine()
        
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("✅ All tables created successfully")
        
        # Initialize the users table and system user
        conn = get_db_connection()
        try:
            # Create user table if it doesn't exist
            create_user_table(conn)
            
            # Create system user (id=1) if needed
            create_system_user_if_needed(conn)
            
            # Explicitly reset the users_id_seq sequence
            reset_user_id_sequence(conn)
            
            logger.info("✅ User tables and system user initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing user table: {str(e)}")
        finally:
            conn.close()
            
        return True
    except Exception as e:
        logger.error(f"❌ Error creating tables: {str(e)}")
        logger.error(traceback.format_exc())
        return False 