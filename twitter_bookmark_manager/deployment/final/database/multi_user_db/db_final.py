"""
PythonAnywhere-specific database module to override the SQLite implementation
with PostgreSQL and avoid ChromaDB imports.
"""
import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from contextlib import contextmanager
from typing import Generator
from dotenv import load_dotenv
from pathlib import Path
import importlib.util
import traceback
import threading
import time
import random
import gc
from sqlalchemy.pool import QueuePool, NullPool

# Configure logging
logger = logging.getLogger(__name__)

# Global database objects
_engine = None
_session_factory = None
_vector_store = None
_connection_monitor = None
_last_memory_usage = "0MB"

# Simple retry decorator for database operations
def with_retries(max_attempts=2, backoff_factor=0.3):
    """Decorator to retry database operations that fail due to connection issues"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # Only retry on connection-related errors
                    if any(err in str(e).lower() for err in 
                           ['connection', 'reset', 'broken pipe', 'timeout']):
                        logger.warning(f"Connection error in {func.__name__}, attempt {attempt+1}/{max_attempts}: {e}")
                        if attempt < max_attempts - 1:
                            # Add jitter to prevent thundering herd
                            sleep_time = backoff_factor * (2 ** attempt) * (0.8 + 0.4 * random.random())
                            time.sleep(sleep_time)
                            # Try to recreate engine on connection failures
                            global _engine
                            try:
                                if _engine:
                                    _engine.dispose()
                                setup_database()
                            except:
                                pass
                        else:
                            logger.error(f"Failed after {max_attempts} attempts: {e}")
                            raise
                    else:
                        # Don't retry on non-connection errors
                        logger.error(f"Non-connection error in {func.__name__}: {e}")
                        raise
            raise last_exception
        return wrapper
    return decorator

def get_memory_usage():
    """Get current memory usage of the process"""
    global _last_memory_usage
    try:
        # Cross-platform memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        _last_memory_usage = f"{memory_mb:.1f}MB"
        return _last_memory_usage
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
        # Try using platform-specific approach as fallback
        try:
            if sys.platform == 'linux':
                memory_usage = os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()
                memory_mb = float(memory_usage) / 1024
                _last_memory_usage = f"{memory_mb:.1f}MB"
                return _last_memory_usage
        except Exception:
            pass
        # Return last known value if we can't update
        return _last_memory_usage

def setup_database():
    """Set up the database connection with more aggressive connection settings."""
    global _engine, _session_factory, _vector_store
    
    try:
        logger.info("Loading database module with aggressive connection settings")
        
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
        
        # More aggressive connection pool settings
        pool_size = 3  # Smaller pool size
        max_overflow = 5  # Less overflow
        pool_timeout = 10  # Shorter timeout
        pool_recycle = 300  # Recycle connections more frequently
        connect_timeout = 5  # Shorter connect timeout
        
        # First check if DATABASE_URL is provided (Railway recommends this approach)
        DATABASE_URL = os.getenv("DATABASE_URL")
        if DATABASE_URL:
            logger.info("Using DATABASE_URL for connection with aggressive settings")
            _engine = create_engine(
                DATABASE_URL,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": connect_timeout,
                    "application_name": "TwitterBookmarkManager",
                    # TCP keepalive settings to detect stale connections earlier
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 3
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
            
            # Create database engine with PostgreSQL and aggressive connection settings
            logger.info(f"Creating PostgreSQL engine with aggressive connection settings")
            _engine = create_engine(
                DATABASE_URI,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": connect_timeout,
                    "application_name": "TwitterBookmarkManager",
                    # TCP keepalive settings to detect stale connections earlier
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 3
                }
            )
        
        logger.info("✅ Created PostgreSQL engine successfully with aggressive settings")
        
        # Test the connection
        with _engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1
            logger.info("✅ Database connection test successful")
        
        # Create session factory with standard settings
        _session_factory = scoped_session(sessionmaker(
            bind=_engine,
            # Expire on commit for more predictable behavior
            expire_on_commit=True
        ))
        logger.info("✅ Created session factory")
        
        # Start connection monitor with standard interval
        start_connection_monitor()
        
        # Log current memory usage
        mem_usage = get_memory_usage()
        logger.info(f"Current memory usage: {mem_usage}")
        
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
            
            # Log memory usage periodically too
            mem_usage = get_memory_usage()
            logger.debug(f"Memory usage: {mem_usage}")
            
            # Force garbage collection to help with memory management
            gc.collect()
            
        except Exception as e:
            logger.error(f"Database connection monitor: Connection test failed: {e}")
            # Try to recreate the engine
            try:
                if _engine:
                    _engine.dispose()
                setup_database()
                logger.info("Database engine recreated after monitor detected failure")
            except Exception as re:
                logger.error(f"Failed to recreate engine: {re}")
        
        # Sleep for 2 minutes before next check (more aggressive interval)
        time.sleep(120)

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

@with_retries(max_attempts=2)
def get_session():
    """Get a new SQLAlchemy session with health check"""
    global _session_factory
    
    # Check engine health
    check_engine_health()
    
    if _session_factory is None:
        setup_database()  # Try to set up again if not already done
        
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

# Apply the retry decorator to the db_session context manager
@contextmanager
def db_session() -> Generator:
    """Context manager for database sessions with extremely aggressive connection handling"""
    session = None
    
    try:
        session = get_session()
        # Set a shorter statement timeout
        session.execute(text("SET statement_timeout = 2000"))  # Even shorter timeout: 2 seconds
        yield session
        
        # Try to commit changes with a timeout
        try:
            session.commit()
            logger.debug("Session committed successfully")
        except Exception as commit_error:
            session.rollback()
            logger.warning(f"Commit failed, rolling back: {commit_error}")
            raise
            
    except Exception as session_error:
        logger.error(f"Session error: {session_error}")
        raise
        
    finally:
        if session:
            try:
                # Force immediate return of connection to pool
                session.close()
                # Explicitly remove session
                if hasattr(_session_factory, 'remove'):
                    _session_factory.remove()
                logger.debug("Session closed and removed")
            except Exception as close_error:
                logger.warning(f"Error closing session: {close_error}")

@with_retries(max_attempts=2)
def check_engine_health():
    """Check if the engine/pool is healthy and recreate if needed"""
    global _engine, _session_factory
    
    try:
        if not _engine:
            logger.warning("Database engine is None, creating new engine")
            setup_database()
            return {"healthy": False, "message": "Engine was None, recreated"}
            
        try:
            # Quick connection test
            with _engine.connect() as conn:
                # Set a short timeout for the health check
                conn.execute(text("SET statement_timeout = 1000"))  # 1 second timeout
                result = conn.execute(text("SELECT 1")).scalar()
                if result == 1:
                    return {"healthy": True, "message": "Connection test passed"}
                else:
                    raise Exception("Connection test returned unexpected result")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            logger.warning("Recreating database engine due to connection failure")
            
            # Dispose of existing engine
            try:
                if _engine:
                    _engine.dispose()
            except Exception as dispose_error:
                logger.error(f"Error disposing engine: {dispose_error}")
                
            # Create new engine
            setup_database()
            return {"healthy": False, "message": f"Connection failed: {str(e)}, engine recreated"}
            
        return {"healthy": True, "message": "Engine health check passed"}
    except Exception as e:
        logger.error(f"Error in check_engine_health: {e}")
        return {"healthy": False, "message": f"Error during health check: {str(e)}"}

@with_retries(max_attempts=2)
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

# Add a connection cleanup function that will be registered to run on app shutdown
def cleanup_db_connections():
    """Force cleanup of all database connections in the pool"""
    global _engine, _session_factory
    
    try:
        if _session_factory:
            logger.info("Cleaning up all database sessions")
            try:
                _session_factory.remove()
                logger.info("✓ All sessions removed")
            except Exception as e:
                logger.error(f"Error removing sessions: {e}")
                
        if _engine:
            logger.info("Disposing of database engine and connection pool")
            try:
                _engine.dispose()
                logger.info("✓ Engine disposed")
            except Exception as e:
                logger.error(f"Error disposing engine: {e}")
                
        # Force garbage collection
        gc.collect()
        logger.info("Forced garbage collection during cleanup")
    except Exception as e:
        logger.error(f"Unhandled exception in cleanup_db_connections: {e}")
        logger.error(traceback.format_exc())
        
# Helper function to close all active sessions
def close_all_sessions():
    """Close all active sessions and remove them from the registry"""
    global _session_factory
    if _session_factory:
        try:
            _session_factory.remove()
            logger.info("All sessions have been closed and removed")
        except Exception as e:
            logger.error(f"Error closing all sessions: {e}")
            
    # Also dispose of the engine to close all connections
    if _engine:
        try:
            _engine.dispose()
            logger.info("Engine disposed, all connections closed")
        except Exception as e:
            logger.error(f"Error disposing engine: {e}") 