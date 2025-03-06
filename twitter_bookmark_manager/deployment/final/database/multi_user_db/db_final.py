"""
SQLAlchemy database module for multi-user support with robust connection handling.
Optimized for Railway PostgreSQL with aggressive error handling and retry mechanisms.
"""

import os
import sys
import time
import logging
import threading
import secrets
import contextlib
import functools
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, ForeignKey, Boolean, func, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import exc, event
from sqlalchemy.engine import Engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('db_final.log')
    ]
)
logger = logging.getLogger('db_final')

# Global variables
_engine = None  # SQLAlchemy engine
_session_factory = None  # Session factory
_vector_store = None  # Vector store instance
_last_connection_error = None  # Track last connection error
_connection_error_time = None  # When the last error occurred

# Thread-local storage for sessions
_local_storage = threading.local()
_sessions_lock = threading.RLock()
_active_sessions = set()  # Track all active sessions
_engine_lock = threading.RLock()  # Lock for engine operations

# Retry decorator for database operations
def with_db_retry(max_tries=5, backoff_in_seconds=1):
    """
    Retry decorator for database operations that may fail due to connection issues.
    
    Args:
        max_tries: Maximum number of retry attempts
        backoff_in_seconds: Initial backoff time, will be doubled on each retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _last_connection_error, _connection_error_time
            
            retry_count = 0
            last_exception = None
            
            while retry_count < max_tries:
                try:
                    return func(*args, **kwargs)
                except (exc.OperationalError, exc.DisconnectionError) as e:
                    retry_count += 1
                    last_exception = e
                    error_message = str(e).lower()
                    
                    # Record connection error
                    _last_connection_error = str(e)
                    _connection_error_time = datetime.now()
                    
                    # Determine if this is a connection closed error
                    is_connection_closed = any(msg in error_message for msg in [
                        "connection closed", 
                        "server closed the connection", 
                        "connection reset",
                        "broken pipe",
                        "timeout"
                    ])
                    
                    if retry_count < max_tries:
                        # Calculate backoff time with exponential backoff
                        wait_time = backoff_in_seconds * (2 ** (retry_count - 1))
                        logger.warning(f"Database operation failed, retrying in {wait_time}s (attempt {retry_count}/{max_tries}): {e}")
                        time.sleep(wait_time)
                        
                        # Force reconnect on connection issues
                        if is_connection_closed:
                            try:
                                logger.info("Connection closed unexpectedly, forcing reconnection...")
                                with _engine_lock:
                                    setup_database(force_reconnect=True)
                                logger.info("Forced database reconnection for retry")
                            except Exception as reconnect_error:
                                logger.error(f"Failed to reconnect to database: {reconnect_error}")
                    else:
                        logger.error(f"Database operation failed after {max_tries} attempts: {e}")
                        raise
                        
            # If we get here, we've exhausted our retries
            raise last_exception
        return wrapper
    return decorator

def get_db_url() -> str:
    """
    Get the database URL from environment variables or use SQLite as fallback.
    Handles different environment variable naming depending on deployment.
    """
    # Try getting PostgreSQL connection info from Railway environment variables
    db_user = os.environ.get('PGUSER') or os.environ.get('DB_USER')
    db_password = os.environ.get('PGPASSWORD') or os.environ.get('DB_PASSWORD')
    db_host = os.environ.get('PGHOST') or os.environ.get('DB_HOST')
    db_port = os.environ.get('PGPORT') or os.environ.get('DB_PORT', '5432')
    db_name = os.environ.get('PGDATABASE') or os.environ.get('DB_NAME')
    
    # Check if we have all required PostgreSQL environment variables
    if all([db_user, db_password, db_host, db_name]):
        logger.info(f"Using PostgreSQL database at {db_host}:{db_port}/{db_name}")
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    else:
        logger.warning("⚠️ No PostgreSQL credentials found in environment, defaulting to SQLite")
        # Use SQLite as fallback
        db_path = os.environ.get('SQLITE_PATH', 'twitter_bookmarks.db')
        return f"sqlite:///{db_path}"

def setup_database(force_reconnect: bool = False, show_sql: bool = False) -> Engine:
    """
    Set up the database connection with optimized parameters.
    
    Args:
        force_reconnect: Whether to force recreation of the engine
        show_sql: Whether to enable SQL statement logging
        
    Returns:
        SQLAlchemy engine
    """
    global _engine, _session_factory
    
    # Use lock to prevent race conditions during engine setup/reset
    with _engine_lock:
        if _engine is not None and not force_reconnect:
            logger.info("Using existing database connection")
            return _engine
            
        try:
            # Get database URL
            db_url = get_db_url()
            db_type = 'postgresql' if 'postgresql' in db_url else 'sqlite'
            
            # Engine parameters dict
            engine_params = {
                'echo': show_sql,
                'pool_pre_ping': True,  # Test connections before use
            }
            
            # Add settings for PostgreSQL
            if db_type == 'postgresql':
                engine_params.update({
                    # Use NullPool to disable connection pooling
                    'poolclass': NullPool,
                    'connect_args': {
                        'connect_timeout': 3,  # Very short connection timeout
                        'keepalives': 1,  # Enable TCP keepalives
                        'keepalives_idle': 10,  # Shorter time between keepalives
                        'keepalives_interval': 2,  # Shorter interval between keepalives
                        'keepalives_count': 5,  # More keepalives before giving up
                        'application_name': 'twitter_bookmark_manager'  # Identify in pg_stat_activity
                    }
                })
                
            # Dispose of existing engine if forcing reconnect
            if force_reconnect and _engine is not None:
                try:
                    _engine.dispose()
                    logger.info("Disposed existing engine for reconnection")
                except Exception as e:
                    logger.warning(f"Error disposing engine: {e}")
                    
            # Close all active sessions before creating a new engine
            close_all_sessions()
                    
            # Create engine
            _engine = create_engine(db_url, **engine_params)
            
            # Set up statement timeout for PostgreSQL to prevent hangs
            if db_type == 'postgresql':
                @event.listens_for(_engine, "connect")
                def set_pg_statement_timeout(dbapi_connection, connection_record):
                    # Set statement timeout to 5 seconds (extremely short)
                    cursor = dbapi_connection.cursor()
                    cursor.execute("SET statement_timeout = '5s';")
                    cursor.close()
            
            # Create session factory
            _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
            
            logger.info(f"✅ Database connection established: {db_type} with NullPool (no connection pooling)")
            
            # Initialize tables
            create_tables()
            
            return _engine
            
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            raise

@event.listens_for(Engine, "connect")
def receive_connect(dbapi_connection, connection_record):
    """Log when a connection is created"""
    logger.debug(f"Connection established: {id(dbapi_connection)}")

@event.listens_for(Engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Perform connection health check on checkout"""
    # Check connection health
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
    except Exception as e:
        logger.warning(f"Connection {id(dbapi_connection)} checkout failed: {e}")
        # Let SQLAlchemy handle the reconnect
        raise exc.DisconnectionError("Connection test failed")

@event.listens_for(Engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log when a connection is returned to the pool"""
    logger.debug(f"Connection {id(dbapi_connection)} returned to pool")

@with_db_retry(max_tries=5, backoff_in_seconds=1)
def get_engine() -> Engine:
    """Get the SQLAlchemy engine, initializing if necessary"""
    global _engine
    with _engine_lock:
        if _engine is None:
            setup_database()
        return _engine

@with_db_retry(max_tries=5, backoff_in_seconds=1)
def create_session() -> Session:
    """Create a new SQLAlchemy session with retry"""
    global _session_factory, _active_sessions
    with _engine_lock:
        if _session_factory is None:
            setup_database()
    
    # Create new session
    session = _session_factory()
    
    # Track session for cleanup
    with _sessions_lock:
        _active_sessions.add(session)
        
    return session

def close_session(session: Session):
    """Safely close a session and remove from tracking"""
    global _active_sessions
    try:
        if session:
            session.close()
            # Remove from tracking
            with _sessions_lock:
                if session in _active_sessions:
                    _active_sessions.remove(session)
    except Exception as e:
        logger.error(f"Error closing session: {e}")

def close_all_sessions():
    """Close all tracked sessions"""
    global _active_sessions
    with _sessions_lock:
        sessions_to_close = list(_active_sessions)
        
    for session in sessions_to_close:
        try:
            session.close()
        except Exception as e:
            logger.error(f"Error closing session: {e}")
            
    with _sessions_lock:
        _active_sessions.clear()
        
    logger.info(f"Closed all tracked sessions")

@contextlib.contextmanager
def db_session():
    """Context manager for database sessions with automatic cleanup and retry"""
    session = None
    try:
        # Create session with retry
        retry_count = 0
        max_retries = 5
        last_exception = None
        
        while retry_count < max_retries:
            try:
                session = create_session()
                break
            except (exc.OperationalError, exc.DisconnectionError) as e:
                retry_count += 1
                last_exception = e
                
                if retry_count < max_retries:
                    wait_time = 1 * (2 ** (retry_count - 1))
                    logger.warning(f"Session creation failed, retrying in {wait_time}s (attempt {retry_count}/{max_retries}): {e}")
                    time.sleep(wait_time)
                    
                    # Force reconnect
                    try:
                        with _engine_lock:
                            setup_database(force_reconnect=True)
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect to database: {reconnect_error}")
                else:
                    logger.error(f"Session creation failed after {max_retries} attempts")
                    raise last_exception
        
        if not session:
            raise Exception("Failed to create session after retries")
            
        yield session
        
        # Commit with retry
        retry_count = 0
        while retry_count < max_retries:
            try:
                session.commit()
                break
            except (exc.OperationalError, exc.DisconnectionError) as e:
                retry_count += 1
                session.rollback()
                
                if retry_count < max_retries:
                    wait_time = 1 * (2 ** (retry_count - 1))
                    logger.warning(f"Commit failed, retrying in {wait_time}s (attempt {retry_count}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Commit failed after {max_retries} attempts")
                    raise
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        if session:
            close_session(session)

@with_db_retry(max_tries=5, backoff_in_seconds=1)
def get_db_connection():
    """Create and return a database session with retry"""
    return create_session()

# Alias for backward compatibility
def get_db_session():
    """Alias for create_session for backward compatibility"""
    return create_session()

def create_tables():
    """Create database tables if they don't exist"""
    from sqlalchemy import (Table, Column, Integer, String, Boolean, 
                          Text, DateTime, MetaData, ForeignKey, 
                          JSON, Float, SmallInteger, UniqueConstraint)
    
    # Get the engine
    engine = get_engine()
    metadata = MetaData()
    
    # Users table for multi-user support
    users = Table('users', metadata,
        Column('id', Integer, primary_key=True),
        Column('username', String(100), unique=True, nullable=False),
        Column('email', String(255), unique=True, nullable=False),
        Column('password_hash', String(255), nullable=False),
        Column('is_admin', Boolean, default=False),
        Column('created_at', DateTime, default=datetime.utcnow),
        Column('last_login', DateTime),
        Column('is_active', Boolean, default=True)
    )
    
    # Bookmarks table
    bookmarks = Table('bookmarks', metadata,
        Column('id', Integer, primary_key=True),
        Column('text', Text),
        Column('created_at', DateTime),
        Column('author_name', String(100)),
        Column('author_username', String(100)),
        Column('media_files', Text),
        Column('raw_data', JSON),
        Column('user_id', Integer, ForeignKey('users.id')),
        Column('processed', Boolean, default=False),
        Column('retried', SmallInteger, default=0)
    )
    
    # Categories table
    categories = Table('categories', metadata,
        Column('id', Integer, primary_key=True),
        Column('name', String(100)),
        Column('description', Text),
        Column('user_id', Integer, ForeignKey('users.id')),
        Column('created_at', DateTime, default=datetime.utcnow),
        Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        UniqueConstraint('name', 'user_id', name='uix_category_name_user_id')
    )
    
    # BookmarkCategories table - many-to-many relationship
    bookmark_categories = Table('bookmark_categories', metadata,
        Column('id', Integer, primary_key=True),
        Column('bookmark_id', Integer, ForeignKey('bookmarks.id')),
        Column('category_id', Integer, ForeignKey('categories.id')),
        Column('user_id', Integer, ForeignKey('users.id')),
        Column('created_at', DateTime, default=datetime.utcnow),
        UniqueConstraint('bookmark_id', 'category_id', name='uix_bookmark_category')
    )
    
    # Create tables
    try:
        metadata.create_all(engine)
        logger.info("✅ Database tables created or already exist")
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        raise

def check_engine_health() -> Dict[str, Any]:
    """
    Check the health of the database engine and connection pool.
    
    Returns:
        Dict with health status information
    """
    global _engine
    
    if _engine is None:
        return {
            "healthy": False,
            "message": "Engine not initialized",
            "pool": None
        }
        
    try:
        with _engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            check_result = result.scalar() == 1
            
            # Get pool status
            pool_stats = {
                "overflow": _engine.pool.overflow() if hasattr(_engine.pool, 'overflow') else 'n/a',
                "checkedin": _engine.pool.checkedin() if hasattr(_engine.pool, 'checkedin') else 'n/a',
                "checkedout": _engine.pool.checkedout() if hasattr(_engine.pool, 'checkedout') else 'n/a',
                "size": _engine.pool.size() if hasattr(_engine.pool, 'size') else 'n/a'
            }
            
            conn.close()
            
            if not check_result:
                return {
                    "healthy": False,
                    "message": "Database health check query failed",
                    "pool": pool_stats
                }
                
            return {
                "healthy": True,
                "message": "Database connection is healthy",
                "pool": pool_stats
            }
            
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return {
            "healthy": False,
            "message": str(e),
            "pool": None
        }

def check_database_status() -> Dict[str, Any]:
    """
    Check the status of the database by querying record counts.
    
    Returns:
        Dict with database status information
    """
    status = {
        "connection": "failed",
        "tables_exist": False,
        "record_counts": {},
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Test connection
            conn.execute(text("SELECT 1"))
            status["connection"] = "ok"
            
            # Check if tables exist
            results = {}
            tables = ["users", "bookmarks", "categories", "bookmark_categories"]
            
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    results[table] = count
                except Exception as e:
                    results[table] = f"Error: {str(e)}"
                    
            status["tables_exist"] = True
            status["record_counts"] = results
            
    except Exception as e:
        status["error"] = str(e)
        
    return status

def init_database():
    """Initialize the database and create tables"""
    setup_database()
    create_tables()

def cleanup_db_connections():
    """Clean up all database connections"""
    global _engine, _session_factory, _active_sessions
    
    logger.info("Cleaning up database connections")
    
    # Close all tracked sessions
    close_all_sessions()
    
    # Dispose of engine
    if _engine:
        _engine.dispose()
        logger.info("Engine disposed")
        
    # Reset globals
    _engine = None
    _session_factory = None
    _active_sessions = set()
    
    logger.info("Database connections cleaned up")

# Initialize vector store if available
try:
    from database.multi_user_db.vector_store_final import get_vector_store
    _vector_store = get_vector_store()
    logger.info("Vector store initialized")
except Exception as e:
    logger.warning(f"Vector store initialization skipped: {e}")
    _vector_store = None

def get_vector_store():
    """Get the vector store instance"""
    global _vector_store
    if _vector_store is None:
        try:
            from database.multi_user_db.vector_store_final import get_vector_store as init_vector_store
            _vector_store = init_vector_store()
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    return _vector_store 