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
import traceback

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
    Always prioritizes internal Railway PostgreSQL endpoint when in Railway environment.
    """
    # First check for complete DATABASE_URL environment variable
    database_url = os.environ.get('DATABASE_URL')
    
    # Flag to know if we're running in Railway
    is_railway = 'RAILWAY_PROJECT_ID' in os.environ
    
    # If in Railway, always use the internal endpoint
    if is_railway:
        logger.info("Detected Railway environment, prioritizing internal connection")
        
        # Get credentials from environment variables
        db_user = os.environ.get('PGUSER')
        db_password = os.environ.get('PGPASSWORD')
        db_name = os.environ.get('PGDATABASE', 'railway')
        
        if db_user and db_password:
            # Always use internal endpoint in Railway
            internal_url = f"postgresql://{db_user}:{db_password}@postgres.railway.internal:5432/{db_name}"
            logger.info(f"Using Railway internal endpoint: postgresql://{db_user}:****@postgres.railway.internal:5432/{db_name}")
            return internal_url
    
    # If DATABASE_URL exists but contains proxy.rlwy.net, ignore it and build from components
    if database_url and 'proxy.rlwy.net' in database_url:
        logger.warning("⚠️ Ignoring DATABASE_URL with proxy domain and building connection from individual credentials")
        database_url = None
    
    if database_url:
        # Make sure we're using the internal endpoint for Railway
        if 'railway.app' in database_url or 'proxy.rlwy.net' in database_url:
            logger.warning("⚠️ Converting external Railway URL to internal network URL")
            # Replace external endpoints with internal ones
            database_url = database_url.replace('postgresql://', '')
            # Extract credentials and database name
            credentials, rest = database_url.split('@', 1)
            # Replace the host:port with internal endpoint
            database_url = f"postgresql://{credentials}@postgres.railway.internal:5432/railway"
            
        logger.info(f"Using provided DATABASE_URL (sanitized): postgresql://user:****@{database_url.split('@')[1]}")
        return database_url
            
    # Try getting PostgreSQL connection info from Railway environment variables
    db_user = os.environ.get('PGUSER') or os.environ.get('DB_USER')
    db_password = os.environ.get('PGPASSWORD') or os.environ.get('DB_PASSWORD')
    db_host = os.environ.get('PGHOST') or os.environ.get('DB_HOST')
    db_port = os.environ.get('PGPORT') or os.environ.get('DB_PORT', '5432')
    db_name = os.environ.get('PGDATABASE') or os.environ.get('DB_NAME')
    
    # Check if we have all required PostgreSQL environment variables
    if all([db_user, db_password, db_host, db_name]):
        # Force internal Railway hostname if we're in Railway environment
        if is_railway and db_host != 'postgres.railway.internal':
            logger.warning(f"⚠️ Overriding provided host {db_host} with internal Railway endpoint")
            db_host = 'postgres.railway.internal'
            db_port = '5432'  # Always use standard port with internal endpoint
            
        logger.info(f"Using PostgreSQL database at {db_host}:{db_port}/{db_name}")
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    else:
        logger.warning("⚠️ No PostgreSQL credentials found in environment, defaulting to SQLite")
        # Use SQLite as fallback
        db_path = os.environ.get('SQLITE_PATH', 'twitter_bookmarks.db')
        return f"sqlite:///{db_path}"

def setup_database(force_reconnect: bool = False, show_sql: bool = False) -> Engine:
    """
    Set up the database connection with optimized parameters for reliability.
    
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
            # Test the existing connection before reusing it
            try:
                # Simple connection test
                with _engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Verified existing database connection is healthy")
                return _engine
            except Exception as e:
                logger.warning(f"Existing connection failed health check: {e}")
                # Continue to reconnect
            
        try:
            # Get database URL
            db_url = get_db_url()
            db_type = 'postgresql' if 'postgresql' in db_url else 'sqlite'
            
            # Log connection attempt with sanitized URL
            sanitized_url = db_url.replace('postgresql://', 'postgresql://user:****@')
            if 'sqlite' in db_url:
                sanitized_url = db_url
            logger.info(f"Connecting to database: {sanitized_url}")
            
            # Engine parameters dict - base settings
            engine_params = {
                'echo': show_sql,
                'pool_pre_ping': True,  # Test connections before use
            }
            
            # Add settings for PostgreSQL - optimized for Railway
            if db_type == 'postgresql':
                # Determine if we're in Railway - adjust settings accordingly
                is_railway = 'RAILWAY_PROJECT_ID' in os.environ
                
                # Pool settings - more aggressive for Railway
                pool_size = 3 if is_railway else 5
                max_overflow = 7 if is_railway else 10
                
                engine_params.update({
                    # Use QueuePool with adjusted settings
                    'poolclass': QueuePool,
                    'pool_size': pool_size,
                    'max_overflow': max_overflow,
                    'pool_timeout': 20,  # Wait up to 20 seconds for a connection
                    'pool_recycle': 300,  # Recycle connections after 5 minutes in Railway
                    'connect_args': {
                        'connect_timeout': 10,  # Connection timeout (10 seconds)
                        'keepalives': 1,  # Enable TCP keepalives
                        'keepalives_idle': 30,  # Time between keepalives
                        'keepalives_interval': 10,  # Interval between keepalives
                        'keepalives_count': 3,  # Number of keepalives before giving up
                        'application_name': 'twitter_bookmark_manager',  # Identify in pg_stat_activity
                        'options': '-c timezone=UTC'  # Set timezone to UTC
                    }
                })
                
                logger.info(f"Configured PostgreSQL pool: size={pool_size}, max_overflow={max_overflow}, recycle=300s")
                
            # Dispose of existing engine if forcing reconnect
            if force_reconnect and _engine is not None:
                try:
                    _engine.dispose()
                    logger.info("Disposed existing engine for reconnection")
                except Exception as e:
                    logger.warning(f"Error disposing engine: {e}")
                    
            # Close all active sessions before creating a new engine
            close_all_sessions()
            
            # Starting engine creation
            logger.info(f"Creating new database engine for {db_type}")
            _engine = create_engine(db_url, **engine_params)
            
            # Set up statement timeout and other session parameters for PostgreSQL
            if db_type == 'postgresql':
                @event.listens_for(_engine, "connect")
                def set_pg_parameters(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    # Set reasonable timeouts to prevent hanging operations
                    cursor.execute("SET statement_timeout = '20s';")  # 20 second statement timeout
                    cursor.execute("SET lock_timeout = '10s';")  # 10 second lock timeout
                    cursor.execute("SET idle_in_transaction_session_timeout = '60s';")  # 1 minute idle timeout
                    cursor.close()
            
            # Create session factory
            _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
            
            # Test connection before returning
            with _engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.scalar() != 1:
                    raise Exception("Database connection test failed")
            
            logger.info(f"✅ Database connection established and verified: {db_type}")
            
            # Initialize tables
            create_tables()
            
            return _engine
            
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            logger.error(traceback.format_exc())
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
    Comprehensive health check for the database engine with detailed diagnostics.
    
    Returns:
        Dict with health status information and diagnostics
    """
    global _engine, _last_connection_error
    
    # Start with a default response
    health_info = {
        "healthy": False,
        "message": "Engine not initialized",
        "pool": None,
        "last_error": _last_connection_error,
        "last_error_time": _connection_error_time.isoformat() if _connection_error_time else None,
        "diagnostics": {}
    }
    
    if _engine is None:
        return health_info
    
    try:
        # Basic connectivity test
        start_time = time.time()
        with _engine.connect() as conn:
            # Simple query to verify connectivity
            result = conn.execute(text("SELECT 1"))
            check_result = result.scalar() == 1
            
            # Get server version and time from PostgreSQL
            if 'postgresql' in str(_engine.url):
                try:
                    version_result = conn.execute(text("SHOW server_version"))
                    server_version = version_result.scalar()
                    
                    time_result = conn.execute(text("SELECT NOW()"))
                    server_time = time_result.scalar()
                    
                    # Check active connections
                    active_conn_result = conn.execute(text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                    ))
                    active_connections = active_conn_result.scalar()
                    
                    # Advanced diagnostics
                    health_info["diagnostics"] = {
                        "server_version": server_version,
                        "server_time": server_time.isoformat() if server_time else None,
                        "active_connections": active_connections,
                    }
                except Exception as diag_error:
                    # Non-critical error, just log it
                    logger.warning(f"Unable to collect extended diagnostics: {diag_error}")
            
        # Calculate query time
        query_time = time.time() - start_time
            
        # Get pool status for connection pooled engines
        if hasattr(_engine, 'pool'):
            try:
                pool_stats = {
                    "overflow": _engine.pool.overflow() if hasattr(_engine.pool, 'overflow') else 'n/a',
                    "checkedin": _engine.pool.checkedin() if hasattr(_engine.pool, 'checkedin') else 'n/a',
                    "checkedout": _engine.pool.checkedout() if hasattr(_engine.pool, 'checkedout') else 'n/a',
                    "size": _engine.pool.size() if hasattr(_engine.pool, 'size') else 'n/a'
                }
                health_info["pool"] = pool_stats
            except Exception as pool_error:
                health_info["pool_error"] = str(pool_error)
        
        # Update health status
        health_info.update({
            "healthy": check_result,
            "message": "Connection successful" if check_result else "Connection test failed",
            "query_time_ms": round(query_time * 1000, 2),
            "engine_type": str(_engine.url).split('://')[0],
        })
        
        return health_info
        
    except Exception as e:
        # Update the last connection error
        _last_connection_error = str(e)
        _connection_error_time = datetime.now()
        
        # Return detailed error information
        return {
            "healthy": False,
            "message": f"Connection error: {str(e)}",
            "error_type": type(e).__name__,
            "last_error": _last_connection_error,
            "last_error_time": _connection_error_time.isoformat() if _connection_error_time else None
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