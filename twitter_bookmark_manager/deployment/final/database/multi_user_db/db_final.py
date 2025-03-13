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
import traceback
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import psycopg2  # Add direct psycopg2 import

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, ForeignKey, Boolean, func, text, event, inspect
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import exc, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, InterfaceError

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
_direct_conn = None  # Direct psycopg2 connection for emergency access

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

# Get direct PostgreSQL connection parameters from environment
def get_pg_direct_params() -> Dict[str, str]:
    """
    Get PostgreSQL connection parameters directly from environment variables.
    Prioritizes PG* environment variables, then falls back to POSTGRES_* variables,
    and finally extracts from DATABASE_URL if needed.
    """
    # Log all environment variables related to Railway for debugging
    RAILWAY_VARS = [var for var in os.environ.keys() if var.startswith('RAILWAY_')]
    DB_VARS = ["PGUSER", "PGPASSWORD", "PGHOST", "PGPORT", "PGDATABASE",
               "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "DATABASE_URL"]
    
    # Log all relevant environment variables (without sensitive values)
    logger.info("=== ENVIRONMENT VARIABLES DEBUG ===")
    for var in sorted(RAILWAY_VARS + DB_VARS):
        if var in os.environ:
            if "PASSWORD" in var or "URL" in var:
                logger.info(f"{var}: [REDACTED FOR SECURITY]")
            else:
                logger.info(f"{var}: {os.environ.get(var)}")
    
    # First try to use the PG* variables (Railway standard)
    params = {
        "user": os.environ.get("PGUSER"),
        "password": os.environ.get("PGPASSWORD"),
        "host": os.environ.get("PGHOST"),
        "port": os.environ.get("PGPORT", "5432"),
        "database": os.environ.get("PGDATABASE"),
    }
    
    # Fall back to POSTGRES_* variables if needed
    if not params["user"]:
        params["user"] = os.environ.get("POSTGRES_USER")
        if params["user"]:
            logger.info("Using POSTGRES_USER as fallback")
    
    if not params["password"]:
        params["password"] = os.environ.get("POSTGRES_PASSWORD")
        if params["password"]:
            logger.info("Using POSTGRES_PASSWORD as fallback")
    
    if not params["database"]:
        params["database"] = os.environ.get("POSTGRES_DB")
        if params["database"]:
            logger.info("Using POSTGRES_DB as fallback")
    
    # MODIFIED: Don't force postgres.railway.internal - use environment variables as provided
    # This allows using the external proxy hostname when needed
    if 'RAILWAY_PROJECT_ID' in os.environ and not params["host"]:
        # Only use internal hostname as fallback if no host is specified
        params["host"] = "postgres.railway.internal"
        logger.info(f"Using default internal PostgreSQL hostname: {params['host']}")
    
    # If we're still missing critical parameters, try to extract from DATABASE_URL
    if not all([params["user"], params["password"], params["host"], params["database"]]):
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            logger.info("Extracting connection parameters from DATABASE_URL")
            try:
                # Extract credentials from DATABASE_URL
                from urllib.parse import urlparse
                parsed_url = urlparse(db_url)
                
                if not params["user"] and parsed_url.username:
                    params["user"] = parsed_url.username
                
                if not params["password"] and parsed_url.password:
                    params["password"] = parsed_url.password
                
                # MODIFIED: Don't override host from DATABASE_URL if already set
                if not params["host"] and parsed_url.hostname:
                    params["host"] = parsed_url.hostname
                
                if not params["port"] and parsed_url.port:
                    params["port"] = str(parsed_url.port)
                
                if not params["database"] and parsed_url.path:
                    params["database"] = parsed_url.path.lstrip('/')
            except Exception as e:
                logger.error(f"Error parsing DATABASE_URL: {e}")
    
    # Log which parameters we were able to resolve (without revealing the password)
    safe_params = {**params}
    if safe_params["password"]:
        safe_params["password"] = "******"
    
    logger.info(f"Resolved database parameters: {safe_params}")
    
    # Verify we have all required parameters
    missing = [k for k, v in params.items() if not v and k != "port"]
    if missing:
        logger.warning(f"Missing required database parameters: {missing}")
    
    return params

# Get a direct psycopg2 connection - useful as a fallback
def get_direct_connection():
    """
    Establish a direct connection to PostgreSQL using psycopg2.
    This is a fallback when SQLAlchemy connection fails.
    """
    try:
        params = get_pg_direct_params()
        logger.info(f"Attempting direct psycopg2 connection to {params['host']}:{params['port']}/{params['database']} as {params['user']}")
        
        # Extra logging to ensure correct parameters
        if params['host'] == 'postgres.railway.internal':
            logger.info("✅ Using correct Railway internal PostgreSQL hostname")
        else:
            logger.warning(f"⚠️ Not using Railway internal hostname. Using: {params['host']}")
        
        conn = psycopg2.connect(
            user=params["user"],
            password=params["password"],
            host=params["host"],
            port=params["port"],
            database=params["database"],
            connect_timeout=10  # 10 seconds timeout
        )
        
        logger.info("Direct psycopg2 connection established successfully")
        return conn
    except Exception as e:
        logger.error(f"Failed to establish direct psycopg2 connection: {e}")
        logger.error(traceback.format_exc())
        # Log raw connection details for debugging (without password)
        safe_params = get_pg_direct_params()
        safe_params["password"] = "******"
        logger.error(f"Connection parameters used: {safe_params}")
        raise

def get_db_url() -> str:
    """
    Get the database URL for SQLAlchemy.
    Prioritizes using DATABASE_URL environment variable if available,
    otherwise constructs URL from individual parameters.
    """
    # First try to use DATABASE_URL directly
    db_url = os.environ.get("DATABASE_URL")
    
    if not db_url:
        # Construct URL from individual parameters
        params = get_pg_direct_params()
        
        # Construct URL
        db_url = (
            f"postgresql://{params['user']}:{params['password']}@"
            f"{params['host']}:{params['port']}/{params['database']}"
        )
        logger.info(f"Constructed database URL from parameters (host: {params['host']})")
    else:
        logger.info("Using DATABASE_URL from environment")
        
        # Check if we need to adapt the DATABASE_URL to use a custom host/port
        custom_host = os.environ.get("PGHOST")
        custom_port = os.environ.get("PGPORT")
        
        if custom_host or custom_port:
            from urllib.parse import urlparse, urlunparse
            
            parsed_url = urlparse(db_url)
            parts = list(parsed_url)
            netloc = parsed_url.netloc
            
            # Replace host/port if specified in environment
            netloc_parts = netloc.split('@')
            if len(netloc_parts) > 1:
                host_port = netloc_parts[1].split(':')
                
                if custom_host and len(host_port) > 0:
                    host_port[0] = custom_host
                    logger.info(f"Overriding hostname in DATABASE_URL with {custom_host}")
                
                if custom_port and len(host_port) > 1:
                    host_port[1] = custom_port
                    logger.info(f"Overriding port in DATABASE_URL with {custom_port}")
                elif custom_port:
                    host_port.append(custom_port)
                    logger.info(f"Adding port {custom_port} to DATABASE_URL")
                
                netloc_parts[1] = ':'.join(host_port)
                parts[1] = '@'.join(netloc_parts)
                db_url = urlunparse(parts)
    
    # Log the URL we're using (with password masked)
    safe_url = db_url
    if "://" in safe_url:
        parts = safe_url.split("://", 1)
        if "@" in parts[1]:
            auth_rest = parts[1].split("@", 1)
            if ":" in auth_rest[0]:
                user_pass = auth_rest[0].split(":", 1)
                user_pass[1] = "********"
                auth_rest[0] = ":".join(user_pass)
            safe_url = parts[0] + "://" + auth_rest[0] + "@" + auth_rest[1]
            
    logger.info(f"Using database URL: {safe_url}")
    return db_url

def setup_database(force_reconnect: bool = False, show_sql: bool = False) -> Engine:
    """
    Simple and robust database connection setup optimized for Railway.
    
    Args:
        force_reconnect: Whether to force recreation of the engine
        show_sql: Whether to enable SQL statement logging
        
    Returns:
        SQLAlchemy engine
    """
    global _engine, _session_factory
    
    # Use lock to prevent race conditions
    with _engine_lock:
        if _engine is not None and not force_reconnect:
            # Quick validation of existing engine
            try:
                with _engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return _engine
            except Exception as e:
                logger.warning(f"Existing engine failed validation: {e}")
                # Continue to create a new engine
        
        # Try to establish a direct connection first as a validation
        direct_conn = get_direct_connection()
        if not direct_conn:
            logger.warning("Direct connection failed, but will attempt SQLAlchemy anyway")
            
        try:
            # Get database URL
            db_url = get_db_url()
            is_postgresql = 'postgresql' in db_url
            
            # ULTRA SIMPLIFIED APPROACH for Railway
            # Fewer parameters, less complexity = fewer points of failure
            engine_params = {
                'echo': show_sql,
                'pool_pre_ping': True,
            }
            
            if is_postgresql:
                engine_params.update({
                    'poolclass': QueuePool,
                    'pool_size': 3,                 # Small pool size
                    'max_overflow': 5,              # Limited overflow
                    'pool_timeout': 10,             # Short timeout
                    'pool_recycle': 300,            # 5 minute recycle
                    'connect_args': {
                        'connect_timeout': 10,
                        'application_name': 'twitter_bookmark_manager_sqlalchemy'
                    }
                })
            
            # Dispose of existing engine
            if _engine is not None:
                try:
                    _engine.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing engine: {e}")
            
            # Clear all active sessions
            close_all_sessions()
            
            # Create the engine
            logger.info("Creating new SQLAlchemy engine")
            _engine = create_engine(db_url, **engine_params)
            
            # Set basic statement timeout for PostgreSQL
            if is_postgresql:
                @event.listens_for(_engine, "connect")
                def set_pg_timeout(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("SET statement_timeout = '20s'")
                    cursor.close()
            
            # Create session factory
            _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
            
            # Verify connection
            with _engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"✅ SQLAlchemy engine created successfully")
            return _engine
            
        except Exception as e:
            logger.error(f"❌ SQLAlchemy engine creation failed: {e}")
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

# Get a database connection for multi-purpose use
@with_db_retry(max_tries=5, backoff_in_seconds=1)
def get_db_connection():
    """
    Get a database connection with improved reliability.
    Tries both SQLAlchemy and direct psycopg2 approaches.
    """
    try:
        # First try SQLAlchemy
        engine = get_engine()
        
        # Create SQLAlchemy connection
        return engine.connect()
    except Exception as e:
        logger.warning(f"SQLAlchemy connection failed: {e}, trying direct psycopg2 connection")
        
        # Fallback to direct psycopg2 connection
        conn = get_direct_connection()
        if conn:
            return conn
        else:
            raise Exception("Both SQLAlchemy and direct connection methods failed")

# Alias for backward compatibility
def get_db_session():
    """Alias for create_session for backward compatibility"""
    return create_session()

def get_bookmarks_for_user(user_id):
    """
    Get all bookmarks for a specific user.
    Uses SQLAlchemy for consistency like the PythonAnywhere implementation.
    
    Args:
        user_id: User ID to get bookmarks for
        
    Returns:
        List of Bookmark objects
    """
    from .models_final import Bookmark
    
    logger.info(f"Fetching bookmarks for user {user_id}")
    
    bookmarks = []
    
    try:
        # Get connection using session for consistency
        with db_session() as session:
            # Use SQLAlchemy session.execute with text query for better control
            query = """
                SELECT bookmark_id, text, created_at, author as author_name, author_id as author_username, 
                       media_files, raw_data, user_id
                FROM bookmarks
                WHERE user_id = :user_id
                ORDER BY created_at DESC
            """
            
            result = session.execute(text(query), {"user_id": user_id})
            rows = result.fetchall()
            logger.info(f"Found {len(rows)} bookmarks for user {user_id}")
            
            # Convert to Bookmark objects with error handling for each row
            for row in rows:
                try:
                    bookmarks.append(Bookmark.from_row(row))
                except Exception as e:
                    logger.error(f"Error converting row to Bookmark: {e}")
                    logger.error(f"Problematic row: {row}")
                    # Continue to next row
            
        return bookmarks
        
    except Exception as e:
        logger.error(f"Error fetching bookmarks for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        return []

def create_tables():
    """Create database tables if they don't already exist using the most reliable available method"""
    logger.info("Initializing database tables")
    
    # Try to get a connection - first attempt with SQLAlchemy
    try:
        # Get engine and metadata
        engine = get_engine()
        metadata = MetaData()
        
        # Define tables using SQLAlchemy constructs
        # Users table
        users = Table('users', metadata,
            Column('id', Integer, primary_key=True),
            Column('username', String(100), unique=True),
            Column('email', String(255), unique=True),
            Column('twitter_id', String(100), unique=True),
            Column('auth_provider', String(50)),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('last_login', DateTime),
            Column('profile_data', Text)
        )
        
        # Bookmarks table
        bookmarks = Table('bookmarks', metadata,
            Column('id', Integer, primary_key=True),
            Column('bookmark_id', String(100), unique=True),
            Column('text', Text),
            Column('created_at', String(50)),
            Column('author', String(100)),
            Column('author_id', String(100)),
            Column('media_files', Text),
            Column('raw_data', Text),  # Use Text for compatibility instead of JSON
            Column('user_id', Integer, ForeignKey('users.id')),
            Column('processed', Boolean, default=False),
            Column('retried', Integer, default=0)
        )
        
        # Categories table
        categories = Table('categories', metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String(100)),
            Column('description', Text),
            Column('user_id', Integer, ForeignKey('users.id')),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        )
        
        # BookmarkCategories table - many-to-many relationship
        bookmark_categories = Table('bookmark_categories', metadata,
            Column('id', Integer, primary_key=True),
            Column('bookmark_id', Integer, ForeignKey('bookmarks.id')),
            Column('category_id', Integer, ForeignKey('categories.id')),
            Column('user_id', Integer, ForeignKey('users.id')),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Attempt to create tables using SQLAlchemy
        try:
            # Create tables
            metadata.create_all(engine)
            logger.info("✅ Database tables created or already exist via SQLAlchemy")
            return
        except Exception as sqlalchemy_error:
            logger.warning(f"SQLAlchemy table creation failed: {sqlalchemy_error}")
            # Fall through to direct method
    except Exception as e:
        logger.warning(f"Unable to create tables via SQLAlchemy: {e}")
        
    # If SQLAlchemy fails, try direct psycopg2 connection for PostgreSQL
    try:
        # Get direct connection
        conn = get_direct_connection()
        if not conn:
            logger.error("Failed to get direct connection for table creation")
            raise Exception("No database connection available for table creation")
        
        # Create tables directly with SQL
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        # 1. Users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE,
            email VARCHAR(255) UNIQUE,
            twitter_id VARCHAR(100) UNIQUE,
            auth_provider VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW(),
            last_login TIMESTAMP,
            profile_data TEXT
        )
        """)
        
        # 2. Bookmarks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id SERIAL PRIMARY KEY,
            bookmark_id VARCHAR(100) UNIQUE,
            text TEXT,
            created_at VARCHAR(50),
            author VARCHAR(100),
            author_id VARCHAR(100),
            media_files TEXT,
            raw_data TEXT,
            user_id INTEGER REFERENCES users(id),
            processed BOOLEAN DEFAULT FALSE,
            retried INTEGER DEFAULT 0
        )
        """)
        
        # 3. Categories table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            description TEXT,
            user_id INTEGER REFERENCES users(id),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """)
        
        # 4. BookmarkCategories table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookmark_categories (
            id SERIAL PRIMARY KEY,
            bookmark_id INTEGER REFERENCES bookmarks(id),
            category_id INTEGER REFERENCES categories(id),
            user_id INTEGER REFERENCES users(id),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(bookmark_id, category_id)
        )
        """)
        
        # Commit the changes
        conn.commit()
        cursor.close()
        
        logger.info("✅ Database tables created or already exist via direct SQL")
    except Exception as direct_error:
        logger.error(f"❌ Direct SQL table creation failed: {direct_error}")
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
        "timestamp": datetime.now().isoformat(),
        "healthy": False
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
            
            all_tables_exist = True
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    results[table] = count
                except Exception as e:
                    all_tables_exist = False
                    results[table] = f"Error: {str(e)}"
                    
            status["tables_exist"] = all_tables_exist
            status["record_counts"] = results
            
            # If connection is OK but tables don't exist, try to create them
            if status["connection"] == "ok" and not all_tables_exist:
                logger.warning("Database connection successful but tables are missing. Attempting to create tables...")
                
                try:
                    # Run create_tables function to ensure all tables exist
                    create_tables()
                    status["tables_exist"] = True
                    status["message"] = "Tables created successfully"
                    status["healthy"] = True
                except Exception as table_error:
                    status["message"] = f"Failed to create tables: {str(table_error)}"
            else:
                # If connection and tables are OK, mark as healthy
                status["healthy"] = True
            
    except Exception as e:
        status["error"] = str(e)
        status["message"] = str(e)
        
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