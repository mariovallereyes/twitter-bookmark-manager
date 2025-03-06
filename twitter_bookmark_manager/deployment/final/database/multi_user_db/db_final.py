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
import psycopg2  # Add direct psycopg2 import

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
    """Get PostgreSQL connection parameters directly from environment variables"""
    return {
        'dbname': os.environ.get('PGDATABASE', 'railway'),
        'user': os.environ.get('PGUSER', 'postgres'),
        'password': os.environ.get('PGPASSWORD', ''),
        'host': 'postgres.railway.internal',  # Always use internal endpoint
        'port': '5432',
        'connect_timeout': '10',
        'application_name': 'twitter_bookmark_manager_direct'
    }

# Get a direct psycopg2 connection - useful as a fallback
def get_direct_connection():
    """
    Get a direct psycopg2 connection to PostgreSQL without using SQLAlchemy.
    This serves as a backup when SQLAlchemy pool has issues.
    """
    global _direct_conn
    
    # If we already have a connection, check if it's still valid
    if _direct_conn:
        try:
            # Test connection with a simple query
            cur = _direct_conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
            logger.debug("Reusing existing direct psycopg2 connection")
            return _direct_conn
        except Exception as e:
            logger.warning(f"Existing direct connection failed: {e}")
            try:
                _direct_conn.close()
            except:
                pass
            _direct_conn = None
    
    # Create a new connection
    try:
        # Get connection parameters
        params = get_pg_direct_params()
        logger.info(f"Creating direct psycopg2 connection to {params['host']}:{params['port']}/{params['dbname']}")
        
        # Create the connection
        conn = psycopg2.connect(**params)
        conn.autocommit = True  # Set autocommit mode
        
        # Test the connection
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        cur.close()
        
        logger.info(f"✅ Direct PostgreSQL connection established: {version}")
        _direct_conn = conn
        return conn
    except Exception as e:
        logger.error(f"❌ Direct PostgreSQL connection failed: {e}")
        return None

def get_db_url() -> str:
    """
    Get the database URL from environment variables or use SQLite as fallback.
    Always prioritizes internal Railway PostgreSQL endpoint when in Railway environment.
    """
    # Always use the internal endpoint for Railway PostgreSQL
    if 'RAILWAY_PROJECT_ID' in os.environ:
        db_user = os.environ.get('PGUSER')
        db_password = os.environ.get('PGPASSWORD')
        db_name = os.environ.get('PGDATABASE', 'railway')
        
        if db_user and db_password:
            # Construct internal URL
            internal_url = f"postgresql://{db_user}:{db_password}@postgres.railway.internal:5432/{db_name}"
            logger.info(f"Using Railway internal PostgreSQL URL: {internal_url.replace(db_password, '****')}")
            return internal_url
    
    # Fallback to other methods if not in Railway or missing credentials
    database_url = os.environ.get('DATABASE_URL')
    
    # If DATABASE_URL exists but contains proxy.rlwy.net, fix it
    if database_url and 'proxy.rlwy.net' in database_url:
        logger.warning("⚠️ Found external proxy URL, converting to internal endpoint")
        database_url = database_url.replace('postgresql://', '')
        credentials, rest = database_url.split('@', 1)
        host_port, db_name = rest.split('/', 1)
        database_url = f"postgresql://{credentials}@postgres.railway.internal:5432/{db_name}"
        logger.info(f"Converted to internal URL: postgresql://user:****@postgres.railway.internal:5432/{db_name}")
        return database_url
    
    # Use individual connection parameters as another fallback
    db_user = os.environ.get('PGUSER') or os.environ.get('DB_USER')
    db_password = os.environ.get('PGPASSWORD') or os.environ.get('DB_PASSWORD')
    db_host = os.environ.get('PGHOST') or os.environ.get('DB_HOST')
    db_port = os.environ.get('PGPORT') or os.environ.get('DB_PORT', '5432')
    db_name = os.environ.get('PGDATABASE') or os.environ.get('DB_NAME')
    
    # Check if we have all required PostgreSQL environment variables
    if all([db_user, db_password, db_host, db_name]):
        # Force internal Railway hostname in Railway environment
        if 'RAILWAY_PROJECT_ID' in os.environ and db_host != 'postgres.railway.internal':
            db_host = 'postgres.railway.internal'
            db_port = '5432'
            
        url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        logger.info(f"Using constructed PostgreSQL URL: postgresql://user:****@{db_host}:{db_port}/{db_name}")
        return url
    
    # Final fallback to SQLite
    logger.warning("⚠️ No PostgreSQL credentials found, defaulting to SQLite")
    db_path = os.environ.get('SQLITE_PATH', 'twitter_bookmarks.db')
    return f"sqlite:///{db_path}"

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