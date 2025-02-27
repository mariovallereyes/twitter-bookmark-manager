import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator, List, Dict, Any
from .models import Base
import logging
from dotenv import load_dotenv
from sqlalchemy.ext.declarative import declarative_base
from .vector_store import ChromaStore
from sqlalchemy import text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create database engine with absolute path
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'twitter_bookmarks.db')}"
logger.info(f"Using database path: {DATABASE_URL}")
engine = create_engine(DATABASE_URL)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class DatabaseManager:
    def __init__(self):
        # SQLite setup
        self.database_url = DATABASE_URL
        self.engine = engine
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.ScopedSession = scoped_session(self.SessionFactory)

    def init_db(self) -> None:
        """Initialize the SQLite database, creating all tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("SQLite database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise

    def drop_db(self) -> None:
        """Drop all SQLite tables (useful for testing)"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator:
        """Get a database session with automatic cleanup"""
        session = self.ScopedSession()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
            self.ScopedSession.remove()

# Create global instances
db = DatabaseManager()

def init_database():
    """Initialize the database (called at application startup)"""
    try:
        db.init_db()
        logger.info("Database initialization complete")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def get_db_session():
    """Get a database session (for use in application code)"""
    session = SessionLocal()
    # Verify connection with proper text declaration
    session.execute(text("SELECT 1"))
    return session

_vector_store = None

def get_vector_store():
    """Get or create vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaStore()
    return _vector_store