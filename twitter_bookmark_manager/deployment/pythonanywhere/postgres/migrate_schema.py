"""
PostgreSQL schema migration script for PythonAnywhere deployment.
This script handles type conversions and schema adjustments without modifying local code.
"""
import os
import sys
from pathlib import Path
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

def get_connection():
    """Get PostgreSQL connection with proper settings"""
    return psycopg2.connect(
        dbname='mariovallereyes$default',
        user='mariovallereyes',
        password=os.getenv('POSTGRES_PASSWORD'),
        host='mariovallereyes-4374.postgres.pythonanywhere-services.com',
        port='14374'
    )

def migrate_schema():
    """Apply PostgreSQL-specific schema adjustments"""
    try:
        conn = get_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        try:
            # Drop existing tables if they exist
            logger.info("Dropping existing tables...")
            cur.execute("""
                DROP TABLE IF EXISTS bookmark_categories CASCADE;
                DROP TABLE IF EXISTS media CASCADE;
                DROP TABLE IF EXISTS bookmarks CASCADE;
                DROP TABLE IF EXISTS categories CASCADE;
                DROP TABLE IF EXISTS users CASCADE;
                DROP TABLE IF EXISTS conversations CASCADE;
            """)
            
            # Create tables with proper types
            logger.info("Creating tables with correct types...")
            
            # Users table
            cur.execute("""
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    twitter_id VARCHAR(255) UNIQUE NOT NULL,
                    access_token VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Categories table
            cur.execute("""
                CREATE TABLE categories (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Bookmarks table with VARCHAR id
            cur.execute("""
                CREATE TABLE bookmarks (
                    id VARCHAR(255) PRIMARY KEY,
                    text TEXT,
                    created_at TIMESTAMP,
                    author_name VARCHAR(255),
                    author_username VARCHAR(255),
                    media_files JSONB,
                    raw_data JSONB,
                    user_id INTEGER REFERENCES users(id)
                )
            """)
            
            # Media table with VARCHAR bookmark_id
            cur.execute("""
                CREATE TABLE media (
                    id SERIAL PRIMARY KEY,
                    bookmark_id VARCHAR(255) REFERENCES bookmarks(id),
                    type VARCHAR(50),
                    url VARCHAR(512),
                    local_path VARCHAR(512),
                    hash VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Bookmark categories association table with VARCHAR bookmark_id
            cur.execute("""
                CREATE TABLE bookmark_categories (
                    bookmark_id VARCHAR(255) REFERENCES bookmarks(id),
                    category_id INTEGER REFERENCES categories(id),
                    PRIMARY KEY (bookmark_id, category_id)
                )
            """)
            
            # Conversations table
            cur.execute("""
                CREATE TABLE conversations (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_input TEXT NOT NULL,
                    system_response JSONB NOT NULL,
                    bookmarks_used JSONB,
                    is_archived BOOLEAN DEFAULT FALSE,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    importance_score FLOAT
                )
            """)
            
            logger.info("✓ Schema migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during schema migration: {str(e)}")
            raise
        finally:
            cur.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        migrate_schema()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1) 