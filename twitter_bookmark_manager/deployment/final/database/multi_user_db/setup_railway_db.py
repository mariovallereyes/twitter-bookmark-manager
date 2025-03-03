"""
Setup script for initializing the Railway PostgreSQL database with the correct schema.
This script uses the DATABASE_PUBLIC_URL from Railway to create all necessary tables and indexes.
"""

import os
import sys
import psycopg2
import logging
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('setup_railway_db')

# Load environment variables from .env.final
env_path = Path(__file__).parents[4] / '.env.final'
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.error(f"Could not find .env.final at {env_path}")
    sys.exit(1)

def parse_database_url(url):
    """Parse DATABASE_PUBLIC_URL into connection parameters"""
    parsed = urlparse(url)
    return {
        'dbname': parsed.path[1:],  # Remove leading slash
        'user': parsed.username,
        'password': parsed.password,
        'host': parsed.hostname,
        'port': parsed.port
    }

def get_connection(db_url):
    """Connect to PostgreSQL database using DATABASE_PUBLIC_URL"""
    try:
        conn_params = parse_database_url(db_url)
        conn = psycopg2.connect(**conn_params)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def create_tables(conn):
    """Create all necessary tables"""
    cursor = conn.cursor()
    
    # Create users table
    logger.info("Creating users table...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(255) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE,
        auth_provider VARCHAR(50) NOT NULL,
        provider_user_id VARCHAR(255) NOT NULL,
        display_name VARCHAR(255),
        profile_image_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    );
    ''')

    # Create bookmarks table
    logger.info("Creating bookmarks table...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS bookmarks (
        id SERIAL PRIMARY KEY,
        tweet_id VARCHAR(255) NOT NULL,
        tweet_text TEXT,
        tweet_author VARCHAR(255),
        tweet_author_id VARCHAR(255),
        tweet_created_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_id INTEGER REFERENCES users(id)
    );
    ''')

    # Create categories table
    logger.info("Creating categories table...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_id INTEGER REFERENCES users(id)
    );
    ''')

    # Create bookmark_categories table
    logger.info("Creating bookmark_categories table...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS bookmark_categories (
        id SERIAL PRIMARY KEY,
        bookmark_id INTEGER REFERENCES bookmarks(id) ON DELETE CASCADE,
        category_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
        UNIQUE(bookmark_id, category_id)
    );
    ''')

    # Create media table
    logger.info("Creating media table...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS media (
        id SERIAL PRIMARY KEY,
        media_key VARCHAR(255) NOT NULL,
        type VARCHAR(50),
        url TEXT,
        preview_image_url TEXT,
        bookmark_id INTEGER REFERENCES bookmarks(id) ON DELETE CASCADE
    );
    ''')

    # Create conversations table
    logger.info("Creating conversations table...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id SERIAL PRIMARY KEY,
        conversation_id VARCHAR(255) NOT NULL,
        bookmark_id INTEGER REFERENCES bookmarks(id) ON DELETE CASCADE
    );
    ''')

    # Drop and recreate tables if needed
    cursor.execute("""
    DO $$
    BEGIN
        -- Drop existing tables if they exist but have wrong structure
        IF EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'users' 
            AND table_schema = 'public'
        ) AND NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'users' 
            AND column_name = 'username'
        ) THEN
            DROP TABLE IF EXISTS conversations CASCADE;
            DROP TABLE IF EXISTS media CASCADE;
            DROP TABLE IF EXISTS bookmark_categories CASCADE;
            DROP TABLE IF EXISTS bookmarks CASCADE;
            DROP TABLE IF EXISTS categories CASCADE;
            DROP TABLE IF EXISTS users CASCADE;
            
            -- Recreate tables
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE,
                auth_provider VARCHAR(50) NOT NULL,
                provider_user_id VARCHAR(255) NOT NULL,
                display_name VARCHAR(255),
                profile_image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
        END IF;
    END $$;
    """)

    conn.commit()
    logger.info("✅ Tables created successfully")

def create_indexes(conn):
    """Create necessary indexes for performance"""
    cursor = conn.cursor()
    logger.info("Creating indexes...")
    
    try:
        # Indexes for users table
        cursor.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'users' AND column_name = 'auth_provider'
            ) THEN
                CREATE INDEX IF NOT EXISTS idx_users_auth_provider ON users(auth_provider);
            END IF;
        END $$;
        """)
        
        cursor.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'users' AND column_name = 'provider_user_id'
            ) THEN
                CREATE INDEX IF NOT EXISTS idx_users_provider_id ON users(provider_user_id);
            END IF;
        END $$;
        """)
        
        # Indexes for bookmarks table
        cursor.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'bookmarks' AND column_name = 'user_id'
            ) THEN
                CREATE INDEX IF NOT EXISTS idx_bookmarks_user_id ON bookmarks(user_id);
            END IF;
        END $$;
        """)
        
        cursor.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'bookmarks' AND column_name = 'tweet_id'
            ) THEN
                CREATE INDEX IF NOT EXISTS idx_bookmarks_tweet_id ON bookmarks(tweet_id);
            END IF;
        END $$;
        """)
        
        # Indexes for categories table
        cursor.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'categories' AND column_name = 'user_id'
            ) THEN
                CREATE INDEX IF NOT EXISTS idx_categories_user_id ON categories(user_id);
            END IF;
        END $$;
        """)
        
        # Create unique constraint for category names per user
        cursor.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'categories' AND column_name = 'user_id'
            ) AND EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'categories' AND column_name = 'name'
            ) THEN
                BEGIN
                    ALTER TABLE categories ADD CONSTRAINT unique_category_name_per_user 
                    UNIQUE (name, user_id);
                EXCEPTION
                    WHEN duplicate_table THEN
                        NULL;
                END;
            END IF;
        END $$;
        """)
        
        conn.commit()
        logger.info("✅ Successfully created all indexes")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        conn.rollback()
        raise

def create_system_user(conn):
    """Create the system user (ID=1) if it doesn't exist"""
    cursor = conn.cursor()
    logger.info("Creating system user...")
    
    cursor.execute("SELECT id FROM users WHERE id = 1")
    if not cursor.fetchone():
        cursor.execute('''
        INSERT INTO users (id, username, email, auth_provider, provider_user_id, display_name)
        VALUES (1, 'system', 'system@example.com', 'system', 'system', 'System User')
        ON CONFLICT (id) DO NOTHING
        ''')
        conn.commit()

def main():
    """Main function to set up the database"""
    logger.info("Starting database setup for Railway")
    
    # Get DATABASE_PUBLIC_URL from environment
    db_url = os.environ.get('DATABASE_PUBLIC_URL')
    if not db_url:
        logger.error("DATABASE_PUBLIC_URL environment variable not set")
        sys.exit(1)
    
    try:
        # Connect to database
        conn = get_connection(db_url)
        
        # Create tables
        create_tables(conn)
        
        # Create indexes
        create_indexes(conn)
        
        # Create system user
        create_system_user(conn)
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during database setup: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main() 