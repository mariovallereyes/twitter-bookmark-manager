"""
Database schema update script for multi-user support.
This script updates the PostgreSQL database schema to support multiple users.
It adds the users table and modifies existing tables to include user_id.
"""

import os
import sys
import psycopg2
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('db_schema_update_multi_user')

# Fix import path for user_model_final
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
sys.path.insert(0, parent_dir)

try:
    from .user_model_final import create_user_table, alter_tables_for_multi_user, create_system_user_if_needed
except ImportError:
    # Alternative import path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))))
    try:
        from deployment.final.database.multi_user_db.user_model_final import create_user_table, alter_tables_for_multi_user, create_system_user_if_needed
    except ImportError:
        logger.error("Failed to import user_model_final. Check import paths.")
        # Define the core functions here to avoid dependencies
        def create_user_table(conn):
            """Create the users table if it doesn't exist"""
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE,
                auth_provider VARCHAR(50) DEFAULT 'system',
                provider_user_id VARCHAR(255) DEFAULT 'system',
                display_name VARCHAR(255),
                profile_image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_admin BOOLEAN DEFAULT FALSE
            );
            ''')
            conn.commit()
            
        def alter_tables_for_multi_user(conn):
            """Add user_id column to existing tables"""
            cursor = conn.cursor()
            
            # Check if bookmarks table exists
            cursor.execute("SELECT to_regclass('public.bookmarks');")
            if cursor.fetchone()[0] is not None:
                # Check if user_id column exists in bookmarks table
                cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bookmarks' AND column_name='user_id';")
                if not cursor.fetchone():
                    # Add user_id to bookmarks table
                    cursor.execute("ALTER TABLE bookmarks ADD COLUMN user_id INTEGER REFERENCES users(id);")
                    
                    # Set existing bookmarks to a default system user
                    cursor.execute("UPDATE bookmarks SET user_id = 1 WHERE user_id IS NULL;")
            
            # Check if categories table exists
            cursor.execute("SELECT to_regclass('public.categories');")
            if cursor.fetchone()[0] is not None:
                # Check if user_id column exists in categories table
                cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='categories' AND column_name='user_id';")
                if not cursor.fetchone():
                    # Add user_id to categories table
                    cursor.execute("ALTER TABLE categories ADD COLUMN user_id INTEGER REFERENCES users(id);")
                    
                    # Set existing categories to a default system user
                    cursor.execute("UPDATE categories SET user_id = 1 WHERE user_id IS NULL;")
            
            conn.commit()
            
        def create_system_user_if_needed(conn):
            """Create a system user (ID=1) if it doesn't exist"""
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE id = 1")
            if not cursor.fetchone():
                cursor.execute('''
                INSERT INTO users (id, username, email, auth_provider, provider_user_id, display_name, is_admin)
                VALUES (1, 'system', 'system@example.com', 'system', 'system', 'System User', TRUE)
                ON CONFLICT (id) DO NOTHING
                ''')
                conn.commit()
                
                # Reset sequence
                cursor.execute("SELECT setval('users_id_seq', (SELECT MAX(id) FROM users), true);")
                conn.commit()

def get_connection():
    """Connect to PostgreSQL database using environment variables"""
    try:
        # Try using DATABASE_URL first
        if os.environ.get('DATABASE_URL'):
            conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
            return conn
            
        # Otherwise use individual parameters
        conn = psycopg2.connect(
            dbname=os.environ.get('POSTGRES_DB') or os.environ.get('DB_NAME'),
            user=os.environ.get('POSTGRES_USER') or os.environ.get('DB_USER'),
            password=os.environ.get('POSTGRES_PASSWORD') or os.environ.get('DB_PASSWORD'),
            host=os.environ.get('POSTGRES_HOST') or os.environ.get('DB_HOST'),
            port=os.environ.get('POSTGRES_PORT') or os.environ.get('DB_PORT')
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def rename_data_column_to_raw_data(conn):
    """Rename the 'data' column to 'raw_data' in the bookmarks table"""
    try:
        cursor = conn.cursor()
        
        # Check if bookmarks table exists
        cursor.execute("SELECT to_regclass('public.bookmarks');")
        if cursor.fetchone()[0] is None:
            logger.info("Bookmarks table doesn't exist yet. It will be created with raw_data column.")
            # Create bookmarks table with raw_data
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                id VARCHAR(255) PRIMARY KEY,
                raw_data JSONB NOT NULL,
                user_id INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            logger.info("✅ Created bookmarks table with raw_data column")
            return
            
        # Check if 'data' column exists
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bookmarks' AND column_name='data';")
        data_column_exists = cursor.fetchone() is not None
        
        # Check if 'raw_data' column exists
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bookmarks' AND column_name='raw_data';")
        raw_data_column_exists = cursor.fetchone() is not None
        
        if data_column_exists and not raw_data_column_exists:
            logger.info("Renaming 'data' column to 'raw_data' in bookmarks table")
            cursor.execute("ALTER TABLE bookmarks RENAME COLUMN data TO raw_data;")
            logger.info("✅ Successfully renamed 'data' column to 'raw_data'")
        elif not data_column_exists and not raw_data_column_exists:
            logger.warning("Neither 'data' nor 'raw_data' column exists in bookmarks table. Creating 'raw_data' column.")
            # Add the raw_data column if neither exists
            cursor.execute("ALTER TABLE bookmarks ADD COLUMN raw_data JSONB;")
            logger.info("✅ Added 'raw_data' column to bookmarks table")
        elif data_column_exists and raw_data_column_exists:
            logger.warning("Both 'data' and 'raw_data' columns exist. This is an unexpected state.")
        else:
            logger.info("Column 'raw_data' already exists in bookmarks table - no action needed")
            
    except Exception as e:
        logger.error(f"Error renaming data column: {e}")
        raise

def update_database_schema_for_multi_user():
    """Update the database schema to support multi-user functionality"""
    logger.info("Starting database schema update for multi-user support")
    
    conn = None
    try:
        conn = get_connection()
        
        # Begin transaction
        logger.info("Beginning database transaction")
        
        # 1. Create the users table
        logger.info("Creating users table if it doesn't exist")
        create_user_table(conn)
        
        # 2. Create system user (for backward compatibility)
        logger.info("Ensuring system user exists")
        create_system_user_if_needed(conn)
        
        # 3. Rename data column to raw_data if needed
        logger.info("Checking if 'data' column needs to be renamed to 'raw_data'")
        rename_data_column_to_raw_data(conn)
        
        # 4. Alter existing tables to add user_id
        logger.info("Adding user_id to existing tables")
        alter_tables_for_multi_user(conn)
        
        # 5. Create indexes for performance
        cursor = conn.cursor()
        logger.info("Creating indexes for user_id columns")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookmarks_user_id ON bookmarks(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_categories_user_id ON categories(user_id);")
        
        # 6. Create a composite unique constraint for category names per user
        logger.info("Creating unique constraint for category names per user")
        cursor.execute("""
        DO $$
        BEGIN
            BEGIN
                ALTER TABLE categories ADD CONSTRAINT unique_category_name_per_user 
                UNIQUE (name, user_id);
            EXCEPTION
                WHEN duplicate_table THEN
                    NULL;
            END;
        END $$;
        """)
        
        conn.commit()
        logger.info("Database schema update for multi-user support completed successfully")
        
    except Exception as e:
        logger.error(f"Error updating database schema: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logger.info("Running multi-user database schema update")
    update_database_schema_for_multi_user()
    logger.info("Multi-user database schema update completed") 