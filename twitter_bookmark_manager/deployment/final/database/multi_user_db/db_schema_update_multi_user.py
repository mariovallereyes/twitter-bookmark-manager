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

# Import the user model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deployment.final.database.multi_user_db.user_model_final import create_user_table, alter_tables_for_multi_user, create_system_user_if_needed

def get_connection():
    """Connect to PostgreSQL database using environment variables"""
    try:
        conn = psycopg2.connect(
            dbname=os.environ.get('POSTGRES_DB'),
            user=os.environ.get('POSTGRES_USER'),
            password=os.environ.get('POSTGRES_PASSWORD'),
            host=os.environ.get('POSTGRES_HOST'),
            port=os.environ.get('POSTGRES_PORT')
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def update_database_schema_for_multi_user():
    """Update the database schema to support multi-user functionality"""
    logger.info("Starting database schema update for multi-user support")
    
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
        
        # 3. Alter existing tables to add user_id
        logger.info("Adding user_id to existing tables")
        alter_tables_for_multi_user(conn)
        
        # 4. Create indexes for performance
        cursor = conn.cursor()
        logger.info("Creating indexes for user_id columns")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookmarks_user_id ON bookmarks(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_categories_user_id ON categories(user_id);")
        
        # 5. Create a composite unique constraint for category names per user
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