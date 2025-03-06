#!/usr/bin/env python
"""
Fix Categories Schema for Twitter Bookmark Manager
This script specifically targets and fixes issues with the categories and bookmark_categories tables.
"""

import os
import sys
import logging
import psycopg2
from psycopg2 import sql
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('fix_categories_schema')

def get_db_connection():
    """Get a direct connection to the database using Railway environment variables"""
    # First try DATABASE_URL
    if os.environ.get('DATABASE_URL'):
        logger.info("Connecting using DATABASE_URL")
        return psycopg2.connect(os.environ.get('DATABASE_URL'))
    
    # Try Railway PostgreSQL environment variables
    logger.info("Connecting using Railway PostgreSQL environment variables")
    return psycopg2.connect(
        host="postgres.railway.internal",
        dbname=os.environ.get('PGDATABASE', 'railway'),
        user=os.environ.get('PGUSER', 'postgres'),
        password=os.environ.get('PGPASSWORD'),
        port=os.environ.get('PGPORT', '5432')
    )

def fix_categories_table(conn):
    """Fix the categories table schema"""
    cursor = conn.cursor()
    logger.info("Checking categories table...")
    
    # Check if description column exists
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='categories' AND column_name='description';")
    if not cursor.fetchone():
        logger.info("Adding missing 'description' column to categories table")
        cursor.execute("ALTER TABLE categories ADD COLUMN description TEXT;")
        logger.info("✅ Added description column")
    else:
        logger.info("✅ Description column already exists")
    
    # Check if updated_at column exists
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='categories' AND column_name='updated_at';")
    if not cursor.fetchone():
        logger.info("Adding missing 'updated_at' column to categories table")
        cursor.execute("ALTER TABLE categories ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
        logger.info("✅ Added updated_at column")
    else:
        logger.info("✅ Updated_at column already exists")
    
    conn.commit()
    logger.info("Categories table schema fixed")

def fix_bookmark_categories_table(conn):
    """Fix the bookmark_categories table schema"""
    cursor = conn.cursor()
    logger.info("Checking bookmark_categories table...")
    
    # Check if id column exists
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bookmark_categories' AND column_name='id';")
    if not cursor.fetchone():
        logger.info("Adding missing 'id' column to bookmark_categories table")
        cursor.execute("ALTER TABLE bookmark_categories ADD COLUMN id SERIAL PRIMARY KEY;")
        logger.info("✅ Added id column")
    else:
        logger.info("✅ Id column already exists")
    
    # Check if user_id column exists
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bookmark_categories' AND column_name='user_id';")
    if not cursor.fetchone():
        logger.info("Adding missing 'user_id' column to bookmark_categories table")
        cursor.execute("ALTER TABLE bookmark_categories ADD COLUMN user_id INTEGER REFERENCES users(id);")
        logger.info("✅ Added user_id column")
    else:
        logger.info("✅ User_id column already exists")
    
    # Check if created_at column exists
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bookmark_categories' AND column_name='created_at';")
    if not cursor.fetchone():
        logger.info("Adding missing 'created_at' column to bookmark_categories table")
        cursor.execute("ALTER TABLE bookmark_categories ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
        logger.info("✅ Added created_at column")
    else:
        logger.info("✅ Created_at column already exists")
    
    # Check if bookmark_id is character varying and needs to be converted to integer
    cursor.execute("SELECT data_type FROM information_schema.columns WHERE table_name='bookmark_categories' AND column_name='bookmark_id';")
    data_type = cursor.fetchone()
    if data_type and data_type[0] == 'character varying':
        logger.info("Bookmark_id is character varying, investigating if conversion is possible...")
        
        # First check if the table is actually empty
        cursor.execute("SELECT COUNT(*) FROM bookmark_categories;")
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("Table is empty, recreating bookmark_id column with correct data type")
            cursor.execute("ALTER TABLE bookmark_categories DROP COLUMN bookmark_id;")
            cursor.execute("ALTER TABLE bookmark_categories ADD COLUMN bookmark_id INTEGER REFERENCES bookmarks(id);")
            logger.info("✅ Recreated bookmark_id column as INTEGER")
        else:
            logger.warning("Cannot convert bookmark_id to INTEGER as the table has existing data")
            logger.info("Creating an additional integer version of the column for compatibility")
            try:
                cursor.execute("ALTER TABLE bookmark_categories ADD COLUMN bookmark_id_int INTEGER REFERENCES bookmarks(id);")
                logger.info("✅ Added bookmark_id_int column as INTEGER for compatibility")
            except Exception as e:
                logger.error(f"Failed to add bookmark_id_int column: {e}")
    
    # Add unique constraint if missing
    cursor.execute("""
    SELECT con.conname
    FROM pg_constraint con
    JOIN pg_class rel ON rel.oid = con.conrelid
    JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
    WHERE rel.relname = 'bookmark_categories' 
    AND con.contype = 'u'
    AND array_to_string(con.conkey, ',') = '2,3'
    """)
    
    if not cursor.fetchone():
        logger.info("Adding unique constraint on bookmark_id and category_id")
        try:
            cursor.execute("""
            ALTER TABLE bookmark_categories 
            ADD CONSTRAINT unique_bookmark_category 
            UNIQUE (bookmark_id, category_id);
            """)
            logger.info("✅ Added unique constraint")
        except Exception as e:
            logger.error(f"Failed to add unique constraint: {e}")
    else:
        logger.info("✅ Unique constraint already exists")
    
    conn.commit()
    logger.info("Bookmark_categories table schema fixed")

def create_indexes(conn):
    """Create helpful indexes"""
    cursor = conn.cursor()
    logger.info("Creating useful indexes if they don't exist...")
    
    # Check and create indexes to improve performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_categories_user_id ON categories(user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookmark_categories_bookmark_id ON bookmark_categories(bookmark_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookmark_categories_category_id ON bookmark_categories(category_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookmark_categories_user_id ON bookmark_categories(user_id);")
    
    conn.commit()
    logger.info("✅ Indexes created or already exist")

def fix_schema():
    """Fix categories and bookmark_categories tables schema"""
    conn = None
    try:
        logger.info("Starting categories schema fix...")
        conn = get_db_connection()
        
        # 1. Fix categories table
        fix_categories_table(conn)
        
        # 2. Fix bookmark_categories table
        fix_bookmark_categories_table(conn)
        
        # 3. Create helpful indexes
        create_indexes(conn)
        
        logger.info("✅ Schema fix completed successfully")
        return {"success": True, "message": "Categories schema fixed successfully"}
    
    except Exception as e:
        logger.error(f"❌ Error fixing schema: {e}")
        logger.error(traceback.format_exc())
        if conn:
            conn.rollback()
        return {"success": False, "error": str(e)}
    
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    result = fix_schema()
    print(f"Schema fix result: {result}") 