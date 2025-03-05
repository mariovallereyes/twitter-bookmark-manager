#!/usr/bin/env python3
"""
Database Diagnostics Tool for Twitter Bookmark Manager
This script can be run manually to test database connections, 
check schema, and run basic diagnostics.
"""

import os
import sys
import logging
import json
import psycopg2
import traceback
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_diagnostics')

def load_environment_variables():
    """Load environment variables from various possible locations"""
    env_paths = [
        Path(os.path.join(os.path.dirname(__file__), ".env.final")).resolve(),
        Path('/home/mariovallereyes/twitter_bookmark_manager/.env.final').resolve(),
        Path(__file__).parents[3] / '.env.final'
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"✅ Loaded environment variables from {env_path}")
            return True
    
    logger.warning("No .env.final file found in any expected location, using environment variables only")
    return False

def get_connection_url():
    """Get database connection URL"""
    # First check if DATABASE_URL is provided
    DATABASE_URL = os.getenv("DATABASE_URL")
    if DATABASE_URL:
        logger.info("Using DATABASE_URL for connection")
        return DATABASE_URL

    # Fall back to individual components
    logger.info("DATABASE_URL not found, using individual connection parameters")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_PORT = os.getenv("DB_PORT", "14374")  # Default PostgreSQL port for Railway
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        missing = []
        if not DB_USER: missing.append("DB_USER")
        if not DB_PASSWORD: missing.append("DB_PASSWORD")
        if not DB_HOST: missing.append("DB_HOST")
        if not DB_NAME: missing.append("DB_NAME")
        error_msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Create connection string from individual components
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=prefer"

def check_direct_connection():
    """Check direct connection to database using psycopg2"""
    logger.info("Testing direct connection with psycopg2...")
    
    try:
        # First check if DATABASE_URL is provided
        DATABASE_URL = os.getenv("DATABASE_URL")
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL)
        else:
            # Use individual components
            conn = psycopg2.connect(
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT", "14374"),
                database=os.getenv("DB_NAME")
            )
        
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
            logger.info("✅ Direct database connection successful!")
        
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"❌ Direct connection failed: {str(e)}")
        traceback.print_exc()
        return False

def check_sqlalchemy_connection():
    """Check connection using SQLAlchemy"""
    logger.info("Testing SQLAlchemy connection...")
    
    try:
        db_url = get_connection_url()
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=600,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "application_name": "TwitterBookmarkManager_Diagnostics"
            }
        )
        
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1
            logger.info("✅ SQLAlchemy connection successful!")
        
        return True, engine
    
    except Exception as e:
        logger.error(f"❌ SQLAlchemy connection failed: {str(e)}")
        traceback.print_exc()
        return False, None

def check_schema(engine):
    """Check database schema"""
    logger.info("Checking database schema...")
    
    try:
        with engine.connect() as connection:
            # Check bookmarks table schema
            result = connection.execute(
                text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'bookmarks'")
            )
            
            columns = [{"column_name": row[0], "data_type": row[1]} for row in result]
            
            logger.info(f"✅ Found {len(columns)} columns in bookmarks table:")
            for col in columns:
                logger.info(f"  - {col['column_name']} ({col['data_type']})")
            
            # Check if our critical 'raw_data' column exists
            raw_data_column = next((col for col in columns if col['column_name'] == 'raw_data'), None)
            if raw_data_column:
                logger.info(f"✅ Found critical 'raw_data' column with type {raw_data_column['data_type']}")
            else:
                logger.error("❌ Critical 'raw_data' column not found in bookmarks table!")
            
            # Check if 'data' column exists (which should not)
            data_column = next((col for col in columns if col['column_name'] == 'data'), None)
            if data_column:
                logger.warning(f"⚠️ Found unexpected 'data' column with type {data_column['data_type']}")
            else:
                logger.info("✅ No 'data' column found in bookmarks table (as expected)")
            
            return columns
    
    except Exception as e:
        logger.error(f"❌ Error checking schema: {str(e)}")
        traceback.print_exc()
        return None

def check_connection_pool(engine):
    """Check connection pool status"""
    logger.info("Checking connection pool status...")
    
    try:
        pool_stats = {
            "overflow": engine.pool.overflow(),
            "checkedin": engine.pool.checkedin(),
            "checkedout": engine.pool.checkedout(),
            "size": engine.pool.size(),
            "total_connections": engine.pool.checkedin() + engine.pool.checkedout()
        }
        
        logger.info(f"✅ Pool stats: {json.dumps(pool_stats, indent=2)}")
        return pool_stats
    
    except Exception as e:
        logger.error(f"❌ Error checking connection pool: {str(e)}")
        traceback.print_exc()
        return None

def count_bookmarks(engine):
    """Count bookmarks in the database"""
    logger.info("Counting bookmarks...")
    
    try:
        with engine.connect() as connection:
            # Count total bookmarks
            result = connection.execute(text("SELECT COUNT(*) FROM bookmarks"))
            total_count = result.scalar()
            
            # Count bookmarks per user
            result = connection.execute(text(
                "SELECT user_id, COUNT(*) FROM bookmarks GROUP BY user_id ORDER BY COUNT(*) DESC"
            ))
            user_counts = [{"user_id": row[0], "bookmark_count": row[1]} for row in result]
            
            logger.info(f"✅ Found {total_count} total bookmarks")
            logger.info(f"✅ Bookmarks by user:")
            for uc in user_counts:
                logger.info(f"  - User {uc['user_id']}: {uc['bookmark_count']} bookmarks")
            
            return total_count, user_counts
    
    except Exception as e:
        logger.error(f"❌ Error counting bookmarks: {str(e)}")
        traceback.print_exc()
        return None, None

def test_sample_query(engine):
    """Test a sample query to retrieve a bookmark"""
    logger.info("Testing sample query...")
    
    try:
        with engine.connect() as connection:
            # Get a sample bookmark
            result = connection.execute(text(
                "SELECT id, text, created_at, author_name, author_username, raw_data, user_id FROM bookmarks LIMIT 1"
            ))
            
            row = result.fetchone()
            if row:
                logger.info(f"✅ Successfully retrieved sample bookmark with ID: {row[0]}")
                logger.info(f"  - Tweet text: {row[1][:50]}...")
                logger.info(f"  - Author: {row[3]} (@{row[4]})")
                logger.info(f"  - User ID: {row[6]}")
                return True
            else:
                logger.warning("⚠️ No bookmarks found in the database")
                return False
    
    except Exception as e:
        logger.error(f"❌ Error testing sample query: {str(e)}")
        traceback.print_exc()
        return False

def run_diagnostics():
    """Run all diagnostics"""
    logger.info("Starting database diagnostics...")
    
    # Load environment variables
    load_environment_variables()
    
    # Check direct connection
    direct_connection_ok = check_direct_connection()
    
    # Check SQLAlchemy connection
    sqlalchemy_connection_ok, engine = check_sqlalchemy_connection()
    
    if not sqlalchemy_connection_ok:
        logger.error("Cannot continue diagnostics without a working connection.")
        return
    
    # Check schema
    schema = check_schema(engine)
    
    # Check connection pool
    pool_stats = check_connection_pool(engine)
    
    # Count bookmarks
    total_bookmarks, user_counts = count_bookmarks(engine)
    
    # Test sample query
    sample_query_ok = test_sample_query(engine)
    
    # Print summary
    logger.info("\n----- DIAGNOSTICS SUMMARY -----")
    logger.info(f"Direct connection: {'✅ OK' if direct_connection_ok else '❌ FAILED'}")
    logger.info(f"SQLAlchemy connection: {'✅ OK' if sqlalchemy_connection_ok else '❌ FAILED'}")
    logger.info(f"Database schema: {'✅ OK' if schema else '❌ FAILED'}")
    logger.info(f"Connection pool: {'✅ OK' if pool_stats else '❌ FAILED'}")
    logger.info(f"Total bookmarks: {total_bookmarks if total_bookmarks is not None else 'UNKNOWN'}")
    logger.info(f"Sample query: {'✅ OK' if sample_query_ok else '❌ FAILED'}")
    logger.info("-----------------------------\n")

if __name__ == "__main__":
    run_diagnostics() 