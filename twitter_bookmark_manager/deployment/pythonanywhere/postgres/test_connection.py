"""
Test PostgreSQL connection for PythonAnywhere deployment.
"""
import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env.pythonanywhere
project_root = Path('/home/mariovallereyes/twitter_bookmark_manager').resolve()
env_path = project_root / '.env.pythonanywhere'
if env_path.exists():
    load_dotenv(env_path, override=True)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f"No .env.pythonanywhere file found at {env_path}")

def setup_database():
    """Set up the database and user"""
    try:
        # Connection parameters for superuser
        conn_params = {
            'dbname': 'mariovallereyes$default',
            'user': 'super',
            'password': os.getenv('POSTGRES_PASSWORD'),
            'host': 'mariovallereyes-4374.postgres.pythonanywhere-services.com',
            'port': '14374'
        }
        
        # Connect as superuser
        logger.info("Connecting as superuser...")
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        try:
            # Create regular user if not exists
            logger.info("Creating regular user...")
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mariovallereyes') THEN
                        CREATE USER mariovallereyes WITH PASSWORD %s;
                    END IF;
                END
                $$;
            """, (os.getenv('POSTGRES_PASSWORD'),))
            
            # Grant privileges
            logger.info("Granting privileges...")
            cur.execute("""
                GRANT ALL PRIVILEGES ON DATABASE "mariovallereyes$default" TO mariovallereyes;
                GRANT USAGE ON SCHEMA public TO mariovallereyes;
                GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mariovallereyes;
                GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mariovallereyes;
            """)
            
            logger.info("Database setup completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during database setup: {str(e)}")
            raise
        finally:
            cur.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        raise

def test_postgres_connection(use_super=False):
    """Test connection to PostgreSQL database"""
    try:
        # Debug: Check environment variable
        password = os.getenv('POSTGRES_PASSWORD')
        logger.info(f"Password from env: {'SET' if password else 'NOT SET'}")
        if password:
            logger.info(f"Password length: {len(password)}")
            logger.info(f"First and last chars: {password[0]}...{password[-1]}")
        
        # Connection parameters
        username = 'super' if use_super else 'mariovallereyes'
        conn_params = {
            'dbname': 'mariovallereyes$default',
            'user': username,
            'password': password,
            'host': 'mariovallereyes-4374.postgres.pythonanywhere-services.com',
            'port': '14374'
        }
        
        logger.info(f"Testing connection as {username}...")
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Test query
        cur = conn.cursor()
        logger.info("Executing test query...")
        cur.execute('SELECT version();')
        version = cur.fetchone()
        logger.info(f"PostgreSQL version: {version[0]}")
        
        # Close connections
        cur.close()
        conn.close()
        logger.info("Connection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # First test superuser connection
        logger.info("="*50)
        logger.info("Testing superuser connection...")
        if test_postgres_connection(use_super=True):
            # If superuser connection works, set up database
            logger.info("="*50)
            logger.info("Setting up database...")
            setup_database()
            
            # Test regular user connection
            logger.info("="*50)
            logger.info("Testing regular user connection...")
            if test_postgres_connection(use_super=False):
                logger.info("All tests passed successfully!")
                sys.exit(0)
            else:
                logger.error("Regular user connection failed!")
                sys.exit(1)
        else:
            logger.error("Superuser connection failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1) 