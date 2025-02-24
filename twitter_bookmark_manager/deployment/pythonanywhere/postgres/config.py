"""
PostgreSQL and Vector Store configuration settings for PythonAnywhere deployment.
"""
import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load PythonAnywhere environment
PA_BASE_DIR = '/home/mariovallereyes/twitter_bookmark_manager'
env_path = Path(PA_BASE_DIR) / '.env.pythonanywhere'
if env_path.exists():
    load_dotenv(env_path, override=True)

# PostgreSQL Connection Settings
PG_CONFIG = {
    'dbname': 'mariovallereyes$default',
    'user': 'mariovallereyes',
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': 'mariovallereyes-4374.postgres.pythonanywhere-services.com',
    'port': '14374'
}

# SQLAlchemy URL format with URL-encoded password
def get_database_url() -> str:
    """Get database URL with proper password encoding"""
    password = quote_plus(PG_CONFIG['password']) if PG_CONFIG['password'] else ''
    return (
        f"postgresql://{PG_CONFIG['user']}:{password}@"
        f"{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    )

# Vector Store Settings
VECTOR_STORE_CONFIG = {
    'persist_directory': os.path.join(PA_BASE_DIR, 'database', 'vector_db')
}

# Environment detection
def is_pythonanywhere() -> bool:
    """Check if we're running on PythonAnywhere"""
    return os.path.exists('/home/mariovallereyes') 