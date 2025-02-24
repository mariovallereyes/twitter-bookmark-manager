import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the absolute path to the project root
project_path = Path('/home/mariovallereyes/twitter_bookmark_manager').resolve()
package_path = project_path

# Add project directory to Python path
if str(project_path) not in sys.path:
    sys.path.insert(0, str(project_path))

# Change working directory to the package directory before any imports
os.chdir(package_path)

# Load PythonAnywhere-specific environment variables
env_path = project_path / '.env.pythonanywhere'
if not env_path.exists():
    raise RuntimeError(f"PythonAnywhere environment file not found at {env_path}")

# Force load PythonAnywhere environment (override=True ensures it takes precedence)
load_dotenv(env_path, override=True)

# Verify critical environment variables
required_vars = [
    'POSTGRES_PASSWORD',
    'DATABASE_URL',
    'PA_BASE_DIR',
    'GEMINI_API_KEY',
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Import your Flask app
from api_server import app as application 