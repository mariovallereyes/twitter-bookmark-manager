import sys
import os
from pathlib import Path

# Get the absolute path to the project root
project_path = Path('/home/mariovallereyes/twitter_bookmark_manager').resolve()
package_path = project_path

# Add project directory to Python path
if str(project_path) not in sys.path:
    sys.path.insert(0, str(project_path))

# Change working directory to the package directory before any imports
os.chdir(package_path)

# Import your Flask app
from api_server import app as application 