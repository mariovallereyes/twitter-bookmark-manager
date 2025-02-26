import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import importlib.util
import types

# Set up logging
logging_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

logger.info("="*50)
logger.info("Initializing WSGI Application for PythonAnywhere")

# IMPORTANT CHANGE: Set root path to parent directory where api_server.py is located
project_path = Path('/home/mariovallereyes').resolve()
package_path = project_path / 'twitter_bookmark_manager'

# Add project directory to Python path
if str(project_path) not in sys.path:
    sys.path.insert(0, str(project_path))
    logger.info(f"Added {project_path} to Python path")

# Add twitter_bookmark_manager to Python path too
if str(package_path) not in sys.path:
    sys.path.insert(0, str(package_path))
    logger.info(f"Added {package_path} to Python path")

# Change working directory to the package directory before any imports
os.chdir(package_path)
logger.info(f"Changed working directory to: {os.getcwd()}")

# Set PythonAnywhere environment flag
os.environ['PYTHONANYWHERE_ENVIRONMENT'] = 'true'
logger.info("Set PYTHONANYWHERE_ENVIRONMENT flag")

# CRITICAL: Block ChromaDB imports to prevent SQLite version errors
class DummyChromaDB:
    """Dummy module that raises an import error when accessed"""
    def __getattr__(self, name):
        raise ImportError("ChromaDB is not supported on PythonAnywhere due to SQLite version constraints")

# Add the dummy ChromaDB module to sys.modules to block any direct imports
sys.modules['chromadb'] = DummyChromaDB()
logger.info("✅ Blocked direct ChromaDB imports")

# HUGGINGFACE_HUB FIX: Create compatibility layer for huggingface_hub to fix missing cached_download
try:
    # Import the real huggingface_hub module first
    import huggingface_hub
    
    # Check if cached_download is missing
    if not hasattr(huggingface_hub, 'cached_download'):
        logger.info("Patching huggingface_hub with cached_download compatibility function")
        
        # In newer versions, cached_download was replaced with hf_hub_download
        if hasattr(huggingface_hub, 'hf_hub_download'):
            # Create a compatibility function that maps to the new API
            def cached_download(url_or_filename, **kwargs):
                logger.info(f"Redirecting cached_download to hf_hub_download for: {url_or_filename}")
                # Extract repo_id and filename from URL if possible, or use defaults
                repo_id = kwargs.get('repo_id', 'sentence-transformers')
                filename = url_or_filename.split('/')[-1] if isinstance(url_or_filename, str) else 'model'
                return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
            
            # Add the compatibility function to the module
            huggingface_hub.cached_download = cached_download
            logger.info("✅ Successfully patched huggingface_hub with cached_download compatibility")
        else:
            logger.warning("Could not patch huggingface_hub - neither cached_download nor hf_hub_download found")
    else:
        logger.info("huggingface_hub already has cached_download, no patch needed")
        
except ImportError as e:
    logger.error(f"Error importing huggingface_hub: {e}")
    # If we can't import the real module, create a mock one
    class MockHuggingFaceHub:
        """Mock module for huggingface_hub with required functionality"""
        def __init__(self):
            # Add all required attributes and functions
            self.cached_download = self._mock_cached_download
            self.hf_hub_url = self._mock_hf_hub_url
            self.HfApi = type('HfApi', (), {})
            self.HfFolder = type('HfFolder', (), {})
            self.Repository = type('Repository', (), {})
            
        def _mock_cached_download(self, url_or_filename, **kwargs):
            logger.warning(f"Mock cached_download called for: {url_or_filename}")
            # Return a mock path that will satisfy the immediate import needs
            return str(package_path / 'mock_model_path')
            
        def _mock_hf_hub_url(self, *args, **kwargs):
            return "https://mock-huggingface-url.com/model"
            
    # Only replace the module if it's not already loaded correctly
    if 'huggingface_hub' not in sys.modules or not hasattr(sys.modules['huggingface_hub'], 'cached_download'):
        mock_module = MockHuggingFaceHub()
        # Add missing attributes expected by sentence-transformers
        for name in ['HfApi', 'HfFolder', 'Repository', 'hf_hub_url', 'cached_download']:
            if hasattr(mock_module, name):
                setattr(sys.modules.get('huggingface_hub', mock_module), name, getattr(mock_module, name))
        
        if 'huggingface_hub' not in sys.modules:
            sys.modules['huggingface_hub'] = mock_module
            
        logger.info("✅ Installed mock huggingface_hub module with required functionality")

# Pre-load our PythonAnywhere database modules to ensure they're used
logger.info("Loading PythonAnywhere-specific database modules...")

# Create the database namespace if it doesn't exist
if 'database' not in sys.modules:
    database_module = types.ModuleType('database')
    database_module.__path__ = [str(package_path / 'database')]
    sys.modules['database'] = database_module
    logger.info("Created database namespace")

# Load the models module
try:
    # Path to models module
    models_path = package_path / 'database' / 'models.py'
    
    if models_path.exists():
        # Create a module spec and load it
        spec = importlib.util.spec_from_file_location('database.models', models_path)
        if spec:
            models_module = importlib.util.module_from_spec(spec)
            # Add it to sys.modules before executing
            sys.modules['database.models'] = models_module
            # Execute the module code
            spec.loader.exec_module(models_module)
            logger.info("✅ Successfully loaded database models module")
        else:
            logger.error(f"Failed to create spec for {models_path}")
    else:
        logger.error(f"Models module not found: {models_path}")
except Exception as e:
    logger.error(f"Error loading models module: {e}")
    import traceback
    logger.error(traceback.format_exc())

# Load the PythonAnywhere db module
try:
    # Path to PA-specific modules - FIXED: removed duplicate twitter_bookmark_manager
    pa_db_path = package_path / 'deployment' / 'pythonanywhere' / 'database' / 'db_pa.py'
    
    if pa_db_path.exists():
        # Create a module spec and load it
        spec = importlib.util.spec_from_file_location('database.db', pa_db_path)
        if spec:
            db_module = importlib.util.module_from_spec(spec)
            # Add it to sys.modules before executing
            sys.modules['database.db'] = db_module
            # Execute the module code
            spec.loader.exec_module(db_module)
            logger.info("✅ Successfully loaded PythonAnywhere db module")
        else:
            logger.error(f"Failed to create spec for {pa_db_path}")
    else:
        logger.error(f"PythonAnywhere db module not found: {pa_db_path}")
except Exception as e:
    logger.error(f"Error loading PythonAnywhere db module: {e}")
    import traceback
    logger.error(traceback.format_exc())

# Load the PythonAnywhere vector store module
try:
    # Path to PA-specific vector store - FIXED: removed duplicate twitter_bookmark_manager
    pa_vector_store_path = package_path / 'deployment' / 'pythonanywhere' / 'database' / 'vector_store_pa.py'
    
    if pa_vector_store_path.exists():
        # Create a module spec and load it
        spec = importlib.util.spec_from_file_location('database.vector_store', pa_vector_store_path)
        if spec:
            vector_store_module = importlib.util.module_from_spec(spec)
            # Add it to sys.modules before executing
            sys.modules['database.vector_store'] = vector_store_module
            # Execute the module code
            spec.loader.exec_module(vector_store_module)
            
            # Make VectorStore available as ChromaStore for compatibility
            if hasattr(vector_store_module, 'VectorStore'):
                vector_store_module.ChromaStore = vector_store_module.VectorStore
                logger.info("✅ Successfully loaded PythonAnywhere vector store and aliased VectorStore as ChromaStore")
            else:
                logger.error("VectorStore class not found in vector_store_pa.py")
        else:
            logger.error(f"Failed to create spec for {pa_vector_store_path}")
    else:
        logger.error(f"PythonAnywhere vector store module not found: {pa_vector_store_path}")
except Exception as e:
    logger.error(f"Error loading PythonAnywhere vector store module: {e}")
    import traceback
    logger.error(traceback.format_exc())

# Install fallback import hook for any imports we missed with pre-loading
original_import = __import__

def import_hook(name, *args, **kwargs):
    if name == 'database.models' and 'database.models' not in sys.modules:
        logger.info("🔄 Import hook redirecting database.models")
        models_path = package_path / 'database' / 'models.py'
        spec = importlib.util.spec_from_file_location('database.models', models_path)
        if spec:
            models_module = importlib.util.module_from_spec(spec)
            sys.modules['database.models'] = models_module
            spec.loader.exec_module(models_module)
            return models_module
            
    if name == 'database.db' and 'database.db' not in sys.modules:
        logger.info("🔄 Import hook redirecting database.db -> PA-specific implementation")
        # Import our PythonAnywhere-specific module instead - FIXED: removed duplicate twitter_bookmark_manager
        pa_db_path = package_path / 'deployment' / 'pythonanywhere' / 'database' / 'db_pa.py'
        spec = importlib.util.spec_from_file_location('database.db', pa_db_path)
        if spec:
            db_module = importlib.util.module_from_spec(spec)
            sys.modules['database.db'] = db_module
            spec.loader.exec_module(db_module)
            return db_module
    
    if name == 'database.vector_store' and 'database.vector_store' not in sys.modules:
        logger.info("🔄 Import hook redirecting database.vector_store -> PA-specific implementation")
        # Import our PythonAnywhere-specific vector store module instead - FIXED: removed duplicate twitter_bookmark_manager
        pa_vector_store_path = package_path / 'deployment' / 'pythonanywhere' / 'database' / 'vector_store_pa.py'
        spec = importlib.util.spec_from_file_location('database.vector_store', pa_vector_store_path)
        if spec:
            vector_store_module = importlib.util.module_from_spec(spec)
            sys.modules['database.vector_store'] = vector_store_module
            spec.loader.exec_module(vector_store_module)
            vector_store_module.ChromaStore = vector_store_module.VectorStore
            return vector_store_module
    
    if name == 'chromadb':
        raise ImportError("ChromaDB is not supported on PythonAnywhere")
    
    # Special handling for huggingface_hub imports to ensure they have cached_download
    if name == 'huggingface_hub' and 'huggingface_hub' in sys.modules:
        # Make sure it has the cached_download function
        if not hasattr(sys.modules['huggingface_hub'], 'cached_download'):
            logger.info("🔄 Import hook adding cached_download to huggingface_hub")
            # Add a simple compatibility function
            def cached_download(url_or_filename, **kwargs):
                logger.warning(f"Mock cached_download called via import hook for: {url_or_filename}")
                return str(package_path / 'mock_model_path')
            
            sys.modules['huggingface_hub'].cached_download = cached_download
        return sys.modules['huggingface_hub']
    
    # For any other import, use the original mechanism
    return original_import(name, *args, **kwargs)

# Replace the built-in import function with our hook
sys.modules['builtins'].__import__ = import_hook
logger.info("✅ Enhanced import hook installed for database modules")

# Load PythonAnywhere-specific environment variables
env_path = package_path / '.env.pythonanywhere'
if not env_path.exists():
    error_msg = f"PythonAnywhere environment file not found at {env_path}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

# Force load PythonAnywhere environment (override=True ensures it takes precedence)
load_dotenv(env_path, override=True)
logger.info(f"Loaded environment variables from {env_path}")

# Verify critical environment variables
required_vars = [
    'POSTGRES_PASSWORD',
    'DATABASE_URL',
    'PA_BASE_DIR',
    'GEMINI_API_KEY',
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

logger.info("All required environment variables are present")
logger.info("="*50)

# Import your Flask app
try:
    # Import from the root directory where api_server.py is located
    from api_server import app as application
    logger.info("✅ Successfully imported Flask application from api_server.py")
except Exception as e:
    logger.error(f"Error importing application: {e}")
    import traceback
    logger.error(traceback.format_exc())
    raise