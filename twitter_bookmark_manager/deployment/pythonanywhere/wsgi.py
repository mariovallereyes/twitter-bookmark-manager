import sys
import importlib
from pathlib import Path

package_path = Path(__file__).resolve().parent.parent

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
        # Import our PythonAnywhere-specific module instead
        pa_db_path = package_path / 'deployment' / 'pythonanywhere' / 'database' / 'db_pa.py'
        spec = importlib.util.spec_from_file_location('database.db', pa_db_path)
        if spec:
            db_module = importlib.util.module_from_spec(spec)
            sys.modules['database.db'] = db_module
            spec.loader.exec_module(db_module)
            return db_module
    
    if name == 'database.vector_store' and 'database.vector_store' not in sys.modules:
        logger.info("🔄 Import hook redirecting database.vector_store -> PA-specific implementation")
        # Import our PythonAnywhere-specific vector store module instead
        pa_vector_store_path = package_path / 'deployment' / 'pythonanywhere' / 'database' / 'vector_store_pa.py'
        spec = importlib.util.spec_from_file_location('database.vector_store', pa_vector_store_path)
        if spec:
            vector_store_module = importlib.util.module_from_spec(spec)
            sys.modules['database.vector_store'] = vector_store_module
            spec.loader.exec_module(vector_store_module)
            vector_store_module.ChromaStore = vector_store_module.VectorStore
            return vector_store_module
            
    if name == 'core.search' and 'core.search' not in sys.modules:
        logger.info("🔄 Import hook redirecting core.search -> PA-specific implementation")
        # Import our PythonAnywhere-specific search module instead
        pa_search_path = package_path / 'deployment' / 'pythonanywhere' / 'database' / 'search_pa.py'
        spec = importlib.util.spec_from_file_location('core.search', pa_search_path)
        if spec:
            search_module = importlib.util.module_from_spec(spec)
            sys.modules['core.search'] = search_module
            spec.loader.exec_module(search_module)
            return search_module
    
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