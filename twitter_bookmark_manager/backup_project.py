import py7zr
from datetime import datetime
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def should_exclude(path):
    exclude_patterns = [
        'venv',
        '__pycache__',
        '.git',
        '.pytest_cache',
        '.mypy_cache',
        '.coverage',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.DS_Store',
        'models',
        'sentence-transformers',
        'temp_uploads',
        'update_log_*.txt'
    ]
    
    return any(pattern in path for pattern in exclude_patterns)

def backup_project():
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path('backups')
        backup_dir.mkdir(exist_ok=True)
        
        backup_name = backup_dir / f'twitter_bookmark_manager_backup_{timestamp}.7z'
        
        # Log start
        logger.info(f"Starting backup process...")
        
        # Count only relevant files
        total_files = 0
        files_to_backup = []
        for root, dirs, files in os.walk('.'):
            if should_exclude(root):
                continue
            for file in files:
                if not should_exclude(file):
                    total_files += 1
                    files_to_backup.append(os.path.join(root, file))
        
        logger.info(f"Found {total_files} files to backup")
        
        # Create archive with filtered files
        with py7zr.SevenZipFile(backup_name, 'w') as archive:
            logger.info("Creating archive...")
            for file in files_to_backup:
                logger.info(f"Adding: {file}")
                archive.write(file)
            logger.info("Archive creation completed")
        
        logger.info(f"✓ Backup created successfully: {backup_name}")
        logger.info(f"Backup size: {backup_name.stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ Error during backup: {str(e)}")
        raise

if __name__ == "__main__":
    backup_project()