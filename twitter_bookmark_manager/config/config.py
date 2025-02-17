from pathlib import Path
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent

class Config:
    """Base configuration class"""
    # Server settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY')
    
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/database/twitter_bookmarks.db')
    
    # Model settings
    MISTRAL_MODEL_PATH = os.getenv('MISTRAL_MODEL_PATH', 'models/mistral-7b-instruct-v0.1.Q4_K_M.gguf')
    CHAT_MODEL = os.getenv('CHAT_MODEL', 'mistral')  # 'mistral' or 'gemini'
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Twitter API settings
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    TWITTER_CLIENT_ID = os.getenv('TWITTER_CLIENT_ID')
    TWITTER_CLIENT_SECRET = os.getenv('TWITTER_CLIENT_SECRET')
    TWITTER_REDIRECT_URI = os.getenv('TWITTER_REDIRECT_URI', 'https://localhost:5000/callback')
    
    # Security settings
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
    
    # API rate limits (requests per minute)
    RATE_LIMIT_SEARCH = 60
    RATE_LIMIT_CHAT = 30
    RATE_LIMIT_UPLOAD = 10
    
    # File size limits
    MAX_UPLOAD_SIZE_MB = 10
    
    # Backup settings
    BACKUP_DIR = BASE_DIR / 'backups'
    JSON_HISTORY_DIR = BASE_DIR / 'database' / 'json_history'
    EXCLUDED_BACKUP_PATHS = [
        'venv',
        '__pycache__',
        '.git',
        'models',
        'temp_uploads'
    ]
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate required configuration settings"""
        missing = []
        
        # Required settings that must be present
        required = [
            'SECRET_KEY',
            'ENCRYPTION_KEY',
        ]
        
        # Check required settings
        for setting in required:
            if not getattr(cls, setting):
                missing.append(setting)
        
        # Model-specific validation
        if cls.CHAT_MODEL == 'gemini' and not cls.GEMINI_API_KEY:
            missing.append('GEMINI_API_KEY')
        
        if missing:
            raise ValueError(f"Missing required configuration settings: {', '.join(missing)}")
        
        return {
            'debug': cls.DEBUG,
            'database_url': cls.DATABASE_URL,
            'chat_model': cls.CHAT_MODEL,
            'rate_limits': {
                'search': cls.RATE_LIMIT_SEARCH,
                'chat': cls.RATE_LIMIT_CHAT,
                'upload': cls.RATE_LIMIT_UPLOAD
            }
        }

# Load and validate configuration
config = Config()
config.validate() 