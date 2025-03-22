"""
Search configuration module for the chat system.
Provides default configuration for search parameters and caching settings.
"""

import os
import logging
from typing import Any, Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)

class SearchConfig:
    """Configuration for search parameters and caching"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
        self._load_defaults()
        logger.info("Initialized SearchConfig with defaults")
    
    def _load_defaults(self) -> None:
        """Load default configuration"""
        default_config = {
            'vector_search': {
                'enabled': True,
                'alpha': 0.7,  # Weight for vector search vs. keyword search
                'retry_attempts': 2,
                'cache_ttl': 3600,  # 1 hour cache TTL
            },
            'sql_search': {
                'enabled': True,
                'retry_attempts': 3,
                'retry_backoff': 1.5,  # Exponential backoff factor
            },
            'hybrid_search': {
                'enabled': True,
                'reranking_enabled': False,  # Enable when reranking is implemented
                'min_matching_results': 2,  # Minimum results to attempt hybrid search
            },
            'caching': {
                'enabled': True,
                'default_ttl': 3600,  # 1 hour default TTL
                'max_size': 1000,     # Maximum cache entries
            }
        }
        
        # Apply defaults for any missing keys
        for section, options in default_config.items():
            if section not in self._config:
                self._config[section] = {}
            
            for key, value in options.items():
                if key not in self._config[section]:
                    self._config[section][key] = value
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        try:
            return self._config.get(section, {}).get(key, default)
        except Exception as e:
            logger.error(f"Error getting config {section}.{key}: {e}")
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value"""
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section][key] = value
        logger.debug(f"Updated config: {section}.{key} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary"""
        return self._config.copy()

# Global instance
_config_instance = None

def get_search_config() -> SearchConfig:
    """Get the global search configuration instance"""
    global _config_instance
    if _config_instance is None:
        # Load from environment or use defaults
        _config_instance = SearchConfig()
    return _config_instance

def reset_search_config() -> None:
    """Reset the search configuration to defaults"""
    global _config_instance
    _config_instance = SearchConfig()
    logger.info("Reset search configuration to defaults") 