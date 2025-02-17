import pytest
from core.auth import TwitterAuthHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_twitter_auth_handler_initialization():
    """Test TwitterAuthHandler initializes with correct credentials"""
    auth_handler = TwitterAuthHandler()
    
    assert auth_handler.client_id == os.getenv('TWITTER_CLIENT_ID')
    assert auth_handler.client_secret == os.getenv('TWITTER_CLIENT_SECRET')
    assert auth_handler.redirect_uri == os.getenv('TWITTER_REDIRECT_URI')
    assert auth_handler.oauth2_user_handler is None

def test_initialize_oauth2():
    """Test OAuth2 handler initialization"""
    auth_handler = TwitterAuthHandler()
    oauth2_handler = auth_handler.initialize_oauth2()
    
    assert oauth2_handler is not None
    assert auth_handler.oauth2_user_handler is not None

def test_get_authorization_url():
    """Test authorization URL generation"""
    auth_handler = TwitterAuthHandler()
    auth_url = auth_handler.get_authorization_url()  # Removed tuple unpacking
    
    assert isinstance(auth_url, str)
    assert "https://" in auth_url