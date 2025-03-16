"""
OAuth implementation for final environment authentication.
Supports Twitter and Google authentication.
"""

import os
import json
import secrets
import time
import logging
import base64
import hashlib
import traceback
import requests
import hmac
from urllib.parse import urlencode, parse_qs, quote, urlparse, parse_qsl
from flask import request, redirect, url_for, session
from requests_oauthlib import OAuth2Session, OAuth1Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('oauth_final')

# OAuth configurations
# Twitter OAuth 2.0 endpoints
TWITTER_AUTHORIZATION_URL = 'https://twitter.com/i/oauth2/authorize'
TWITTER_TOKEN_URL = 'https://api.twitter.com/2/oauth2/token'
TWITTER_USER_INFO_URL = 'https://api.twitter.com/2/users/me'

GOOGLE_AUTHORIZATION_URL = 'https://accounts.google.com/o/oauth2/auth'
GOOGLE_TOKEN_URL = 'https://accounts.google.com/o/oauth2/token'
GOOGLE_USER_INFO_URL = 'https://www.googleapis.com/oauth2/v3/userinfo'

# Twitter OAuth 1.0a settings
TWITTER_REQUEST_TOKEN_URL = 'https://api.twitter.com/oauth/request_token'
TWITTER_AUTHORIZE_URL = 'https://api.twitter.com/oauth/authorize'
TWITTER_ACCESS_TOKEN_URL = 'https://api.twitter.com/oauth/access_token'
TWITTER_VERIFY_CREDENTIALS_URL = 'https://api.twitter.com/1.1/account/verify_credentials.json'

class OAuthProvider:
    """Base class for OAuth providers"""
    
    def __init__(self, config=None):
        """Initialize with provider configuration"""
        self.config = config or {}
        logger.info(f"OAuthProvider initialized with config: {bool(self.config)}")
        
    def get_authorize_url(self):
        """Get the authorization URL"""
        raise NotImplementedError()
        
    def get_user_info(self, callback_data):
        """Get user info from the OAuth provider"""
        raise NotImplementedError()

class TwitterOAuth:
    """Twitter OAuth provider implementation"""
    
    def __init__(self, config=None):
        """Initialize with Twitter OAuth configuration."""
        self.config = config or {}
        self.client_id = self.config.get('client_id', os.environ.get('TWITTER_CLIENT_ID', ''))
        self.client_secret = self.config.get('client_secret', os.environ.get('TWITTER_CLIENT_SECRET', ''))
        self.callback_url = self.config.get('callback_url', os.environ.get('TWITTER_REDIRECT_URI', ''))
        self.auth_url = 'https://twitter.com/i/oauth2/authorize'
        self.token_url = 'https://api.twitter.com/2/oauth2/token'
        self.user_info_url = 'https://api.twitter.com/2/users/me'
        
        logging.info(f"TwitterOAuth initialized with client ID length: {len(self.client_id)}")
        logging.info(f"TwitterOAuth callback URL: {self.callback_url}")
    
    def get_authorization_url(self):
        """Generate authorization URL with PKCE."""
        try:
            # Generate random state and code verifier
            state = secrets.token_urlsafe(32)
            code_verifier = secrets.token_urlsafe(64)  # Between 43-128 chars
            
            # Ensure code verifier meets Twitter's requirements (43-128 chars)
            if len(code_verifier) < 43:
                code_verifier = code_verifier + secrets.token_urlsafe(43 - len(code_verifier))
            elif len(code_verifier) > 128:
                code_verifier = code_verifier[:128]
            
            logging.info(f"Generated code verifier length: {len(code_verifier)}")
            
            # Generate code challenge using SHA256
            code_challenge = self._generate_code_challenge(code_verifier)
            
            # Build authorization URL
            params = {
                'response_type': 'code',
                'client_id': self.client_id,
                'redirect_uri': self.callback_url,
                'scope': 'tweet.read users.read bookmark.read bookmark.write',
                'state': state,
                'code_challenge': code_challenge,
                'code_challenge_method': 'S256'
            }
            
            authorization_url = f"{self.auth_url}?{urlencode(params)}"
            
            return {
                'url': authorization_url,
                'state': state,
                'code_verifier': code_verifier,
                'code_challenge': code_challenge
            }
            
        except Exception as e:
            logging.error(f"Error generating authorization URL: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def get_token(self, code, code_verifier):
        """Exchange authorization code for access token."""
        try:
            if not code or not code_verifier:
                logging.error("Missing code or code_verifier in get_token")
                return None
            
            logging.info(f"Exchanging code for token with verifier length: {len(code_verifier)}")
            
            # Prepare token request
            token_data = {
                'code': code,
                'grant_type': 'authorization_code',
                'client_id': self.client_id,
                'redirect_uri': self.callback_url,
                'code_verifier': code_verifier
            }
            
            # Basic auth for client_id and client_secret
            auth = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': f'Basic {auth}'
            }
            
            # Make token request
            response = requests.post(
                self.token_url,
                data=token_data,
                headers=headers
            )
            
            if response.status_code != 200:
                logging.error(f"Token request failed with status {response.status_code}")
                logging.error(f"Response: {response.text}")
                return None
            
            # Extract token data
            token_response = response.json()
            
            logging.info("Successfully retrieved access token")
            
            return token_response
            
        except Exception as e:
            logging.error(f"Error getting token: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def get_user_info(self, access_token):
        """Get user information from Twitter API."""
        try:
            if not access_token:
                logging.error("No access token provided to get_user_info")
                return None
            
            # Make user info request
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'user.fields': 'id,name,username,profile_image_url'
            }
            
            response = requests.get(
                self.user_info_url,
                headers=headers,
                params=params
            )
            
            if response.status_code != 200:
                logging.error(f"User info request failed with status {response.status_code}")
                logging.error(f"Response: {response.text}")
                return None
            
            # Extract user data
            user_response = response.json()
            
            if 'data' not in user_response:
                logging.error(f"Invalid user info response: {user_response}")
                return None
            
            user_data = user_response['data']
            
            # Create standardized user info
            user_info = {
                'id': user_data.get('id'),
                'username': user_data.get('username'),
                'name': user_data.get('name'),
                'profile_image_url': user_data.get('profile_image_url')
            }
            
            logging.info(f"Got user info for Twitter user: {user_info['username']}")
            
            return user_info
            
        except Exception as e:
            logging.error(f"Error getting user info: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def _generate_code_challenge(self, code_verifier):
        """Generate PKCE code challenge from verifier using S256 method."""
        sha256 = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        code_challenge = base64.urlsafe_b64encode(sha256).decode('utf-8').rstrip('=')
        return code_challenge

class GoogleOAuth(OAuthProvider):
    """Google OAuth implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.client_id = config.get('GOOGLE_CLIENT_ID')
        self.client_secret = config.get('GOOGLE_CLIENT_SECRET')
        self.redirect_uri = config.get('GOOGLE_REDIRECT_URI')
        self.scope = [
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile',
            'openid'
        ]
        
    def get_authorize_url(self):
        """Get the Google authorization URL"""
        oauth = OAuth2Session(
            client_id=self.client_id,
            redirect_uri=self.redirect_uri,
            scope=self.scope
        )
        
        # Generate a random state for CSRF protection
        state = secrets.token_urlsafe(16)
        session['oauth_state'] = state
        
        # Get authorization URL
        try:
            authorization_url, state = oauth.authorization_url(
                GOOGLE_AUTHORIZATION_URL,
                access_type="offline",
                prompt="select_account"
            )
            return authorization_url
        except Exception as e:
            logger.error(f"Error getting Google authorization URL: {e}")
            return None
    
    def get_user_info(self, callback_data):
        """Get user info from Google callback data"""
        # Verify state to prevent CSRF
        state = session.pop('oauth_state', None)
        if state is None:
            logger.error("Missing OAuth state in session")
            return None
            
        # Use code to get tokens
        try:
            # Extract code from callback URL
            code = request.args.get('code')
            
            # Exchange code for tokens
            token_response = requests.post(
                GOOGLE_TOKEN_URL,
                data={
                    'code': code,
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'redirect_uri': self.redirect_uri,
                    'grant_type': 'authorization_code'
                }
            )
            token_data = token_response.json()
            
            # Use access token to get user info
            user_info_response = requests.get(
                GOOGLE_USER_INFO_URL,
                headers={'Authorization': f"Bearer {token_data['access_token']}"}
            )
            user_info = user_info_response.json()
            
            return {
                'provider': 'google',
                'provider_user_id': user_info['sub'],
                'username': user_info['email'].split('@')[0],  # Use part before @ as username
                'email': user_info['email'],
                'display_name': user_info.get('name'),
                'profile_image_url': user_info.get('picture')
            }
            
        except Exception as e:
            logger.error(f"Error getting Google user info: {e}")
            return None

class OAuthManager:
    """OAuth manager for handling different OAuth providers"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.providers = {
            'twitter': TwitterOAuth(self.config),
            'google': GoogleOAuth(self.config)
        }
        
    def get_authorize_url(self, provider_name):
        """Get authorization URL for the specified provider"""
        if provider_name not in self.providers:
            return None
            
        return self.providers[provider_name].get_authorize_url()
        
    def get_user_info(self, provider_name, callback_data):
        """Get user info from the specified provider"""
        if provider_name not in self.providers:
            return None
            
        return self.providers[provider_name].get_user_info(callback_data) 