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
from urllib.parse import urlencode, parse_qs
import requests
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

class OAuthProvider:
    """Base class for OAuth providers"""
    
    def __init__(self, config):
        self.config = config
        
    def get_authorize_url(self):
        """Get the authorization URL"""
        raise NotImplementedError()
        
    def get_user_info(self, callback_data):
        """Get user info from the OAuth provider"""
        raise NotImplementedError()

class TwitterOAuth(OAuthProvider):
    """Twitter OAuth 2.0 implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        # Get Twitter OAuth 2.0 credentials from config
        twitter_config = config.get('twitter', {})
        
        self.client_id = twitter_config.get('client_id')
        self.client_secret = twitter_config.get('client_secret')
        self.callback_url = twitter_config.get('callback_url')
        self.scopes = ['tweet.read', 'users.read', 'offline.access']
        
        # Log credentials for debugging
        logger.info(f"TwitterOAuth initialized:")
        logger.info(f"  - client_id is {'present' if self.client_id else 'MISSING'}")
        logger.info(f"  - client_secret is {'present' if self.client_secret else 'MISSING'}")
        logger.info(f"  - callback_url is {'present' if self.callback_url else 'MISSING'}")
        
    def get_authorize_url(self):
        """Get the Twitter authorization URL using OAuth 2.0"""
        # Log the exact callback URL being used (for debugging)
        logger.info(f"TwitterOAuth: actual callback_url value: {self.callback_url}")
        
        try:
            # Generate a code verifier and challenge for PKCE
            code_verifier = secrets.token_urlsafe(64)
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).decode().rstrip('=')
            
            # Store code verifier in session for later
            session['twitter_code_verifier'] = code_verifier
            
            # Generate a state parameter for CSRF protection
            state = secrets.token_urlsafe(32)
            session['twitter_oauth_state'] = state
            
            # Build authorization URL
            params = {
                'response_type': 'code',
                'client_id': self.client_id,
                'redirect_uri': self.callback_url,
                'scope': ' '.join(self.scopes),
                'state': state,
                'code_challenge': code_challenge,
                'code_challenge_method': 'S256'
            }
            authorization_url = f"{TWITTER_AUTHORIZATION_URL}?{urlencode(params)}"
            
            logger.info(f"Generated Twitter OAuth 2.0 authorization URL")
            return authorization_url
            
        except Exception as e:
            logger.error(f"Error getting Twitter authorization URL: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_user_info(self, callback_data):
        """Get user info from Twitter OAuth 2.0 callback data"""
        try:
            # Verify state parameter to prevent CSRF
            state = callback_data.get('state')
            expected_state = session.pop('twitter_oauth_state', None)
            if not state or state != expected_state:
                logger.error(f"State mismatch: received {state}, expected {expected_state}")
                return None
            
            # Extract authorization code
            code = callback_data.get('code')
            if not code:
                logger.error("No authorization code received from Twitter")
                return None
            
            # Get code verifier from session
            code_verifier = session.pop('twitter_code_verifier', None)
            if not code_verifier:
                logger.error("No code verifier in session")
                return None
            
            # Exchange code for access token
            logger.info("Exchanging authorization code for access token")
            token_data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'code_verifier': code_verifier,
                'grant_type': 'authorization_code',
                'redirect_uri': self.callback_url
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            token_response = requests.post(
                TWITTER_TOKEN_URL,
                data=token_data,
                headers=headers
            )
            
            if token_response.status_code != 200:
                logger.error(f"Token request failed: {token_response.status_code} - {token_response.text}")
                return None
                
            tokens = token_response.json()
            access_token = tokens.get('access_token')
            
            # Get user info
            logger.info("Fetching user info from Twitter")
            user_response = requests.get(
                TWITTER_USER_INFO_URL,
                headers={
                    'Authorization': f'Bearer {access_token}',
                }
            )
            
            if user_response.status_code != 200:
                logger.error(f"User info request failed: {user_response.status_code} - {user_response.text}")
                return None
                
            user_data = user_response.json()
            
            # In OAuth 2.0, Twitter returns user data in a different format
            user = user_data.get('data', {})
            
            return {
                'id': user.get('id'),
                'username': user.get('username'),
                'name': user.get('name'),
                'profile_image_url': user.get('profile_image_url', ''),
                'access_token': access_token,
                'refresh_token': tokens.get('refresh_token', '')
            }
            
        except Exception as e:
            logger.error(f"Error getting Twitter user info: {e}")
            logger.error(traceback.format_exc())
            return None

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