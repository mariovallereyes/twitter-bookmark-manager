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
    
    def __init__(self, config=None):
        """Initialize with optional config override."""
        super().__init__()
        if config:
            self.config = {'twitter': config}
    
    def get_authorization_url(self):
        """Generate Twitter OAuth 2.0 authorization URL."""
        try:
            logger.info("Generating Twitter OAuth 2.0 authorization URL")
            
            # Get provider config
            provider_config = self.config.get('twitter', {})
            if not provider_config:
                logger.error("Twitter provider config not found")
                return None
                
            # Generate PKCE code verifier and challenge
            code_verifier = secrets.token_urlsafe(64)
            # Ensure code verifier is not too long
            if len(code_verifier) > 128:
                code_verifier = code_verifier[:128]
                
            # Generate code challenge using S256 method
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).decode().rstrip('=')
            
            # Generate state parameter for CSRF protection
            state = secrets.token_urlsafe(32)
            
            # Twitter authorization URL
            base_url = 'https://twitter.com/i/oauth2/authorize'
            
            # Required parameters for OAuth 2.0 with PKCE
            params = {
                'response_type': 'code',
                'client_id': provider_config['client_id'],
                'redirect_uri': provider_config['callback_url'],
                'scope': 'tweet.read users.read',
                'state': state,
                'code_challenge': code_challenge,
                'code_challenge_method': 'S256'
            }
            
            # Build full authorization URL
            auth_url = f"{base_url}?{urlencode(params)}"
            
            # Log URL generation (without exposing sensitive data)
            client_id_partial = f"{provider_config['client_id'][:5]}...{provider_config['client_id'][-5:]}"
            logger.info(f"Generated authorization URL with client_id: {client_id_partial}")
            logger.info(f"Using callback URL: {provider_config['callback_url']}")
            
            # Return all data needed for later validation
            return {
                'url': auth_url,
                'state': state,
                'code_verifier': code_verifier,
                'code_challenge': code_challenge
            }
        except Exception as e:
            logger.error(f"Error generating authorization URL: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_user_info(self, callback_data):
        """Get user info from Twitter OAuth 2.0 callback data"""
        try:
            # Verify state parameter to prevent CSRF
            state = callback_data.get('state')
            expected_state = session.pop('twitter_oauth_state', None)
            
            # Log state values for debugging
            logger.info(f"State comparison: received={state}, expected={expected_state}")
            
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
                
            logger.info(f"Code verifier length: {len(code_verifier)}")
            
            # Exchange code for access token
            logger.info("Exchanging authorization code for access token")
            
            # Format token request properly for Twitter
            token_payload = {
                'code': code,
                'grant_type': 'authorization_code',
                'client_id': self.config['twitter']['client_id'],
                'redirect_uri': self.config['twitter']['callback_url'],
                'code_verifier': code_verifier
            }
            
            # Twitter expects client authentication via Basic Auth
            auth_string = f"{self.config['twitter']['client_id']}:{self.config['twitter']['client_secret']}"
            auth_bytes = auth_string.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            
            headers = {
                'Authorization': f'Basic {auth_b64}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Log token request details (without sensitive info)
            logger.info(f"Token request URL: {TWITTER_TOKEN_URL}")
            logger.info(f"Token request payload keys: {list(token_payload.keys())}")
            
            token_response = requests.post(
                TWITTER_TOKEN_URL,
                data=token_payload,
                headers=headers
            )
            
            # Log the response status
            logger.info(f"Token response status: {token_response.status_code}")
            if token_response.status_code != 200:
                logger.error(f"Token request failed: {token_response.status_code} - {token_response.text}")
                return None
                
            tokens = token_response.json()
            access_token = tokens.get('access_token')
            
            # Get user info
            logger.info("Fetching user info from Twitter")
            user_response = requests.get(
                f"{TWITTER_USER_INFO_URL}?user.fields=profile_image_url,name,username",
                headers={
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
            )
            
            # Log user info response status
            logger.info(f"User info response status: {user_response.status_code}")
            if user_response.status_code != 200:
                logger.error(f"User info request failed: {user_response.status_code} - {user_response.text}")
                return None
                
            user_data = user_response.json()
            
            # In OAuth 2.0, Twitter returns user data in a different format
            user = user_data.get('data', {})
            
            # Log user data received
            safe_user = {k: v for k, v in user.items() if k != 'id'}
            logger.info(f"Received user data keys: {list(user.keys())}")
            logger.info(f"User info (without ID): {safe_user}")
            
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

    def get_token(self, code, code_verifier):
        """Exchange authorization code for token."""
        try:
            # Log the start of token exchange
            logger.info("Starting token exchange with Twitter OAuth 2.0")
            
            # Get provider config
            provider_config = self.config.get('twitter', {})
            if not provider_config:
                logger.error("No Twitter provider configuration found")
                return None
            
            # Prepare token request data
            token_url = 'https://api.twitter.com/2/oauth2/token'
            
            # Log parameters (without exposing sensitive data)
            logger.info(f"Token request to: {token_url}")
            logger.info(f"Code verifier length: {len(code_verifier)}")
            logger.info(f"Code length: {len(code)}")
            
            # Create form data for token request
            data = {
                'client_id': provider_config['client_id'],
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': provider_config['callback_url'],
                'code_verifier': code_verifier
            }
            
            # Create Basic Auth header for client_id and client_secret
            auth_string = f"{provider_config['client_id']}:{provider_config['client_secret']}"
            auth_bytes = auth_string.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            
            headers = {
                'Authorization': f'Basic {auth_b64}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Log headers (without exposing sensitive data)
            logger.info(f"Authorization header present: {bool(headers.get('Authorization'))}")
            logger.info(f"Content-Type: {headers.get('Content-Type')}")
            
            # Make the token request
            logger.info("Sending token request to Twitter...")
            response = requests.post(token_url, data=data, headers=headers)
            
            # Log response code
            logger.info(f"Token response status: {response.status_code}")
            
            # Check if the request was successful
            if response.status_code != 200:
                logger.error(f"Token request failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
            
            # Parse response
            token_data = response.json()
            logger.info("Successfully obtained token from Twitter")
            
            return token_data
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            logger.error(traceback.format_exc())
            return None

    def get_user_info(self, access_token):
        """Get user information using access token."""
        try:
            # Log the start of user info retrieval
            logger.info("Getting user info from Twitter")
            
            # Twitter API endpoint for user info
            user_info_url = 'https://api.twitter.com/2/users/me'
            
            # Parameters to request additional user fields
            params = {
                'user.fields': 'profile_image_url,name,username'
            }
            
            # Headers with access token
            headers = {
                'Authorization': f'Bearer {access_token}'
            }
            
            # Make the request
            logger.info(f"Sending user info request to Twitter: {user_info_url}")
            response = requests.get(user_info_url, params=params, headers=headers)
            
            # Log response code
            logger.info(f"User info response status: {response.status_code}")
            
            # Check if the request was successful
            if response.status_code != 200:
                logger.error(f"User info request failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
            
            # Parse response
            user_data = response.json()
            
            # Extract user data from Twitter's response structure
            if 'data' in user_data:
                user_info = user_data['data']
                # Log user info received (with ID partially masked for privacy)
                user_id = user_info.get('id', '')
                logger.info(f"Received user info for Twitter ID: {user_id[:3]}...{user_id[-3:] if len(user_id) > 6 else ''}")
                return user_info
            else:
                logger.error(f"Missing data in user response: {user_data}")
                return None
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
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