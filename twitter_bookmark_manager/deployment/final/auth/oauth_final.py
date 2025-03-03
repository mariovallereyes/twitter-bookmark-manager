"""
OAuth implementation for final environment authentication.
Supports Twitter and Google authentication.
"""

import os
import json
import secrets
import time
import logging
from urllib.parse import urlencode
import requests
from flask import request, redirect, url_for, session
from requests_oauthlib import OAuth2Session, OAuth1Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('oauth_final')

# OAuth configurations
TWITTER_REQUEST_TOKEN_URL = 'https://api.twitter.com/oauth/request_token'
TWITTER_AUTHORIZATION_URL = 'https://api.twitter.com/oauth/authorize'
TWITTER_ACCESS_TOKEN_URL = 'https://api.twitter.com/oauth/access_token'
TWITTER_USER_INFO_URL = 'https://api.twitter.com/1.1/account/verify_credentials.json'

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
    """Twitter OAuth implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.consumer_key = config.get('TWITTER_CONSUMER_KEY')
        self.consumer_secret = config.get('TWITTER_CONSUMER_SECRET')
        self.callback_url = config.get('TWITTER_CALLBACK_URL')
        
    def get_authorize_url(self):
        """Get the Twitter authorization URL"""
        # Create OAuth1 session
        oauth = OAuth1Session(
            client_key=self.consumer_key,
            client_secret=self.consumer_secret,
            callback_uri=self.callback_url
        )
        
        # Get request token
        try:
            oauth.fetch_request_token(TWITTER_REQUEST_TOKEN_URL)
            authorization_url = oauth.authorization_url(TWITTER_AUTHORIZATION_URL)
            
            # Save oauth session state
            session['oauth_state'] = oauth.__dict__.get('_client').__dict__
            
            return authorization_url
        except Exception as e:
            logger.error(f"Error getting Twitter authorization URL: {e}")
            return None
    
    def get_user_info(self, callback_data):
        """Get user info from Twitter callback data"""
        # Reconstruct OAuth session
        oauth_state = session.pop('oauth_state', None)
        if not oauth_state:
            logger.error("Missing OAuth state in session")
            return None
            
        oauth = OAuth1Session(
            client_key=self.consumer_key,
            client_secret=self.consumer_secret
        )
        # Restore state
        oauth.__dict__.get('_client').__dict__.update(oauth_state)
        
        # Get access token
        oauth_response = oauth.parse_authorization_response(callback_data)
        oauth_token = oauth_response.get('oauth_token')
        oauth_verifier = oauth_response.get('oauth_verifier')
        
        # Get access token
        try:
            oauth = OAuth1Session(
                client_key=self.consumer_key,
                client_secret=self.consumer_secret,
                resource_owner_key=oauth_token,
                verifier=oauth_verifier
            )
            tokens = oauth.fetch_access_token(TWITTER_ACCESS_TOKEN_URL)
            
            # Use access token to get user info
            oauth = OAuth1Session(
                client_key=self.consumer_key,
                client_secret=self.consumer_secret,
                resource_owner_key=tokens['oauth_token'],
                resource_owner_secret=tokens['oauth_token_secret']
            )
            
            # Get user info
            response = oauth.get(
                f"{TWITTER_USER_INFO_URL}?include_email=true"
            )
            user_info = response.json()
            
            return {
                'provider': 'twitter',
                'provider_user_id': str(user_info['id']),
                'username': user_info['screen_name'],
                'email': user_info.get('email'),
                'display_name': user_info['name'],
                'profile_image_url': user_info['profile_image_url_https'].replace('_normal', '')
            }
            
        except Exception as e:
            logger.error(f"Error getting Twitter user info: {e}")
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
        provider = self.providers.get(provider_name)
        if not provider:
            logger.error(f"Unknown OAuth provider: {provider_name}")
            return None
            
        return provider.get_authorize_url()
        
    def get_user_info(self, provider_name, callback_data):
        """Get user info from the specified provider"""
        provider = self.providers.get(provider_name)
        if not provider:
            logger.error(f"Unknown OAuth provider: {provider_name}")
            return None
            
        return provider.get_user_info(callback_data) 