import os
from dotenv import load_dotenv
import tweepy
from typing import Tuple, Optional

# Load environment variables
load_dotenv()

class TwitterAuthHandler:
    def __init__(self):
        self.client_id = os.getenv('TWITTER_CLIENT_ID')
        self.client_secret = os.getenv('TWITTER_CLIENT_SECRET')
        self.redirect_uri = os.getenv('TWITTER_REDIRECT_URI')
        self.oauth2_user_handler = None
        
    def initialize_oauth2(self) -> tweepy.OAuth2UserHandler:
        """Initialize OAuth2 handler for Twitter API v2"""
        self.oauth2_user_handler = tweepy.OAuth2UserHandler(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=['bookmark.read', 'bookmark.write', 'tweet.read']
        )
        return self.oauth2_user_handler
    
    def get_authorization_url(self) -> Tuple[str, str]:
        """Get the authorization URL and state"""
        if not self.oauth2_user_handler:
            self.initialize_oauth2()
        return self.oauth2_user_handler.get_authorization_url()
    
    def get_access_token(self, code: str) -> Optional[str]:
        """Exchange authorization code for access token"""
        try:
            token = self.oauth2_user_handler.fetch_token(code)
            return token
        except Exception as e:
            print(f"Error getting access token: {e}")
            return None
    
    def get_client(self, access_token: str) -> tweepy.Client:
        """Get authenticated Tweepy client"""
        return tweepy.Client(
            bearer_token=access_token,
            consumer_key=self.client_id,
            consumer_secret=self.client_secret
        )