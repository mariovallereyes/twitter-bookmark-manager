"""
Authentication routes for the PythonAnywhere implementation.
Handles login, logout, and OAuth callback functionality.
"""

import os
import logging
import traceback
from flask import Blueprint, request, redirect, url_for, session, render_template, flash, current_app, jsonify
from urllib.parse import urlparse, urljoin
import json

# Import custom modules
from auth.oauth_final import OAuthManager, TwitterOAuth1
from auth.user_context_final import UserContext, get_user_context, user_required, store_user_in_session
from database.multi_user_db.user_model_final import (
    get_user_by_provider_id, 
    create_user, 
    update_last_login
)

# Set up logging
logger = logging.getLogger('auth_routes_final')

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

def is_safe_url(target):
    """Check if the URL is safe to redirect to"""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

def get_db_connection():
    """Get a database connection from the app context"""
    return current_app.config['get_db_connection']()

def get_oauth_manager():
    """Get the OAuth manager with configuration from the app"""
    # Use the registered callback URL from environment variables
    twitter_callback_url = os.environ.get('TWITTER_REDIRECT_URI', 
                          'https://twitter-bookmark-manager-production.up.railway.app/oauth/callback/twitter')
    
    # Get configuration from the application
    twitter_client_id = os.environ.get('TWITTER_CLIENT_ID', '')
    twitter_client_secret = os.environ.get('TWITTER_CLIENT_SECRET', '')
    
    # Log credential presence (without exposing actual values)
    logger.info(f"Twitter OAuth configuration:")
    logger.info(f"  - Client ID present: {bool(twitter_client_id)}")
    logger.info(f"  - Client Secret present: {bool(twitter_client_secret)}")
    logger.info(f"  - Callback URL: {twitter_callback_url}")
    
    config = {
        'twitter': {
            'client_id': twitter_client_id,
            'client_secret': twitter_client_secret,
            'callback_url': twitter_callback_url
        }
    }
    
    return OAuthManager(config)

def ensure_string_session():
    """Ensure all session values are strings, not bytes"""
    for key in list(session.keys()):
        if isinstance(session[key], bytes):
            try:
                session[key] = session[key].decode('utf-8')
                logger.info(f"Converted session[{key}] from bytes to string")
            except Exception as e:
                logger.warning(f"Could not convert session[{key}] to string: {e}")
                # If conversion fails, remove the problematic key
                session.pop(key, None)

@auth_bp.route('/login')
def login():
    """Render login page."""
    logging.info("Login page requested")
    return render_template('login_final.html')

@auth_bp.route('/login/twitter')
def login_twitter():
    """
    Initiates Twitter OAuth 1.0a authentication process.
    
    Generates a request token and redirects the user to Twitter's
    authorization page to grant access to the application.
    """
    logger.info("-------- Starting Twitter OAuth 1.0a authentication process --------")
    
    try:
        # Get Twitter API credentials from environment variables
        api_key = os.environ.get('TWITTER_API_KEY')
        api_secret = os.environ.get('TWITTER_API_SECRET')
        callback_url = request.url_root.rstrip('/') + url_for('twitter_oauth_callback')
        
        # Log API keys and callback URL lengths without exposing values
        logger.info(f"Twitter API key length: {len(api_key) if api_key else 'None'}")
        logger.info(f"Twitter API secret length: {len(api_secret) if api_secret else 'None'}")
        logger.info(f"Twitter callback URL: {callback_url}")
        
        # Initialize Twitter OAuth 1.0a client
        twitter_oauth = TwitterOAuth1(api_key, api_secret, callback_url)
        
        # Get request token from Twitter
        request_token, request_token_secret, callback_confirmed = twitter_oauth.get_request_token()
        
        if not request_token or not request_token_secret:
            logger.error("Failed to get Twitter request token")
            flash("Authentication failed: Could not get authorization from Twitter.", "error")
            return redirect(url_for('auth.login'))
        
        # Store request token and secret in session for later use
        session['twitter_request_token'] = request_token
        session['twitter_request_token_secret'] = request_token_secret
        logger.info(f"Stored Twitter request token in session: {request_token[:10]}...")
        
        # Generate authorization URL and redirect user
        auth_url = twitter_oauth.get_authorization_url(request_token)
        if not auth_url:
            logger.error("Failed to generate Twitter authorization URL")
            flash("Authentication failed: Unable to redirect to Twitter.", "error")
            return redirect(url_for('auth.login'))
        
        logger.info(f"Redirecting to Twitter authorization URL: {auth_url[:50]}...")
        return redirect(auth_url)
    
    except Exception as e:
        logger.error(f"Error in Twitter authentication: {e}")
        logger.error(traceback.format_exc())
        flash("An error occurred during Twitter authentication. Please try again.", "error")
        return redirect(url_for('auth.login'))

# OAuth login initiator
@auth_bp.route('/login/<provider>')
def oauth_login(provider):
    """Initiate OAuth login flow"""
    if provider not in ['twitter', 'google']:
        flash(f"Unsupported authentication provider: {provider}", "error")
        return redirect(url_for('auth.login'))
        
    # Get authorization URL
    oauth_manager = get_oauth_manager()
    auth_url = oauth_manager.get_authorize_url(provider)
    
    if not auth_url:
        flash(f"Failed to get authorization URL for {provider}", "error")
        return redirect(url_for('auth.login'))
        
    # Redirect to authorization URL
    return redirect(auth_url)

# OAuth callback
@auth_bp.route('/oauth/callback/<provider>')
def oauth_callback(provider):
    """Handle OAuth callback"""
    # Ensure session values are strings
    ensure_string_session()
    
    logger.info(f"OAuth callback received for provider: {provider}")
    logger.info(f"Full request URL: {request.url}")
    logger.info(f"Request args: {request.args}")
    logger.info(f"Request form data: {request.form}")
    logger.info(f"Session keys before processing: {list(session.keys())}")
    
    # Check for OAuth error responses
    if 'error' in request.args:
        error = request.args.get('error')
        error_description = request.args.get('error_description', 'No description provided')
        logger.error(f"OAuth error: {error} - {error_description}")
        flash(f"Authentication error: {error_description}", "error")
        return redirect(url_for('auth.login'))
    
    # Extract OAuth data based on provider
    oauth_data = {}
    if provider == 'twitter':
        # Twitter OAuth 2.0 data
        code = request.args.get('code')
        state = request.args.get('state')
        
        logger.info(f"Twitter callback received - code present: {bool(code)}, state present: {bool(state)}")
        
        if not code:
            flash('Authentication failed: No authorization code received. Please try again.', 'error')
            logger.error(f"Missing authorization code from Twitter callback")
            return redirect(url_for('auth.login'))
        
        # Check if the required session data is available
        code_verifier = session.get('twitter_code_verifier')
        expected_state = session.get('twitter_oauth_state')
        logger.info(f"Session data - code_verifier present: {bool(code_verifier)}, expected_state present: {bool(expected_state)}")
        
        if not code_verifier:
            logger.error("Missing code_verifier in session - session may have been lost")
            flash('Authentication failed: Session data lost. Please try again.', 'error')
            return redirect(url_for('auth.login'))
        
        # Store OAuth data
        oauth_data = {
            'code': code,
            'state': state
        }
        
        # Get user info - don't pop the session values yet, in case the get_user_info method needs them
        try:
            oauth_manager = get_oauth_manager()
            logger.info("Getting user info from Twitter")
            user_info = oauth_manager.get_user_info('twitter', oauth_data)
            
            # Now we can clear the session values
            session.pop('twitter_code_verifier', None)
            session.pop('twitter_oauth_state', None)
            
            if not user_info:
                flash('Failed to get user information from Twitter. Please try again.', 'error')
                logger.error("get_user_info returned None")
                return redirect(url_for('auth.login'))
                
            if 'id' not in user_info:
                flash('Failed to get user ID from Twitter. Please try again.', 'error')
                logger.error(f"Missing ID in user_info: {user_info}")
                return redirect(url_for('auth.login'))
                
            # Process user information
            provider_id = str(user_info.get('id'))
            username = user_info.get('username', '')
            name = user_info.get('name', '')
            avatar_url = user_info.get('profile_image_url', '')
            
            logger.info(f"Received Twitter user info: id={provider_id}, username={username}")
            
            # Find or create user
            db = get_db_connection()
            user = get_user_by_provider_id(db, 'twitter', provider_id)
            
            if not user:
                # Create new user
                logger.info(f"Creating new user: username={username}, provider_id={provider_id}")
                user = create_user(
                    db, 
                    username=username,
                    email=f"{username}@twitter.placeholder",
                    auth_provider='twitter',
                    provider_id=provider_id,
                    display_name=name,
                    profile_image_url=avatar_url
                )
                if not user:
                    logger.error("Failed to create user")
                    flash('Failed to create user account. Please try again.', 'error')
                    return redirect(url_for('auth.login'))
                    
                logger.info(f"Created new user: {username} (ID: {user.id})")
            else:
                # Update last login
                update_last_login(db, user.id)
                logger.info(f"Logged in existing user: {username} (ID: {user.id})")
            
            # Store user ID in session as string
            session['user_id'] = str(user.id)
            logger.info(f"Set user_id in session: {session.get('user_id')}")
            
            # Force session to be saved
            session.modified = True
            
            # Redirect to next_url or home
            next_url = session.pop('next', '/')
            if not is_safe_url(next_url):
                next_url = '/'
                
            logger.info(f"Redirecting to: {next_url}")
            return redirect(next_url)
            
        except Exception as e:
            flash('Authentication failed. Please try again.', 'error')
            logger.error(f"OAuth error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return redirect(url_for('auth.login'))
    else:
        # Unsupported provider
        flash(f'Login with {provider} is not supported.', 'error')
        return redirect(url_for('auth.login'))

# Add a top-level callback route to match the registered redirect URI path without the auth blueprint prefix
@auth_bp.route('/callback/<provider>')
def callback_alt(provider):
    """Alternative OAuth callback route to match the registered redirect URI"""
    logger.info(f"Alternative OAuth callback received for provider: {provider}")
    return oauth_callback(provider)

# Logout
@auth_bp.route('/logout')
def logout():
    """Log out the current user by clearing the session"""
    # Ensure session values are strings first
    ensure_string_session()
    
    try:
        # Get the current user to log the logout action
        user = UserContext.get_current_user()
        if user:
            logger.info(f"Logging out user: {getattr(user, 'username', 'unknown')} (ID: {user.id})")
        
        # Clear all session data
        session.clear()
        
        # Flash logout message
        flash("You have been logged out successfully.", "success")
    except Exception as e:
        # Log the error but continue with logout
        logger.error(f"Error during logout: {str(e)}")
        
    # Redirect to login page
    return redirect(url_for('auth.login'))

# Profile page
@auth_bp.route('/profile')
def profile():
    """User profile page"""
    # Require login
    if not UserContext.is_authenticated():
        return redirect(url_for('auth.login', next=request.url))
        
    # Get current user
    user = UserContext.get_current_user()
    
    # Get admin status
    is_admin = getattr(user, 'is_admin', False)
    
    # Render profile template
    return render_template('profile_final.html', user=user, is_admin=is_admin)

@auth_bp.route('/oauth/callback/twitter')
def oauth_callback_twitter():
    """
    Handles the OAuth 1.0a callback from Twitter.
    
    Processes the callback after user authorizes the application on Twitter,
    exchanges the request token for an access token, and logs the user in.
    """
    logger.info("-------- Twitter OAuth 1.0a callback received --------")
    logger.info(f"Full request URL: {request.url}")
    logger.info(f"Request args: {dict(request.args)}")
    logger.info(f"Session keys before processing: {list(session.keys())}")
    
    try:
        # Check for errors from Twitter
        if request.args.get('denied'):
            denied_token = request.args.get('denied')
            logger.warning(f"User denied access to Twitter account. Token: {denied_token}")
            flash("Twitter authentication was canceled.", "warning")
            return redirect(url_for('auth.login'))
        
        # Get verifier from callback
        oauth_verifier = request.args.get('oauth_verifier')
        oauth_token = request.args.get('oauth_token')
        
        if not oauth_verifier or not oauth_token:
            logger.error("Missing oauth_verifier or oauth_token in callback")
            flash("Authentication failed: Invalid response from Twitter.", "error")
            return redirect(url_for('auth.login'))
        
        # Verify that the token matches what we have in session
        request_token = session.get('twitter_request_token')
        request_token_secret = session.get('twitter_request_token_secret')
        
        if not request_token or not request_token_secret:
            logger.error("Missing request token in session")
            flash("Authentication failed: Session expired or was lost.", "error")
            return redirect(url_for('auth.login'))
        
        if request_token != oauth_token:
            logger.error(f"Token mismatch. Session: {request_token[:10]}... Callback: {oauth_token[:10]}...")
            flash("Authentication failed: Invalid token.", "error")
            return redirect(url_for('auth.login'))
        
        # Get Twitter API credentials
        api_key = os.environ.get('TWITTER_API_KEY')
        api_secret = os.environ.get('TWITTER_API_SECRET')
        callback_url = request.url_root.rstrip('/') + url_for('twitter_oauth_callback')
        
        # Initialize Twitter OAuth 1.0a client
        twitter_oauth = TwitterOAuth1(api_key, api_secret, callback_url)
        
        # Exchange request token for access token
        user_data = twitter_oauth.get_access_token(
            request_token, 
            request_token_secret,
            oauth_verifier
        )
        
        if not user_data:
            logger.error("Failed to get access token from Twitter")
            flash("Authentication failed: Could not complete Twitter authentication.", "error")
            return redirect(url_for('auth.login'))
        
        # Get user information
        access_token = user_data.get('access_token')
        access_token_secret = user_data.get('access_token_secret')
        
        # Get additional user info if needed
        # user_info = twitter_oauth.get_user_info(access_token, access_token_secret)
        
        # Create or update user in the database
        from auth.user_api_final import find_or_create_user
        
        user = find_or_create_user({
            'provider': 'twitter',
            'provider_id': user_data.get('id'),
            'username': user_data.get('screen_name'),
            'display_name': user_data.get('screen_name'),
            'provider_data': json.dumps(user_data)
        })
        
        if not user:
            logger.error("Failed to create or find user")
            flash("Authentication failed: Could not create user account.", "error")
            return redirect(url_for('auth.login'))
        
        # Store user in session
        store_user_in_session(user)
        logger.info(f"User logged in: {user.get('username')} (ID: {user.get('id')})")
        
        # Clear OAuth tokens from session
        session.pop('twitter_request_token', None)
        session.pop('twitter_request_token_secret', None)
        
        # Redirect to home page or return URL
        return_url = session.pop('return_url', url_for('index'))
        return redirect(return_url)
    
    except Exception as e:
        logger.error(f"Error in Twitter OAuth callback: {e}")
        logger.error(traceback.format_exc())
        flash("An error occurred during authentication. Please try again.", "error")
        return redirect(url_for('auth.login')) 