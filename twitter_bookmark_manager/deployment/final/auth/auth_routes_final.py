"""
Authentication routes for the PythonAnywhere implementation.
Handles login, logout, and OAuth callback functionality.
"""

import os
import logging
import traceback
from flask import Blueprint, request, redirect, url_for, session, render_template, flash, current_app, jsonify
from urllib.parse import urlparse, urljoin

# Import custom modules
from auth.oauth_final import OAuthManager
from auth.user_context_final import UserContext
from database.multi_user_db.user_model_final import (
    get_user_by_provider_id, 
    create_user, 
    update_last_login
)

# Set up logging
logger = logging.getLogger('auth_routes_final')

# Create blueprint
auth_bp = Blueprint('auth', __name__)

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

@auth_bp.route('/twitter')
def twitter_auth():
    """Initiate Twitter OAuth flow."""
    logging.info("======= STARTING TWITTER OAUTH FLOW =======")
    try:
        # Import Twitter OAuth provider
        from auth.oauth_final import TwitterOAuth
        
        # Initialize provider with correct config
        provider_config = {
            'client_id': os.environ.get('TWITTER_CLIENT_ID', ''),
            'client_secret': os.environ.get('TWITTER_CLIENT_SECRET', ''),
            'callback_url': os.environ.get('TWITTER_REDIRECT_URI', '')
        }
        
        # Log config (without sensitive info)
        logging.info(f"Twitter client_id length: {len(provider_config['client_id'])}")
        logging.info(f"Twitter client_secret length: {len(provider_config['client_secret'])}")
        logging.info(f"Twitter callback_url: {provider_config['callback_url']}")
        
        # Remove any old OAuth data from session
        for key in ['oauth_state', 'code_verifier', 'code_challenge', 
                   'twitter_oauth_state', 'twitter_code_verifier', 'twitter_code_challenge']:
            if key in session:
                session.pop(key)
        
        # Create OAuth manager
        twitter_oauth = TwitterOAuth(provider_config)
        
        # Generate PKCE values and auth URL
        auth_data = twitter_oauth.get_authorization_url()
        if not auth_data or 'url' not in auth_data:
            logging.error("Failed to generate Twitter authorization URL")
            flash("Failed to connect to Twitter", "error")
            return redirect(url_for('auth.login'))
        
        # Store PKCE data in session using both naming conventions for compatibility
        session['oauth_state'] = auth_data.get('state')
        session['code_verifier'] = auth_data.get('code_verifier')
        session['code_challenge'] = auth_data.get('code_challenge')
        
        # Also store with twitter_ prefix
        session['twitter_oauth_state'] = auth_data.get('state')
        session['twitter_code_verifier'] = auth_data.get('code_verifier')
        session['twitter_code_challenge'] = auth_data.get('code_challenge')
        
        # Log session storage (sanitized)
        logging.info(f"Stored oauth_state in session: {auth_data.get('state')[:5]}... (truncated)")
        logging.info(f"Stored code_verifier in session length: {len(auth_data.get('code_verifier'))}")
        logging.info(f"Session keys after storage: {list(session.keys())}")
        
        # Force session to be saved
        session.modified = True
        
        # Redirect to Twitter auth page
        auth_url = auth_data['url']
        logging.info(f"Redirecting to Twitter authorization URL (truncated): {auth_url[:60]}...")
        
        return redirect(auth_url)
        
    except Exception as e:
        logging.error(f"Twitter auth error: {str(e)}")
        logging.error(traceback.format_exc())
        flash("An error occurred while connecting to Twitter", "error")
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
    """Handle the OAuth callback from Twitter."""
    try:
        logging.info("======= TWITTER OAUTH CALLBACK DEBUG START =======")
        # Log the full request URL and args for debugging
        logging.info(f"Full request URL: {request.url}")
        logging.info(f"Request args: {dict(request.args)}")
        logging.info(f"Session keys: {list(session.keys())}")
        
        # Check for OAuth error response
        if 'error' in request.args:
            error = request.args.get('error')
            error_description = request.args.get('error_description', 'No description provided')
            logging.error(f"OAuth error: {error} - {error_description}")
            flash(f"Authentication failed: {error_description}", "error")
            return redirect(url_for('auth.login'))
        
        # Extract OAuth data
        code = request.args.get('code')
        state = request.args.get('state')
        
        # Verify code and state
        if not code:
            logging.error("No authorization code in callback")
            flash("Authentication failed: No authorization code received", "error")
            return redirect(url_for('auth.login'))
        
        # Try both session key formats for compatibility
        expected_state = session.get('oauth_state') or session.get('twitter_oauth_state')
        if not expected_state:
            logging.error(f"No state in session - Session keys: {list(session.keys())}")
            flash("Authentication failed: Session expired", "error")
            return redirect(url_for('auth.login'))
        
        if state != expected_state:
            logging.error(f"State mismatch: {state} != {expected_state}")
            flash("Authentication failed: Security check failed", "error")
            return redirect(url_for('auth.login'))
        
        # Also try both formats for code verifier
        code_verifier = session.get('code_verifier') or session.get('twitter_code_verifier')
        if not code_verifier:
            logging.error("No code verifier in session")
            flash("Authentication failed: Session data missing", "error")
            return redirect(url_for('auth.login'))
        
        logging.info(f"Code: {code[:5]}... (truncated)")
        logging.info(f"State: {state[:5]}... (truncated)")
        logging.info(f"Code verifier: {code_verifier[:5]}... (truncated)")
        logging.info(f"Code verifier length: {len(code_verifier)}")
        
        # Exchange code for token
        from auth.oauth_final import TwitterOAuth
        
        # Initialize provider with correct config
        provider_config = {
            'client_id': os.environ.get('TWITTER_CLIENT_ID', ''),
            'client_secret': os.environ.get('TWITTER_CLIENT_SECRET', ''),
            'callback_url': os.environ.get('TWITTER_REDIRECT_URI', '')
        }
        
        # Log config (without sensitive info)
        logging.info(f"Twitter client_id length: {len(provider_config['client_id'])}")
        logging.info(f"Twitter client_secret length: {len(provider_config['client_secret'])}")
        logging.info(f"Twitter callback_url: {provider_config['callback_url']}")
        
        twitter_oauth = TwitterOAuth(provider_config)
        
        token_data = twitter_oauth.get_token(code, code_verifier)
        
        if not token_data or 'access_token' not in token_data:
            logging.error(f"Failed to get token: {token_data}")
            flash("Failed to authenticate with Twitter", "error")
            return redirect(url_for('auth.login'))
        
        # Get user info
        user_info = twitter_oauth.get_user_info(token_data['access_token'])
        if not user_info or 'id' not in user_info:
            logging.error(f"Failed to get user info: {user_info}")
            flash("Failed to get user information from Twitter", "error")
            return redirect(url_for('auth.login'))
        
        # Log user info (sanitized)
        logging.info(f"Twitter user ID: {user_info.get('id')}")
        logging.info(f"Twitter username: {user_info.get('username')}")
        
        # Find or create user
        from database.multi_user_db.db_final import get_db_connection, create_user, update_last_login, get_user_by_provider_id
        
        try:
            # Get database connection
            db_conn = get_db_connection()
            
            # Find or create user
            user = get_user_by_provider_id(db_conn, 'twitter', user_info.get('id'))
            if not user:
                # Create new user
                user = create_user(
                    db_conn,
                    username=user_info.get('username', ''),
                    email=f"{user_info.get('username', '')}@twitter.placeholder",
                    auth_provider='twitter',
                    provider_id=user_info.get('id'),
                    display_name=user_info.get('name', ''),
                    profile_image_url=user_info.get('profile_image_url', '')
                )
                logging.info(f"Created new user from Twitter: {user.id} - {user.username}")
            else:
                # Update last login
                update_last_login(db_conn, user.id)
                logging.info(f"Updated existing user from Twitter: {user.id} - {user.username}")
            
            if not user:
                logging.error("Failed to create or update user record")
                flash("An error occurred while setting up your account", "error")
                return redirect(url_for('auth.login'))
            
            # Store user ID in session
            session['user_id'] = str(user.id)
            session['username'] = user.username
            logging.info(f"Stored user ID in session: {user.id}")
        except Exception as db_error:
            logging.error(f"Database error: {str(db_error)}")
            logging.error(traceback.format_exc())
            flash("A database error occurred. Please try again.", "error")
            return redirect(url_for('auth.login'))
        
        # Clear OAuth-related session data
        try:
            for key in ['oauth_state', 'code_verifier', 'code_challenge', 
                      'twitter_oauth_state', 'twitter_code_verifier', 'twitter_code_challenge']:
                if key in session:
                    session.pop(key)
            
            # Force session to be saved
            session.modified = True
        except Exception as session_error:
            logging.error(f"Error clearing session: {str(session_error)}")
            # Continue even if session cleanup fails
        
        logging.info("Authentication successful")
        logging.info("======= TWITTER OAUTH CALLBACK DEBUG END =======")
        
        # Redirect to home
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"OAuth callback error: {str(e)}")
        logging.error(traceback.format_exc())
        flash("An error occurred during authentication", "error")
        return redirect(url_for('auth.login')) 