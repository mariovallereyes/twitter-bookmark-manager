"""
Authentication routes for the PythonAnywhere implementation.
Handles login, logout, and OAuth callback functionality.
"""

import os
import logging
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
    """Display login page"""
    # Ensure session values are strings
    ensure_string_session()
    
    next_url = request.args.get('next', '/') 
    
    # Store the next URL in the session
    if next_url and is_safe_url(next_url):
        session['next'] = next_url
    
    # Clear the entire session except for 'next'
    next_url_temp = session.get('next')
    session.clear()
    if next_url_temp:
        session['next'] = next_url_temp
    
    logger.info("Session cleared for new login attempt")
    logger.info(f"Session keys after clearing: {list(session.keys())}")
        
    return render_template('login_final.html')

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
    logger.info(f"Session keys before processing: {list(session.keys())}")
    
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
        
        # Get user info
        try:
            oauth_manager = get_oauth_manager()
            logger.info("Getting user info from Twitter")
            user_info = oauth_manager.get_user_info('twitter', oauth_data)
            
            if not user_info or 'id' not in user_info:
                flash('Failed to get user information from Twitter. Please try again.', 'error')
                logger.error(f"Failed to get user info: {user_info}")
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
                user = create_user(
                    db, 
                    username=username,
                    email=f"{username}@twitter.placeholder",
                    auth_provider='twitter',
                    provider_id=provider_id,
                    display_name=name,
                    profile_image_url=avatar_url
                )
                logger.info(f"Created new user: {username} (ID: {user.id})")
            else:
                # Update last login
                update_last_login(db, user.id)
                logger.info(f"Logged in existing user: {username} (ID: {user.id})")
            
            # Store user ID in session as string
            session['user_id'] = str(user.id)
            logger.info(f"Set user_id in session: {session.get('user_id')}")
            
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