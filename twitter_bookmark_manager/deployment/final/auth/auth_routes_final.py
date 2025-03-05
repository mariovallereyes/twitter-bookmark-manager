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
    # Construct the callback URL
    twitter_callback_url = url_for('auth.oauth_callback', provider='twitter', _external=True)
    
    # Log the constructed URL for debugging
    logger.info(f"Constructed Twitter callback URL: {twitter_callback_url}")
    
    config = {
        'TWITTER_CONSUMER_KEY': os.environ.get('TWITTER_API_KEY'),
        'TWITTER_CONSUMER_SECRET': os.environ.get('TWITTER_API_SECRET', os.environ.get('TWITTER_CLIENT_SECRET')),
        'TWITTER_CALLBACK_URL': twitter_callback_url,
        'GOOGLE_CLIENT_ID': os.environ.get('GOOGLE_CLIENT_ID'),
        'GOOGLE_CLIENT_SECRET': os.environ.get('GOOGLE_CLIENT_SECRET'),
        'GOOGLE_REDIRECT_URI': url_for('auth.oauth_callback', provider='google', _external=True)
    }
    return OAuthManager(config)

# Login page
@auth_bp.route('/login')
def login():
    """Login page"""
    # Already logged in, redirect to home
    if UserContext.is_authenticated():
        return redirect(url_for('index'))
        
    # Store next parameter for redirect after login
    if request.args.get('next'):
        session['next'] = request.args.get('next')
        
    # Render login template
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
    if provider not in ['twitter', 'google']:
        flash(f"Unsupported authentication provider: {provider}", "error")
        return redirect(url_for('auth.login'))
        
    # Get user info from OAuth provider
    oauth_manager = get_oauth_manager()
    user_info = oauth_manager.get_user_info(provider, request.url)
    
    if not user_info:
        flash(f"Failed to get user info from {provider}", "error")
        return redirect(url_for('auth.login'))
        
    # Check if user exists in database
    conn = get_db_connection()
    try:
        # Look up user by provider and provider_user_id
        user = get_user_by_provider_id(
            conn, 
            user_info['provider'], 
            user_info['provider_user_id']
        )
        
        # Create user if not found
        if not user:
            user = create_user(
                conn,
                username=user_info['username'],
                email=user_info.get('email'),
                auth_provider=user_info['provider'],
                provider_user_id=user_info['provider_user_id'],
                display_name=user_info.get('display_name'),
                profile_image_url=user_info.get('profile_image_url')
            )
        else:
            # Update last login time
            user = update_last_login(conn, user.id)
            
        # Set user in session
        session['user_id'] = user.id
        session.permanent = True
        
        # Log the success
        logger.info(f"User '{user.username}' logged in via {provider}")
        
        # Redirect to next URL or home
        next_url = session.pop('next', None)
        if next_url and is_safe_url(next_url):
            return redirect(next_url)
        return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error in OAuth callback: {e}")
        flash("An error occurred during authentication. Please try again.", "error")
        return redirect(url_for('auth.login'))
    finally:
        conn.close()

# Logout
@auth_bp.route('/logout')
def logout():
    """Log out the current user"""
    user = UserContext.get_current_user()
    if user:
        logger.info(f"User '{user.username}' logged out")
        
    # Clear session
    session.pop('user_id', None)
    session.clear()
    
    # Redirect to home
    return redirect(url_for('index'))

# Profile page
@auth_bp.route('/profile')
def profile():
    """User profile page"""
    # Require login
    if not UserContext.is_authenticated():
        return redirect(url_for('auth.login', next=request.url))
        
    # Get current user
    user = UserContext.get_current_user()
    
    # Render profile template
    return render_template('profile_pa.html', user=user) 