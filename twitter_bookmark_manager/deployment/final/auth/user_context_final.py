"""
User context management for final environment implementation.
Provides a way to track the current user throughout the request lifecycle.
"""

import functools
import flask
from flask import session, g, request, redirect, url_for

class UserContext:
    """User context manager for tracking the current user"""
    
    @staticmethod
    def get_current_user():
        """Get the current user from the request context"""
        if hasattr(g, 'user'):
            return g.user
        return None
    
    @staticmethod
    def set_current_user(user):
        """Set the current user in the request context"""
        g.user = user
        
    @staticmethod
    def is_authenticated():
        """Check if a user is authenticated"""
        return UserContext.get_current_user() is not None
    
    @staticmethod
    def get_user_id():
        """Get the current user ID or return 1 (system user) if not authenticated"""
        user = UserContext.get_current_user()
        if user and user.id:
            return user.id
        return 1  # Default to system user for backward compatibility

def login_required(f):
    """Decorator to require login for a route"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not UserContext.is_authenticated():
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def with_user_context(f):
    """Decorator to add user context to database operations"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Add user_id to kwargs if not already present
        if 'user_id' not in kwargs:
            kwargs['user_id'] = UserContext.get_user_id()
        return f(*args, **kwargs)
    return decorated_function

class UserContextMiddleware:
    """Middleware to set up user context for each request"""
    
    def __init__(self, app, user_loader):
        self.app = app
        self.user_loader = user_loader
        
        # Register before_request handler
        @app.before_request
        def load_user_from_session():
            # Clear any existing user
            g.user = None
            
            # Check if user is logged in via session
            user_id = session.get('user_id')
            if user_id:
                # Load user from database
                g.user = self.user_loader(user_id)
    
    def init_app(self, app):
        """Initialize middleware with a Flask app"""
        self.__init__(app, self.user_loader) 