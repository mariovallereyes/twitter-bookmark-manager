"""
User context management for the final deployment.
Provides user context tracking throughout request lifecycle.
"""

from functools import wraps
from flask import session, g, redirect, url_for, request, current_app
from database.multi_user_db.user_model_final import get_user_by_id

class UserContext:
    """Static class for managing user context"""
    
    @staticmethod
    def get_current_user():
        """Get the current user from the context"""
        if not hasattr(g, 'user'):
            user_id = session.get('user_id')
            if user_id:
                g.user = get_user_by_id(current_app.config['get_db_connection'](), user_id)
            else:
                g.user = None
        return g.user
        
    @staticmethod
    def set_current_user(user):
        """Set the current user in the context"""
        g.user = user
        if user:
            session['user_id'] = user.id
        else:
            session.pop('user_id', None)
            
    @staticmethod
    def is_authenticated():
        """Check if a user is currently authenticated"""
        return UserContext.get_current_user() is not None
        
    @staticmethod
    def get_user_id():
        """Get the current user's ID"""
        user = UserContext.get_current_user()
        return user.id if user else None

def login_required(f):
    """Decorator to require login for a route"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not UserContext.is_authenticated():
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

class UserContextMiddleware:
    """Middleware to set up user context for each request"""
    
    def __init__(self, app, get_user_func):
        self.app = app
        self.get_user_func = get_user_func
        self.init_app(app)
        
    def init_app(self, app):
        """Initialize the middleware with a Flask app"""
        app.before_request(self.load_user_from_session)
        
    def load_user_from_session(self):
        """Check if a user is logged in via session and load from database"""
        user_id = session.get('user_id')
        if user_id:
            user = self.get_user_func(current_app.config['get_db_connection'](), user_id)
            if user:
                UserContext.set_current_user(user)
            else:
                # Invalid user ID in session, clear it
                session.pop('user_id', None) 