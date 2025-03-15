"""
User context management for the final deployment.
Provides user context tracking throughout request lifecycle.
"""

from functools import wraps
from flask import session, g, redirect, url_for, request, current_app, jsonify
from database.multi_user_db.user_model_final import get_user_by_id

class UserContext:
    """Static class for managing user context"""
    
    @staticmethod
    def get_current_user():
        """Get the current user from the context"""
        if not hasattr(g, 'user'):
            user_id = session.get('user_id')
            
            # Convert bytes to string if needed
            if isinstance(user_id, bytes):
                try:
                    user_id = user_id.decode('utf-8')
                    # Update session with string value
                    session['user_id'] = user_id
                    session.modified = True
                    current_app.logger.info(f"Converted user_id from bytes to string: {user_id}")
                except Exception as e:
                    current_app.logger.error(f"Error converting user_id from bytes: {e}")
                    user_id = None
                    session.pop('user_id', None)
            
            if user_id:
                try:
                    db_conn = current_app.config.get('get_db_connection')
                    if not db_conn:
                        current_app.logger.error("Database connection function not found in app config")
                        g.user = None
                        return None
                        
                    conn = db_conn()
                    g.user = get_user_by_id(conn, user_id)
                    
                    # If user not found, clear the session
                    if not g.user:
                        current_app.logger.warning(f"User ID {user_id} from session not found in database")
                        session.pop('user_id', None)
                except Exception as e:
                    current_app.logger.error(f"Error getting user from database: {e}")
                    # Don't clear session here - might be a temporary DB issue
                    g.user = None
            else:
                g.user = None
        return g.user
        
    @staticmethod
    def set_current_user(user):
        """Set the current user in the context"""
        g.user = user
        if user:
            # Ensure we store a string ID, not bytes
            user_id = user.id
            if isinstance(user_id, bytes):
                try:
                    user_id = user_id.decode('utf-8')
                    current_app.logger.info(f"Converted user.id from bytes to string: {user_id}")
                except Exception as e:
                    current_app.logger.error(f"Error converting user.id from bytes: {e}")
            
            # Store as string
            session['user_id'] = str(user_id)
            session.modified = True
            current_app.logger.info(f"Set user in session: {user_id}")
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

def with_user_context(f):
    """Decorator to ensure user context is available and handle API responses"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = UserContext.get_current_user()
        if user is None:
            # Check if this is an API request
            is_api_request = (
                request.path.startswith('/api/') or 
                request.headers.get('Accept') == 'application/json' or
                request.headers.get('Content-Type') == 'application/json'
            )
            
            if is_api_request:
                return jsonify({
                    'success': False,
                    'authenticated': False,
                    'error': 'User not authenticated'
                }), 401
            else:
                return redirect(url_for('auth.login', next=request.url))
                
        # Add user to kwargs for the wrapped function
        kwargs['user'] = user
        return f(*args, **kwargs)
    return decorated_function

def login_required(f):
    """Decorator to require login for a route"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        
        # Log session details for debugging
        current_app.logger.debug(f"login_required check - Path: {request.path} - Session ID: {session.sid if hasattr(session, 'sid') else 'No SID'} - User ID: {user_id}")
        
        # Check for user_id in session first to avoid unnecessary database operations
        if not user_id:
            current_app.logger.warning(f"No user_id in session for path: {request.path}")
            # Check if this is an API or AJAX request
            is_ajax_or_api = (
                request.path.startswith('/api/') or 
                request.headers.get('Accept') == 'application/json' or
                request.headers.get('Content-Type') == 'application/json' or
                request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            )
            
            if is_ajax_or_api:
                return jsonify({
                    'success': False,
                    'authenticated': False,
                    'error': 'User not authenticated. Please log out and log in again.',
                    'status': 'session_missing'
                }), 401
            return redirect(url_for('auth.login', next=request.url))
            
        # Only check the database if there's a user_id in the session
        if not UserContext.is_authenticated():
            current_app.logger.warning(f"User ID {user_id} from session not found in database for path: {request.path}")
            # Check if this is an API or AJAX request
            is_ajax_or_api = (
                request.path.startswith('/api/') or 
                request.headers.get('Accept') == 'application/json' or
                request.headers.get('Content-Type') == 'application/json' or
                request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            )
            
            if is_ajax_or_api:
                return jsonify({
                    'success': False,
                    'authenticated': False,
                    'error': 'User not authenticated. Please log out and log in again.',
                    'status': 'user_not_found'
                }), 401
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