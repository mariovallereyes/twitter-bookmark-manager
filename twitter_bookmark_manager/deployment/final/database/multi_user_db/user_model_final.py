"""
User model for the PythonAnywhere implementation.
Contains the database schema and model for users with OAuth authentication.
"""

from sqlalchemy import text

class User:
    """User model for the Twitter Bookmark Manager"""

    def __init__(self, id=None, username=None, email=None, auth_provider=None, 
                 provider_user_id=None, display_name=None, profile_image_url=None, 
                 created_at=None, last_login=None):
        self.id = id
        self.username = username
        self.email = email
        self.auth_provider = auth_provider  # 'twitter' or 'google'
        self.provider_user_id = provider_user_id  # ID from the OAuth provider
        self.display_name = display_name
        self.profile_image_url = profile_image_url
        self.created_at = created_at
        self.last_login = last_login

    @classmethod
    def from_row(cls, row):
        """Create a User object from database row tuple"""
        if not row:
            return None
            
        return cls(
            id=row[0],
            username=row[1],
            email=row[2],
            auth_provider=row[3],
            provider_user_id=row[4],
            display_name=row[5],
            profile_image_url=row[6],
            created_at=row[7],
            last_login=row[8]
        )

    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'auth_provider': self.auth_provider,
            'provider_user_id': self.provider_user_id,
            'display_name': self.display_name,
            'profile_image_url': self.profile_image_url,
            'created_at': self.created_at,
            'last_login': self.last_login
        }

def create_user_table(conn):
    """Create the users table if it doesn't exist"""
    # Check if this is a SQLAlchemy Session or a database connection
    if hasattr(conn, 'execute'):
        # It's a SQLAlchemy Session
        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE,
            auth_provider VARCHAR(50) NOT NULL,
            provider_user_id VARCHAR(255) NOT NULL,
            display_name VARCHAR(255),
            profile_image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );
        '''))
        conn.commit()
    else:
        # It's a database connection with cursor
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE,
            auth_provider VARCHAR(50) NOT NULL,
            provider_user_id VARCHAR(255) NOT NULL,
            display_name VARCHAR(255),
            profile_image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );
        ''')
        conn.commit()

def alter_tables_for_multi_user(conn):
    """Add user_id column to existing tables to support multi-user functionality"""
    # Check if this is a SQLAlchemy Session or a database connection
    if hasattr(conn, 'execute'):
        # It's a SQLAlchemy Session
        # Check if user_id column exists in bookmarks table
        result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='bookmarks' AND column_name='user_id';"))
        if not result.fetchone():
            # Add user_id to bookmarks table
            conn.execute(text("ALTER TABLE bookmarks ADD COLUMN user_id INTEGER REFERENCES users(id);"))
            
            # Set existing bookmarks to a default system user (will be created separately)
            # This ensures backwards compatibility
            conn.execute(text("UPDATE bookmarks SET user_id = 1 WHERE user_id IS NULL;"))
        
        # Check if user_id column exists in categories table
        result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='categories' AND column_name='user_id';"))
        if not result.fetchone():
            # Add user_id to categories table
            conn.execute(text("ALTER TABLE categories ADD COLUMN user_id INTEGER REFERENCES users(id);"))
            
            # Set existing categories to a default system user
            conn.execute(text("UPDATE categories SET user_id = 1 WHERE user_id IS NULL;"))
        
        conn.commit()
    else:
        # It's a database connection with cursor
        cursor = conn.cursor()
        
        # Check if user_id column exists in bookmarks table
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bookmarks' AND column_name='user_id';")
        if not cursor.fetchone():
            # Add user_id to bookmarks table
            cursor.execute("ALTER TABLE bookmarks ADD COLUMN user_id INTEGER REFERENCES users(id);")
            
            # Set existing bookmarks to a default system user (will be created separately)
            # This ensures backwards compatibility
            cursor.execute("UPDATE bookmarks SET user_id = 1 WHERE user_id IS NULL;")
        
        # Check if user_id column exists in categories table
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='categories' AND column_name='user_id';")
        if not cursor.fetchone():
            # Add user_id to categories table
            cursor.execute("ALTER TABLE categories ADD COLUMN user_id INTEGER REFERENCES users(id);")
            
            # Set existing categories to a default system user
            cursor.execute("UPDATE categories SET user_id = 1 WHERE user_id IS NULL;")
        
        conn.commit()

def get_user_by_provider_id(conn, provider, provider_id):
    """Get a user by their OAuth provider ID"""
    # Check if this is a SQLAlchemy Session or a database connection
    if hasattr(conn, 'execute'):
        # It's a SQLAlchemy Session
        result = conn.execute(text('''
        SELECT id, username, email, auth_provider, provider_user_id, 
               display_name, profile_image_url, created_at, last_login
        FROM users 
        WHERE auth_provider = :provider AND provider_user_id = :provider_id
        '''), {'provider': provider, 'provider_id': provider_id})
        return User.from_row(result.fetchone())
    else:
        # It's a database connection with cursor
        cursor = conn.cursor()
        cursor.execute('''
        SELECT id, username, email, auth_provider, provider_user_id, 
               display_name, profile_image_url, created_at, last_login
        FROM users 
        WHERE auth_provider = %s AND provider_user_id = %s
        ''', (provider, provider_id))
        return User.from_row(cursor.fetchone())

def get_user_by_id(conn, user_id):
    """Get a user by their internal ID"""
    # Check if this is a SQLAlchemy Session or a database connection
    if hasattr(conn, 'execute'):
        # It's a SQLAlchemy Session
        result = conn.execute(text('''
        SELECT id, username, email, auth_provider, provider_user_id, 
               display_name, profile_image_url, created_at, last_login
        FROM users 
        WHERE id = :user_id
        '''), {'user_id': user_id})
        return User.from_row(result.fetchone())
    else:
        # It's a database connection with cursor
        cursor = conn.cursor()
        cursor.execute('''
        SELECT id, username, email, auth_provider, provider_user_id, 
               display_name, profile_image_url, created_at, last_login
        FROM users 
        WHERE id = %s
        ''', (user_id,))
        return User.from_row(cursor.fetchone())

def create_user(conn, username, email, auth_provider, provider_user_id, 
                display_name=None, profile_image_url=None):
    """Create a new user in the database"""
    # Check if this is a SQLAlchemy Session or a database connection
    if hasattr(conn, 'execute'):
        # It's a SQLAlchemy Session
        import datetime
        
        now = datetime.datetime.now()
        result = conn.execute(text('''
        INSERT INTO users (username, email, auth_provider, provider_user_id, display_name, profile_image_url, created_at, last_login)
        VALUES (:username, :email, :auth_provider, :provider_user_id, :display_name, :profile_image_url, :created_at, :last_login)
        RETURNING id, username, email, auth_provider, provider_user_id, display_name, profile_image_url, created_at, last_login
        '''), {
            'username': username,
            'email': email,
            'auth_provider': auth_provider,
            'provider_user_id': provider_user_id,
            'display_name': display_name,
            'profile_image_url': profile_image_url,
            'created_at': now,
            'last_login': now
        })
        conn.commit()
        return User.from_row(result.fetchone())
    else:
        # It's a database connection with cursor
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO users (username, email, auth_provider, provider_user_id, display_name, profile_image_url)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, username, email, auth_provider, provider_user_id, 
                  display_name, profile_image_url, created_at, last_login
        ''', (username, email, auth_provider, provider_user_id, display_name, profile_image_url))
        conn.commit()
        return User.from_row(cursor.fetchone())

def update_last_login(conn, user_id):
    """Update the last login timestamp for a user"""
    # Check if this is a SQLAlchemy Session or a database connection
    if hasattr(conn, 'execute'):
        # It's a SQLAlchemy Session
        import datetime
        
        now = datetime.datetime.now()
        result = conn.execute(text('''
        UPDATE users 
        SET last_login = :now
        WHERE id = :user_id
        RETURNING id, username, email, auth_provider, provider_user_id, display_name, profile_image_url, created_at, last_login
        '''), {'now': now, 'user_id': user_id})
        conn.commit()
        return User.from_row(result.fetchone())
    else:
        # It's a database connection with cursor
        cursor = conn.cursor()
        # Update the last login time
        cursor.execute('''
        UPDATE users 
        SET last_login = NOW() 
        WHERE id = %s
        RETURNING id, username, email, auth_provider, provider_user_id, display_name, profile_image_url, created_at, last_login
        ''', (user_id,))
        updated_user = User.from_row(cursor.fetchone())
        conn.commit()
        return updated_user

def create_system_user_if_needed(conn):
    """Create a system user (ID=1) if it doesn't exist"""
    # Check if this is a SQLAlchemy Session or a database connection
    if hasattr(conn, 'execute'):
        # It's a SQLAlchemy Session
        result = conn.execute(text("SELECT id FROM users WHERE id = 1"))
        if not result.fetchone():
            conn.execute(text('''
            INSERT INTO users (id, username, email, auth_provider, provider_user_id, display_name)
            VALUES (1, 'system', 'system@example.com', 'system', 'system', 'System User')
            ON CONFLICT (id) DO NOTHING
            '''))
            conn.commit()
    else:
        # It's a database connection with cursor
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = 1")
        if not cursor.fetchone():
            cursor.execute('''
            INSERT INTO users (id, username, email, auth_provider, provider_user_id, display_name)
            VALUES (1, 'system', 'system@example.com', 'system', 'system', 'System User')
            ON CONFLICT (id) DO NOTHING
            ''')
            conn.commit() 