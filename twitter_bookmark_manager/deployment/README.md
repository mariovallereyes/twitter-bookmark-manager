# Deployment Guide

This directory contains platform-specific deployment configurations and instructions for the Twitter Bookmark Manager.

## Directory Structure
```
deployment/
├── README.md                   # This file
├── pythonanywhere/            # PythonAnywhere-specific files
│   ├── database/              # Modified database files for PythonAnywhere
│   │   └── vector_store.py    # ChromaDB setup without is_persistent flag
│   └── requirements.txt       # Platform-specific dependencies
└── other_platforms/           # Future platform configurations
```

## PythonAnywhere Deployment

### Prerequisites
1. A PythonAnywhere account
2. Python 3.10 or higher
3. Access to the PythonAnywhere bash console

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd twitter-bookmark-manager
   ```

2. **Create Virtual Environment**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r deployment/pythonanywhere/requirements.txt
   ```

4. **Configure Web App**
   - Go to the Web tab in PythonAnywhere
   - Set Python version to 3.10
   - Set virtualenv path to `/home/yourusername/venv`
   - Set WSGI configuration file path
   - Update allowed hosts in settings if needed

5. **Environment Variables**
   - Set up necessary environment variables in the Web app configuration
   - Required variables:
     - `WIX_DOMAIN`
     - Other API keys as needed

6. **Database Setup**
   - Ensure the vector store directory exists
   - Initialize the database if needed

7. **File Structure**
   - Replace the default `vector_store.py` with the PythonAnywhere-specific version
   - Keep the original version for local development

### Troubleshooting

1. **ChromaDB Issues**
   - If you encounter ChromaDB compatibility issues, ensure you're using the PythonAnywhere-specific version of `vector_store.py`
   - Check the error logs in the PythonAnywhere dashboard

2. **Import Errors**
   - Verify Python path includes the project root
   - Check if all dependencies are installed correctly

3. **Permission Issues**
   - Ensure proper file permissions for the vector store directory
   - Check write permissions for log files

4. **Database Connection Issues**
   - If you see `"could not translate host name \"None\" to address"` errors:
     - Ensure the `.env.pythonanywhere` file has explicit database parameters:
       ```
       # Explicit database connection parameters (for SQLAlchemy)
       DB_USER=mariovallereyes
       DB_PASSWORD=${POSTGRES_PASSWORD}
       DB_HOST=mariovallereyes-4374.postgres.pythonanywhere-services.com
       DB_NAME=mariovallereyes$default
       ```
     - Check the environment variables are loading correctly with the `/api/status` endpoint
     - Reload the web app after making changes to environment variables
     - For troubleshooting, use the `/api/status` endpoint to see detailed database connection status
   
   - If you see `"cannot import name 'get_db_session' from 'database.db'"` errors:
     - This occurs when the PythonAnywhere database module (`db_pa.py`) is missing the `get_db_session` function alias
     - Make sure the following alias is present in `deployment/pythonanywhere/database/db_pa.py`:
       ```python
       # Alias for backward compatibility
       get_db_session = get_session
       ```
     - This ensures compatibility with code that expects the older function name

### Maintenance

1. **Updates**
   - When updating the application, always test changes locally first
   - Use git for version control and easy updates
   - Keep track of dependency changes

2. **Backups**
   - Regularly backup your vector store
   - Document any configuration changes

3. **Monitoring**
   - Check the error logs regularly
   - Monitor disk usage for vector store growth

## Support

For issues specific to PythonAnywhere deployment:
1. Check the PythonAnywhere help pages
2. Review the application error logs
3. Contact support if needed

## Local Development

For local development, continue using the original `vector_store.py` from the main project directory. The PythonAnywhere-specific version is only for deployment.

<!-- New Section: PythonAnywhere Execution Details -->
## PythonAnywhere Execution Details

The PythonAnywhere deployment of Twitter Bookmarks Manager incorporates several modifications to ensure optimal performance and reliability in the production environment, without affecting local development.

### Key Modifications

- **Environment Configuration**:
  - Environment variables are loaded from a dedicated `.env.pythonanywhere` file to configure the application securely.
  - The WSGI configuration (`wsgi.py`) and the API server (`api_server.py`) are adjusted to use absolute paths and appropriate logging for PythonAnywhere.

- **Database Setup**:
  - Instead of using SQLite, PostgreSQL is used as the primary relational database. The connection is managed via the `DATABASE_URL` environment variable.
  - Schema migration is handled by the script `deployment/pythonanywhere/postgres/migrate_schema.py`, and database initialization is performed by `deployment/pythonanywhere/postgres/init_db.py`.

- **Vector Store**:
  - The local ChromaDB vector store is replaced by a Qdrant-based implementation, configured in `deployment/pythonanywhere/database/vector_store_pa.py`.
  - This implementation includes deterministic UUID generation for bookmarks, ensuring consistency for vector operations.

- **Bookmark Update Process**:
  - Bookmark updates in the PythonAnywhere environment are executed via `deployment/pythonanywhere/database/update_bookmarks_pa.py`, which processes bookmarks in batches with enhanced error handling and duplicate checking.

These adjustments isolate the production setup from the local development configuration, allowing local execution to continue using SQLite and ChromaDB without interference. 