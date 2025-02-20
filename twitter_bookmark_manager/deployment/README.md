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