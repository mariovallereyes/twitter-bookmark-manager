# Project Structure

## Overview
This document provides an overview of the project's folder structure, key files, and their purpose. It serves as a reference for understanding how different components interact.

## Folder & File Breakdown

### Root Directory
- **`requirements.txt`** - Lists project dependencies
- **`main.py`** - Primary entry point of the application for local development
- **`.env`** - Stores environment variables (not included in version control)
- **`.env.example`** - Template for environment variables
- **`.env.pythonanywhere`** - Environment configuration specific to PythonAnywhere deployment
- **`.gitignore`** - Specifies files to be ignored by Git
- **`backup_project.py`** - Creates timestamped 7z archives of the entire project
- **`run_ingestion.py`** - Data ingestion script
- **`bookmarks.db`** - SQLite database file
- **`bookmark_launcher.py`** - Launcher script for the bookmark manager
- **`start_bookmarks.bat`** - Windows batch file to start the application
- **`update_log_*.txt`** - Log files for updates
- **`project_structure.txt`** - Project structure overview
- **`project_file_structure.txt`** - Detailed file structure listing
- **`__init__.py`** - Package initialization
- **`api_server.py`** - Flask application for PythonAnywhere deployment, works in tandem with the deployment folder
- **`wsgi.py`** - WSGI interface for PythonAnywhere deployment, handles environment setup and imports api_server.py

### `/00_instructions/` - Documentation
- **`00_MASTER.txt`** - Master documentation file
- **`system_architecture_deployment.md`** - System architecture and deployment guide
- **`project_structure_MD.md`** - This file
- **`01_general_project_overview.md`** - General project overview
- **`03_user_flow.md`** - User interaction flows
- **`04a_bookmark_extraction.md`** - Bookmark extraction process
- **`05_search_system.md`** - Search system documentation
- **`06_chat_rag_system.md`** - Chat and RAG system documentation

### `/core/` - Core Functionality
- **`__init__.py`** - Package initialization
- **`auth.py`** - Authentication logic
- **`data_ingestion.py`** - Data ingestion processing
- **`ai_categorization.py`** - AI-based categorization
- **`deduplication.py`** - Bookmark deduplication
- **`rag.py`** - Retrieval Augmented Generation
- **`search.py`** - Search functionality
- **`populate_vector_store.py`** - Vector database population
- **`process_categories.py`** - Category processing
- **`universal_bookmark_extractor.py`** - Generic bookmark extraction tool
- **`/chat/`** - Chat system components
  - **`__init__.py`** - Chat module initialization
  - **`engine.py`** - Chat engine implementation

### `/database/` - Data Storage & Management
- **`db.py`** - Database operations
- **`models.py`** - Database models
- **`vector_store.py`** - Vector database management
- **`update_bookmarks.py`** - Bookmark updates
- **`twitter_bookmarks.db`** - SQLite database
- **`twitter_bookmarks.json`** - JSON export of bookmarks
- **`/vector_db/`** - Vector database files
  - **`chroma.sqlite3`** - ChromaDB storage
- **`/json_history/`** - Historical bookmark data
  - **`twitter_bookmarks_*.json`** - Daily snapshots

### `/web/` - Web Application
- **`server.py`** - Main backend server
- **`test_server.py`** - Server tests
- **`/static/`** - Frontend assets
  - **`styles.css`** - Main stylesheet
  - **`script.js`** - Client-side JavaScript
  - **`/images/`** - Image assets
- **`/templates/`** - HTML templates
  - **`base.html`** - Base template
  - **`chat.html`** - Chat interface
  - **`index.html`** - Home page

### `/config/` - Configuration
- **`config.py`** - Main configuration class
- **`constants.py`** - Static configuration data
- **`__init__.py`** - Package initialization

### `/tests/` - Unit Tests
- **`test_auth.py`** - Authentication tests
- **`test_data_ingestion.py`** - Data ingestion tests
- **`test_models.py`** - Database model tests
- **`test_ai_categorization.py`** - AI categorization tests
- **`test_search.py`** - Search functionality tests
- **`test_rag.py`** - RAG system tests
- **`conftest.py`** - Test configuration

### `/backups/` - Backup Storage
- Contains 7z archives of project backups
- Naming: `twitter_bookmark_manager_backup_YYYYMMDD_HHMMSS.7z`

### `/models/` - AI Models
- **`mistral-7b-instruct-v0.1.Q4_K_M.gguf`** - Mistral model file

### `/temp_uploads/` - Temporary File Storage
- Temporary storage for uploaded files
- Cleaned periodically

### `/deployment/` - Deployment Configuration
- **`README.md`** - Deployment documentation and instructions
- **`__init__.py`** - Package initialization
- **`/pythonanywhere/`** - PythonAnywhere-specific code and configuration
  - **`/database/`** - Database adapters for PythonAnywhere
    - **`db_pa.py`** - PostgreSQL database connectivity for PythonAnywhere
    - **`vector_store_pa.py`** - Qdrant vector store implementation (replacing ChromaDB)
    - **`update_bookmarks_pa.py`** - PythonAnywhere-specific bookmark update logic
    - **`search_pa.py`** - Adapted search functionality for PythonAnywhere environment
  - **`/postgres/`** - PostgreSQL setup and migration scripts
    - **`config.py`** - PostgreSQL configuration
    - **`init_db.py`** - Database initialization script
    - **`migrate_schema.py`** - Schema migration tools for PostgreSQL
    - **`test_connection.py`** - Utility to test PostgreSQL connectivity

### `/media/` - Media and Assets
- Contains media files used by the application

### `/vector_db/` - Vector Database
- Storage for vector embeddings
- Used by the search functionality

### `/chroma/` - ChromaDB Storage
- ChromaDB vector database files

### `/build/` & `/dist/` - Packaging and Distribution
- **`build/`** - Build artifacts
- **`dist/`** - Distribution files
- **`Bilbeny's Bookmarks.spec`** - PyInstaller spec file
- **`BilbenysBookmarks.spec`** - Alternative PyInstaller spec file

### Additional Directories
- **`.cursor/`** - Cursor IDE configuration
- **`.pytest_cache/`** - pytest cache files
- **`venv/`** - Virtual environment
- **`__pycache__/`** - Python bytecode cache

## Notes
- All paths are relative to the project root
- Some directories are excluded from version control
- Backup system excludes specific directories (venv, cache, etc.)
- The project can be deployed both locally and on PythonAnywhere
- **PythonAnywhere Deployment**: The combination of `api_server.py`, `wsgi.py`, and the `/deployment/pythonanywhere/` directory work together to enable the application to run on PythonAnywhere:
  - `wsgi.py` sets up the Python environment specifically for PythonAnywhere
  - `api_server.py` provides the Flask application, configured for the PythonAnywhere environment
  - The deployment directory contains adaptations for PostgreSQL (replacing SQLite) and Qdrant (replacing ChromaDB)
  - This separation allows the codebase to run both locally and in production without code changes

## Current Directory Structure as of March 02, 2025

Directory structure:
└── mariovallereyes-twitter-bookmark-manager.git/
    ├── Documentation.txt
    ├── Twitter_Bookmark_Manager_Documentation.docx
    ├── api_server.py
    ├── generate_keys.py
    ├── god_script.py
    ├── process_categories_task.py
    ├── requirements.txt
    ├── run_schema_update.py
    ├── wsgi.py
    └── twitter_bookmark_manager/
        ├── __init__.py
        ├── backup_project.py
        ├── bookmark_launcher.py
        ├── main.py
        ├── project_file_structure.txt
        ├── project_structure.txt
        ├── requirements.txt
        ├── run_ingestion.py
        ├── start_bookmarks.bat
        ├── .env.example
        ├── .gitignore
        ├── 00_instructions/
        │   ├── 00_MASTER.txt
        │   ├── 01_general_project_overview.md
        │   ├── 02_main_components.md
        │   ├── 03_user_flow.md
        │   ├── 04a_bookmark_extraction.md
        │   ├── 04b_bookmark_ingestion.md
        │   ├── 05_search_system.md
        │   ├── 06_chat_rag_system.md
        │   ├── api_documentation.md
        │   ├── project_structure_MD.md
        │   ├── system_architecture_deployment.md
        │   └── technical_implementation_details.md
        ├── config/
        │   ├── config.py
        │   ├── constants.py
        │   └── secrets.json
        ├── core/
        │   ├── __init__.py
        │   ├── ai_categorization.py
        │   ├── auth.py
        │   ├── data_ingestion.py
        │   ├── deduplication.py
        │   ├── populate_vector_store.py
        │   ├── process_categories.py
        │   ├── rag.py
        │   ├── search.py
        │   ├── universal_bookmark_extractor.py
        │   ├── chat/
        │   │   ├── __init__.py
        │   │   ├── chat_search.py
        │   │   ├── conversation_manager.py
        │   │   ├── engine.py
        │   │   ├── intent_classifier.py
        │   │   └── prompt_manager.py
        │   └── dashboard/
        │       ├── __init__.py
        │       ├── author_analyzer.py
        │       ├── category_analyzer.py
        │       ├── dashboard_routes.py
        │       ├── data_processor.py
        │       ├── heatmap_analyzer.py
        │       └── topic_analyzer.py
        ├── database/
        │   ├── db.py
        │   ├── models.py
        │   ├── twitter_bookmarks.json
        │   ├── update_bookmarks.py
        │   ├── vector_store.py
        │   └── json_history/
        │       ├── twitter_bookmarks_20250131.json
        │       ├── twitter_bookmarks_20250201.json
        │       ├── twitter_bookmarks_20250202.json
        │       ├── twitter_bookmarks_20250203.json
        │       ├── twitter_bookmarks_20250204.json
        │       ├── twitter_bookmarks_20250206.json
        │       ├── twitter_bookmarks_20250207.json
        │       ├── twitter_bookmarks_20250211.json
        │       ├── twitter_bookmarks_20250216.json
        │       ├── twitter_bookmarks_20250220.json
        │       └── twitter_bookmarks_20250223.json
        ├── deployment/
        │   ├── __init__.py
        │   └── pythonanywhere/
        │       ├── README.md
        │       ├── __init__.py
        │       ├── requirements.txt
        │       ├── wsgi.py
        │       ├── database/
        │       │   ├── __init__.py
        │       │   ├── db_pa.py
        │       │   ├── db_schema_update.py
        │       │   ├── process_categories_pa.py
        │       │   ├── search_pa.py
        │       │   ├── update_bookmarks_pa.py
        │       │   └── vector_store_pa.py
        │       ├── postgres/
        │       │   ├── __init__.py
        │       │   ├── config.py
        │       │   ├── init_db.py
        │       │   ├── migrate_schema.py
        │       │   └── test_connection.py
        │       └── web_pa/
        │           └── templates/
        │               ├── base_pa.html
        │               ├── categories_pa.html
        │               └── index_pa.html
        ├── models/
        │   ├── Unconfirmed 364157.crdownload
        │   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
        ├── test_media/
        ├── tests/
        │   ├── __init__.py
        │   ├── conftest.py
        │   ├── test_ai_categorization.py
        │   ├── test_auth.py
        │   ├── test_bookmark_extractor.py
        │   ├── test_chat.py
        │   ├── test_count_bookmarks_from_json.py
        │   ├── test_data_ingestion.py
        │   ├── test_db.py
        │   ├── test_deduplication.py
        │   ├── test_llm.py
        │   ├── test_models.py
        │   ├── test_rag.py
        │   ├── test_real_query.py
        │   ├── test_search.py
        │   └── test_vector_store.py
        └── web/
            ├── server.py
            ├── test_server.py
            ├── database/
            │   ├── twitter_bookmarks.json
            │   └── json_history/
            │       └── twitter_bookmarks_20250226.json
            ├── static/
            │   ├── script.js
            │   ├── styles.css
            │   ├── images/
            │   └── js/
            │       └── dashboard/
            │           ├── authors.js
            │           ├── categories.js
            │           ├── heatmap.js
            │           ├── main.js
            │           └── topics.js
            └── templates/
                ├── base.html
                ├── chat.html
                ├── dashboard.html
                └── index.html
