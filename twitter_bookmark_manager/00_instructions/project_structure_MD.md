# Project Structure

## Overview
This document provides an overview of the project's folder structure, key files, and their purpose. It serves as a reference for understanding how different components interact.

## Folder & File Breakdown

### Root Directory
- **`requirements.txt`** - Lists project dependencies
- **`main.py`** - Primary entry point of the application
- **`.env`** - Stores environment variables (not included in version control)
- **`.env.example`** - Template for environment variables
- **`.gitignore`** - Specifies files to be ignored by Git
- **`backup_project.py`** - Creates timestamped 7z archives of the entire project
- **`run_ingestion.py`** - Data ingestion script
- **`bookmarks.db`** - SQLite database file
- **`update_log_*.txt`** - Log files for updates
- **`project_structure.txt`** - Project structure overview

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
- **`/chat/`** - Chat system components
  - **`__init__.py`** - Chat module initialization
  - **`engine.py`** - Chat engine implementation

### `/database/` - Data Storage & Management
- **`db.py`** - Database operations
- **`models.py`** - Database models
- **`vector_store.py`** - Vector database management
- **`update_bookmarks.py`** - Bookmark updates
- **`twitter_bookmarks.db`** - SQLite database
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

### Additional Directories
- **`.cursor/`** - Cursor IDE configuration
- **`.pytest_cache/`** - pytest cache files
- **`build/`** - Build artifacts
- **`dist/`** - Distribution files
- **`venv/`** - Virtual environment

## Notes
- All paths are relative to the project root
- Some directories are excluded from version control
- Backup system excludes specific directories (venv, cache, etc.)
