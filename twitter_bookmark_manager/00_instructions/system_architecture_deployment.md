# System Architecture & Deployment Guide - Twitter Bookmarks Manager

## **1. Overview**
This document describes the **system architecture** and **deployment process** of the **Twitter Bookmarks Manager**, with particular focus on the dual-model AI system and its configuration. It covers:
- **System Components & Interactions**
- **AI Model Configuration**
- **Installation & Setup**
- **Running the Application**
- **Production Deployment Recommendations**

---

## **2. System Architecture**
### **Component Breakdown**
The system consists of **six primary components**:

| Component | Description | Primary Files |
|-----------|------------|---------------|
| **Web Interface** | Frontend UI for searching, chatting, and managing bookmarks | `web/templates/base.html`<br>`web/templates/chat.html`<br>`web/templates/index.html`<br>`web/static/` |
| **Backend API** | Flask-based REST API handling search, chat, and database updates | `web/server.py`<br>`core/auth.py`<br>`core/search.py` |
| **Database Layer** | Stores structured bookmark data and vector embeddings | `database/db.py`<br>`database/models.py`<br>`database/vector_store.py`<br>`vector_db/` |
| **Vector Search Engine** | AI-powered semantic search using ChromaDB | `database/vector_store.py`<br>`core/search.py`<br>`vector_db/` |
| **Chat System** | Dual-model conversational AI (Gemini/Mistral) | `core/chat/engine.py`<br>`core/chat/chat_search.py` |
| **Model Management** | Handles model selection and fallback | `core/chat/engine.py`<br>`config/config.py` |

### **Directory Structure Overview**
```
twitter_bookmark_manager/
├── web/                # Frontend & Backend
├── core/              # Core Processing
│   └── chat/         # Chat Engine
├── database/         # Data Storage
├── vector_db/        # Vector Storage
├── config/           # Configuration
├── models/           # AI Models
└── 00_instructions/  # Documentation
```

### **Data Flow & Processing Pipeline**
1. **Bookmark Ingestion:**
   - Users upload a **Twitter bookmark JSON file** via the web interface (`web/templates/base.html`).
   - The system **parses, stores, and categorizes bookmarks** (`database/update_bookmarks.py`, `core/process_categories.py`).
   - **Vector embeddings** are generated (`database/vector_store.py`) for semantic search.

2. **Search & Retrieval:**
   - Users search via **keywords, categories, or author-based queries** (`search.py`).
   - Hybrid search: **vector embeddings (ChromaDB) + SQL text search (SQLite)**.

3. **AI-Powered Chat:**
   - Users interact with **Mistral-7B-powered AI** via `/api/chat` (`engine.py`).
   - AI responses **retrieve relevant bookmarks** using **RAG**.

4. **Database Updates & Maintenance:**
   - Users can **upload new bookmarks**.
   - The system **automatically updates categories & vector embeddings**.

### **API Endpoints & Integration**
The system exposes several REST API endpoints:

#### **1️⃣ Search API**
- **Endpoint:** `/api/search`
- **Method:** POST
- **Purpose:** Search bookmarks with text, categories, or author filters
- **Rate Limit:** 60 requests/minute
- **Response Format:** JSON with ranked results

#### **2️⃣ Chat API**
- **Endpoint:** `/api/chat`
- **Method:** POST
- **Purpose:** AI-powered conversational interface
- **Rate Limit:** 30 requests/minute
- **Response Format:** JSON with AI response and referenced bookmarks

#### **3️⃣ Bookmark Management**
- **Endpoint:** `/api/upload-bookmarks`
- **Method:** POST
- **Purpose:** Upload and process bookmark JSON files
- **Rate Limit:** 10 requests/minute
- **File Size:** Max 10MB

#### **Security & Error Handling**
- Authentication: Not required (planned for future)
- Standard error responses in JSON format
- Rate limiting per endpoint
- File size restrictions

---

## **3. Technology Stack**
| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **Flask** | Backend web framework (API handling) |
| **SQLAlchemy** | ORM for SQLite database interactions |
| **ChromaDB** | Vector search database for semantic search |
| **Mistral-7B** | Primary AI model for chat interactions (via llama-cpp-python) |
| **Gemini 2.0** | Alternative AI model for chat interactions (via google-generativeai) |
| **Sentence Transformers** | Converts text into embeddings for vector search |
| **Alpine.js + TailwindCSS** | Frontend interactivity & styling |

### **Model Configuration**
The system supports two LLM options:
- **Mistral-7B** (default): Local model using llama-cpp-python
- **Gemini 2.0**: Cloud-based option using Google's API

Selection is controlled via the `CHAT_MODEL` environment variable:
```bash
# For Mistral (default)
export CHAT_MODEL=mistral

# For Gemini
export CHAT_MODEL=gemini
export GEMINI_API_KEY="your-api-key"
```

---

## **4. Installation & Setup**
### **1️⃣ Prerequisites**
Ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **SQLite** (bundled with Python)
- **Git** (optional, for version control)

### **2️⃣ Clone the Repository**
```bash
git clone <repository_url>
cd twitter_bookmark_manager
```

### **3️⃣ Create & Activate Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **4️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **5. Running the Application**
### **1️⃣ Initialize the Database**
```bash
python -c "from database.db import init_database; init_database()"
```

### **2️⃣ Start the Flask Server**
```bash
python server.py
```

- The application will run at **`http://127.0.0.1:5000/`**.
- Access the **search UI** and **chat interface** via the browser.

---

## **6. Deployment Guide (Production)**
For production environments, consider:

### **1️⃣ Model Preparation**
For Gemini 2.0:
```bash
# Install Google AI library
pip install google-generativeai

# Set environment variables
export CHAT_MODEL=gemini
export GEMINI_API_KEY=your-api-key
```

For Mistral-7B:
```bash
# Install required packages
pip install torch transformers

# Download model
wget https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/model.gguf
mv model.gguf models/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# Set environment variables
export CHAT_MODEL=mistral
export MISTRAL_MODEL_PATH=models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### **2️⃣ Running with Gunicorn**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 server:app
```

### **3️⃣ Docker Deployment**
Updated Dockerfile with model support:
```dockerfile
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y wget

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Download Mistral model (if using)
RUN if [ "$CHAT_MODEL" = "mistral" ]; then \
    wget -P models/ https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/model.gguf; \
    fi

# Set environment variables
ENV CHAT_MODEL=${CHAT_MODEL:-gemini}
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV MISTRAL_MODEL_PATH=${MISTRAL_MODEL_PATH:-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf}

# Run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "server:app"]
```

Build & run:
```bash
docker build -t twitter_bookmark_manager .
docker run -p 8000:8000 \
  -e CHAT_MODEL=gemini \
  -e GEMINI_API_KEY=your-api-key \
  twitter_bookmark_manager
```

### **4️⃣ PythonAnywhere Deployment**
For deploying on PythonAnywhere, the system uses a different database configuration than local development:

#### **Database Configuration**
- **Local Development**: SQLite + ChromaDB
- **PythonAnywhere**: PostgreSQL + Qdrant

#### **PostgreSQL Setup**
```bash
# Environment variables for PostgreSQL
export POSTGRES_DB=your_database_name
export POSTGRES_USER=your_username
export POSTGRES_PASSWORD=your_password
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
```

#### **Qdrant Vector Store**
```bash
# Qdrant configuration
export VECTOR_STORE_TYPE=qdrant
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_COLLECTION=bookmarks
```

#### **Migration Notes**
- The system automatically detects the environment and uses appropriate database connections
- Local development continues to use SQLite + ChromaDB
- PythonAnywhere deployment uses PostgreSQL + Qdrant
- Data migration scripts are provided in `deployment/pythonanywhere/database/`

#### **Environment Setup**
```bash
# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install platform-specific dependencies
pip install -r deployment/pythonanywhere/requirements.txt
```

#### **ChromaDB Configuration**
PythonAnywhere requires specific ChromaDB configuration due to version compatibility (0.3.21):

1. **Modified Vector Store Implementation**
   - Location: `deployment/pythonanywhere/database/vector_store.py`
   - Key changes:
     - Uses `chromadb.Client()` instead of `PersistentClient`
     - Simplified settings initialization
     - Local caching for sentence transformer model
     - Cosine similarity space configuration

2. **Directory Structure**
```
deployment/
├── pythonanywhere/
│   ├── database/
│   │   └── vector_store.py    # PythonAnywhere-compatible version
│   └── requirements.txt       # Platform-specific dependencies
└── README.md                  # Deployment instructions
```

#### **WSGI Configuration**
Create a WSGI configuration file (`wsgi.py`):
```python
import sys
import os
from pathlib import Path

# Add the project directory to the Python path
project_path = Path(__file__).parent
sys.path.append(str(project_path))

# Import your Flask app
from api_server import app as application

# Optional: Set environment variables if needed
os.environ['WIX_DOMAIN'] = 'your-wix-domain.com'
```

#### **API Server Setup**
The `api_server.py` file is configured specifically for PythonAnywhere:
- CORS configuration for domain access
- Proper path handling for imports
- Enhanced logging setup
- Health check endpoint for monitoring

#### **Deployment Steps**
1. Upload code to PythonAnywhere
2. Configure Web app:
   - Set Python version to 3.10
   - Set virtualenv path
   - Configure WSGI file
   - Set up static files
3. Install dependencies from `deployment/pythonanywhere/requirements.txt`
4. Initialize database and vector store
5. Reload the web app

#### **Troubleshooting**
Common issues and solutions:
- ChromaDB version conflicts: Use provided PythonAnywhere-specific version
- Import errors: Check Python path in WSGI file
- Memory issues: Monitor resource usage in PythonAnywhere dashboard
- Model loading: Use local caching for sentence transformers

---

## **7. Environment Variables & Configuration**

### **7.1 Configuration Management**
The project uses a centralized configuration management system through:
1. Environment variables (loaded from `.env` file)
2. Configuration class (`config/config.py`)
3. Constants (`config/constants.py` for static data)

### **7.2 Environment Variables**
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
# Edit .env with your settings
```

#### **Required Variables**
| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask session encryption key | None (Required) |
| `ENCRYPTION_KEY` | Data encryption key | None (Required) |
| `DATABASE_URL` | SQLite database location | `sqlite:///database/twitter_bookmarks.db` |
| `CHAT_MODEL` | AI model selection (`gemini` or `mistral`) | `gemini` |
| `GEMINI_API_KEY` | Gemini API key (required if using Gemini) | None |
| `MISTRAL_MODEL_PATH` | Path to Mistral model (required if using Mistral) | `models/mistral-7b-instruct-v0.1.Q4_K_M.gguf` |

### **7.3 Model Configuration**

#### **Gemini 2.0 (Default)**
```bash
CHAT_MODEL=gemini
GEMINI_API_KEY=your-api-key-here
```

#### **Mistral-7B (Fallback)**
```bash
CHAT_MODEL=mistral
MISTRAL_MODEL_PATH=models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### **7.4 Model Selection Logic**
1. System attempts to use Gemini 2.0 by default
2. Falls back to Mistral-7B if:
   - Gemini API key is missing
   - Gemini API is unavailable
   - Import errors occur
3. Raises error if neither model is available

### **7.5 Configuration Validation**
The system validates required settings on startup:
```python
from config.config import config

# Validates and returns current configuration
settings = config.validate()
```

### **7.6 Rate Limits & Constants**
Defined in `config.py`:
- Search API: 60 requests/minute
- Chat API: 30 requests/minute
- Upload API: 10 requests/minute
- Max upload size: 10MB

### **7.7 Security Best Practices**
1. Never commit `.env` file (included in `.gitignore`)
2. Use `.env.example` as a template
3. Generate secure keys using `generate_keys.py`
4. Store production secrets securely
5. Use different keys for development/production

## **8. Backup Systems & Data Retention**

### **8.1 Project-wide Backups**
The system includes automated project-wide backups via `backup_project.py`:

```bash
# Run project backup
python backup_project.py
```

#### **Configuration**
- **Backup Location:** `backups/` directory
- **Format:** 7z compressed archives
- **Naming:** `twitter_bookmark_manager_backup_YYYYMMDD_HHMMSS.7z`
- **Excluded Items:**
  - Virtual environment (`venv/`)
  - Cache files (`__pycache__/`)
  - Version control (`.git/`)
  - Model files (`models/`)
  - Temporary uploads (`temp_uploads/`)

#### **Backup Process**
1. Creates timestamped archive
2. Logs backup progress
3. Verifies archive integrity
4. Reports backup size and status

### **8.2 Bookmark History Management**
The system maintains historical bookmark data:

#### **Storage Location**
- Directory: `database/json_history/`
- Format: Daily JSON snapshots
- Naming: `twitter_bookmarks_YYYYMMDD.json`

#### **Retention Policy**
- Maintains daily snapshots
- Enables version comparison
- Facilitates data recovery
- Supports trend analysis

### **8.3 Backup Best Practices**
🔹 Run project backups before major updates
🔹 Verify backup integrity regularly
🔹 Monitor backup storage space
🔹 Test restoration procedures
🔹 Document any backup failures

---

## **9. Future Enhancements**
🔹 **Multi-user authentication** (OAuth-based login)  
🔹 **Automated Twitter API bookmark fetching**  
🔹 **Scalable cloud deployment (AWS, GCP, Azure)**  
🔹 **Performance optimizations for AI search & chat latency**  
🔹 **Integration with other AI models (Gemini 2.0, GPT-4, etc.)**

---

## **10. Gemini 2.0 Integration**
In this update, the AI chat component has been upgraded to use Gemini 2.0 instead of the previous Mistral-7B model. Gemini 2.0 is an advanced large language model offering enhanced natural language understanding, improved context handling, and more fluid conversational responses.

### **Key Integration Points**

#### **1️⃣ API Integration**
The AI chat module now communicates with the Gemini 2.0 API. This requires updating the model loading and inference calls (replacing `llama-cpp-python` with the appropriate Gemini 2.0 integration library or API client).

#### **2️⃣ Configuration Changes**
| Variable | Description |
|----------|-------------|
| `GEMINI_MODEL_PATH` | Path to the Gemini 2.0 model/API endpoint |
| `GEMINI_API_KEY` | Authentication key for Gemini API access |

#### **3️⃣ Performance & Quality Improvements**
🔹 More natural conversation flow  
🔹 Improved handling of long context  
🔹 Better context-aware responses during AI chat interactions

#### **4️⃣ Deployment Adjustments**
The deployment process remains similar, but ensure:
```bash
pip install google-generativeai
```
Update environment variables:
```bash
export GEMINI_API_KEY="your-api-key"
```

---

**This guide provides full details on system setup, architecture, and deployment. Future improvements should be documented here.** 🚀

<!-- New Section: PythonAnywhere Execution Details -->
## 11. PythonAnywhere Execution Details

This section describes the modifications and configurations specific to PythonAnywhere deployment. These adjustments do not affect local execution, which continues to use the original configuration (SQLite for the database and ChromaDB for the vector store).

### Key PythonAnywhere-Specific Adjustments

#### Database Configuration
- **Local Setup**: Uses SQLite with `sqlite:///database/twitter_bookmarks.db`.
- **PythonAnywhere Deployment**: Uses PostgreSQL, configured via the `DATABASE_URL` environment variable. Schema migration and initialization are handled by `deployment/pythonanywhere/postgres/migrate_schema.py` and `deployment/pythonanywhere/postgres/init_db.py`.

#### Vector Store Configuration
- **Local Setup**: Utilizes ChromaDB for vector storage.
- **PythonAnywhere Deployment**: Utilizes Qdrant, implemented in `deployment/pythonanywhere/database/vector_store_pa.py`. Notable modifications include deterministic UUID generation for bookmarks using `uuid.uuid5` and integration with Qdrant's client for upsert, search, and deletion operations.

#### Bookmark Update Process
- PythonAnywhere-specific bookmark updates are managed by `deployment/pythonanywhere/database/update_bookmarks_pa.py`, which updates the PostgreSQL database and Qdrant vector store in batches, featuring enhanced error handling and duplicate prevention.

#### WSGI and API Server Configuration
- The WSGI configuration in `wsgi.py` and the API server in `api_server.py` are modified to load environment variables from `.env.pythonanywhere`, and to adapt file paths and logging for the PythonAnywhere environment.

### Implementation Details

#### WSGI Implementation (`wsgi.py`)
The `wsgi.py` file serves as the entry point for the PythonAnywhere web server and performs several critical functions:

1. **Path Setup and Environment Configuration**:
   - Sets project paths for correct module imports
   - Loads PythonAnywhere-specific environment variables from `.env.pythonanywhere`
   - Configures logging for comprehensive error tracking

2. **Import Handling and Module Patching**:
   - Blocks ChromaDB imports to prevent SQLite version conflicts (PythonAnywhere has an older SQLite version)
   - Creates compatibility layers for libraries like `huggingface_hub`
   - Pre-loads PythonAnywhere database modules to ensure proper initialization

3. **Custom Import Hook**:
   - Installs a fallback import hook to redirect import requests
   - Ensures database imports use the PythonAnywhere-specific implementations
   - Maintains compatibility with existing code expecting certain module paths

#### API Server Implementation (`api_server.py`)
The `api_server.py` file implements the Flask application that handles all HTTP requests:

1. **Initialization and Configuration**:
   - Sets up absolute paths for PythonAnywhere environment
   - Configures enhanced logging with session tracking
   - Creates required directories (logs, temp_uploads, etc.)

2. **API Endpoints**:
   - Implements standard endpoints (`/search`, `/api/chat`, etc.)
   - Adds PythonAnywhere-specific endpoints (`/api/status`, `/debug-database`)
   - Enhances error handling with detailed responses

3. **File Upload Handling**:
   - Implements robust file validation
   - Creates timestamped backups before processing
   - Uses session IDs for tracking operations

4. **Database Update Process**:
   - Implements batch processing with pause/resume capability
   - Tracks progress for long-running operations
   - Uses transaction-based operations for consistency

### PostgreSQL Database Adapter (`db_pa.py`)
The PostgreSQL adapter provides a compatible interface with the following features:

1. **Connection Pooling**:
   - Efficiently manages database connections
   - Implements connection retry logic
   - Uses environment variables for secure configuration

2. **Session Management**:
   - Provides context managers for database sessions
   - Ensures proper transaction handling
   - Maintains backward compatibility with existing code

3. **Database Status Monitoring**:
   - Implements health check functions
   - Provides detailed error reporting
   - Supports administrative queries

### Qdrant Vector Store Implementation (`vector_store_pa.py`)
The Qdrant vector store implementation replaces ChromaDB with the following features:

1. **API Compatibility**:
   - Implements the same interface as the ChromaDB version
   - Allows seamless switching between environments
   - Supports all required operations (add, search, delete)

2. **Deterministic UUID Generation**:
   - Uses `uuid.uuid5` with a namespace for consistent IDs
   - Ensures bookmark vectors have the same ID across operations
   - Prevents duplicate entries and enables efficient updates

3. **Vector Operations**:
   - Optimized batch upsert operations
   - Efficient similarity search with filtering
   - Proper error handling and retry logic

### Environment Configuration (`.env.pythonanywhere`)
The PythonAnywhere environment is configured through a dedicated `.env.pythonanywhere` file:

1. **Database Configuration**:
   - PostgreSQL connection parameters
   - Qdrant connection settings
   - Explicit database credentials

2. **Path Configuration**:
   - Absolute paths for PythonAnywhere environment
   - Directory locations for logs, uploads, and data

3. **AI Model Configuration**:
   - API keys for cloud-based AI services
   - Model selection parameters
   - Performance settings

These changes ensure that the production environment on PythonAnywhere is optimized and runs reliably, while local development remains unaffected.