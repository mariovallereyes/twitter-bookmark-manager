# Twitter Bookmarks Manager - Technical Overview

## Introduction

The **Twitter Bookmarks Manager** is a specialized web application designed to **enhance Twitter's bookmarking system** by allowing users to **import, organize, search, and interact with bookmarks** more efficiently. The project extends Twitter's native functionality by implementing **AI-based categorization, advanced search, and a conversational interface**.

## Key Features

### 1. Bookmark Import & Storage
- Accepts Twitter **bookmark exports in JSON format**.
- Stores bookmarks in a **structured SQLite database**.
- Maintains **historical snapshots** of imports for version control.
- Extracts and organizes **metadata**, including author details, timestamps, and media content.

### 2. Advanced Search Capabilities
- **Hybrid Search System**: Combines **traditional keyword-based text search** with **semantic search using vector embeddings**.
- **Category-Based Filtering**: Enables filtering by **AI-assigned** categories.
- **Author-Based Search**: Find all bookmarks from specific **Twitter users**.
- **Relevance Ranking**: Uses a scoring mechanism to prioritize relevant results.

### 3. AI-Powered Automatic Categorization
- Uses **AI models** to analyze bookmarks and assign relevant **categories** dynamically.
- Allows **multiple category assignments per bookmark** for better organization.
- Evolves its categorization system based on the content imported.

### 4. Interactive Chat Interface (Conversational AI)
- Enables users to **interact with bookmarks via AI**.
- Uses **Retrieval-Augmented Generation (RAG)** to generate contextual responses.
- Supports **natural language queries** to retrieve information.

## Technical Architecture

### 1. Frontend Layer (User Interface)
- **Built with Tailwind CSS & Alpine.js** for a modern and responsive UI.
- Provides **real-time search updates**.
- Includes an **interactive chat module**.
- Offers a **file upload system** for importing bookmarks.

### 2. Application Layer (Backend API & Processing)
- **Flask-based web server** handling user requests.
- **RESTful API endpoints**:
  - `/api/search`: Bookmark search and filtering
  - `/api/chat`: AI-powered conversational interface
  - `/api/upload-bookmarks`: Bookmark ingestion
- **Rate limiting and error handling**:
  - Search: 60 requests/minute
  - Chat: 30 requests/minute
  - Upload: 10 requests/minute
- **Authentication & session management** for user control.

### 3. Data Layer (Database & Storage)
- **SQLite database** for structured data storage.
- **ChromaDB** for **vector-based semantic search**.
- **JSON file storage** for historical bookmark data.
- **Media storage** for Twitter media attachments.

### 4. AI Layer (Natural Language Processing & Categorization)
- Supports multiple LLM backends:
  - **Gemini 2.0** for cloud-based processing (default)
  - **Mistral-7B** for local processing (fallback)
- Uses **SentenceTransformers** for text embedding generation
- Implements **AI-powered classification** for categorizing bookmarks
- **Retrieval-Augmented Generation (RAG)** system for enhanced conversational capabilities
- Dynamic model selection via environment configuration

### 5. Deployment Architecture
- **Dual Deployment Capability**: The system is designed to run in both local development and production environments.
- **Local Development**:
  - Uses **SQLite** for relational database storage
  - Uses **ChromaDB** for vector embeddings
  - Entry point via `main.py` and `server.py`
  - File-based configuration via `.env`
- **PythonAnywhere Production**:
  - Uses **PostgreSQL** for relational database storage
  - Uses **Qdrant** for vector embeddings
  - Entry point via `wsgi.py` and `api_server.py`
  - PythonAnywhere-specific configuration via `.env.pythonanywhere`
  - Enhanced error handling and logging optimized for production
  - Same codebase with environment-specific adapters in the `deployment` folder

## Data Flow Overview

1. **API Request Handling**:
   - Requests arrive at Flask endpoints
   - Rate limiting and validation applied
   - Requests routed to appropriate handlers
   - Standard error responses for failures

2. **Bookmark Import Process**:
   - User uploads **Twitter bookmarks JSON** via `/api/upload-bookmarks`
   - System **validates** the file (size, format)
   - Raw JSON is archived in `database/json_history/` with timestamp
   - Data is **processed and stored** in the database
   - Vector embeddings are **generated and stored** for search

3. **Search Process**:
   - User queries sent to `/api/search`
   - System processes query through:
     - **Vector search (semantic matching)**
     - **SQL text search (keyword matching)**
     - **Category-based filtering**
   - Results are **ranked and returned** as JSON

4. **Chat Interaction**:
   - User messages sent to `/api/chat`
   - System **retrieves relevant bookmarks** using vector search
   - AI **generates a contextual response**
   - Response includes referenced bookmarks and metadata
   - Conversation history is **maintained for context**

## System Requirements

- **Python 3.8+**
- **Flask web framework**
- **SQLite for database** (local development)
- **PostgreSQL** (PythonAnywhere production)
- **ChromaDB for vector storage** (local development)
- **Qdrant for vector storage** (PythonAnywhere production)
- **GPU recommended for AI-powered features** (local development)
- **Web browser for frontend access**

## Next Steps

This document serves as an **overview** of the system. For more detailed breakdowns, refer to:
- **Main Components Documentation**
- **User Flow Documentation**
- **Bookmark Ingestion Overview**
- **update_bookmarks.py Documentation**
- **base.html Documentation**

