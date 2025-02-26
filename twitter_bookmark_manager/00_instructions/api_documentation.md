# API Documentation - Twitter Bookmarks Manager

## **Overview**
This document provides a detailed reference for all API endpoints used in the **Twitter Bookmarks Manager**. These APIs facilitate **search, chat interactions, bookmark ingestion, and database management**. The backend is built with **Flask**, handling both REST API requests and UI rendering.

## **1. API Endpoints**

### **1️⃣ Search Endpoints**
#### **GET `/search`** - Search bookmarks using keywords, categories, or author-based filtering.
##### **Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | String | Search query (optional) |
| `user` | String | Filter by Twitter username (optional) |
| `categories[]` | List | List of categories to filter results (optional) |
| `limit` | Integer | Number of results to return (default: 1000) |

##### **Example Request:**
```http
GET /search?q=machine+learning&categories[]=AI & Machine Learning
```
##### **Response Format:**
```json
{
  "total_results": 15,
  "results": [
    {
      "id": "12345",
      "text": "Deep learning is revolutionizing AI.",
      "author": "@elonmusk",
      "categories": ["AI & Machine Learning"],
      "created_at": "2025-02-05 10:30:00"
    }
  ]
}
```

#### **GET `/recent`** - Retrieve the latest bookmarks.
##### **Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | Integer | Number of results to return (default: 1000) |

##### **Example Request:**
```http
GET /recent?limit=5
```
##### **Response Format:**
```json
{
  "total_results": 5,
  "results": [
    {
      "id": "67890",
      "text": "This AI tool automates coding tasks!",
      "author": "@techguru",
      "categories": ["Productivity & Tools"],
      "created_at": "2025-02-04 14:20:00"
    }
  ]
}
```

---

### **2️⃣ Chat & AI Endpoints**
#### **POST `/api/chat`** - AI-powered conversational search.
##### **Request Body:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | String | User's chat query |

##### **Example Request:**
```json
{
  "message": "What are my latest AI-related bookmarks?"
}
```
##### **Response Format:**
```json
{
  "response": "Here are your latest AI-related bookmarks...",
  "bookmarks_used": 3,
  "success": true,
  "context_used": 3,
  "metadata": {
    "timestamp": "2025-02-05T12:45:30Z",
    "bookmarks_referenced": 3
  }
}
```

---

### **3️⃣ Bookmark Management Endpoints**
#### **POST `/upload-bookmarks`** - Upload new bookmark data (JSON format).
##### **Request:**
- **Multipart form-data** file upload (`.json` file format only).

##### **Response Format:**
```json
{
  "message": "File processed successfully"
}
```

#### **POST `/update-database`** - Process bookmarks and update search database.
##### **Process Workflow:**
1. **Validates & processes uploaded JSON bookmarks.**
2. **Updates the SQL database** (via `update_bookmarks.py`).
3. **Assigns categories to bookmarks** (via `process_categories.py`).
4. **Updates the vector search system** (via `vector_store.py`).

##### **Example Response:**
```json
{
  "message": "Database updated successfully",
  "steps": [
    "File validated",
    "SQL database updated",
    "Categories assigned",
    "Vector store rebuilt"
  ]
}
```

---

### **4️⃣ PythonAnywhere-Specific Endpoints**
#### **GET `/api/status`** - Check database connection status and system health.
##### **Purpose:**
Verifies database connectivity and reports system status, particularly useful for debugging PythonAnywhere deployment issues.

##### **Response Format:**
```json
{
  "database_connection": "ok",
  "category_count": 15,
  "tweet_count": 1250,
  "categories": [
    {"id": 1, "name": "AI & Machine Learning"},
    {"id": 2, "name": "Business & Finance"}
  ]
}
```

#### **GET `/debug-database`** - Detailed database information for troubleshooting.
##### **Purpose:**
Provides deeper insights into database structure and connectivity, intended for admin use during deployment troubleshooting.

##### **Response Format:**
```json
{
  "status": "success",
  "database_connection": "ok",
  "category_count": 15,
  "categories": [...],
  "tweet_count": 1250
}
```

##### **Error Response:**
```json
{
  "error": "Database connection error",
  "details": "could not translate host name to address"
}
```

---

## **2. Backend Interactions & Dependencies**
Each API endpoint interacts with multiple system components:
| Endpoint | Module Interaction |
|----------|------------------|
| `/search` | `search.py` (keyword & vector-based search) |
| `/api/chat` | `engine.py` (AI-powered chat) |
| `/upload-bookmarks` | `update_bookmarks.py`, `process_categories.py` |
| `/update-database` | `vector_store.py`, `db.py` |
| `/api/status` | `db_pa.py` (PythonAnywhere specific) |

---

## **3. Authentication & Security**
Currently, the system **does not require authentication** for API requests. Future enhancements could include:
- **OAuth-based authentication** for user-based access.
- **Rate limiting & API keys** for controlled access.
- **Secure handling of user data** when Twitter API integration is added.

---

## **4. Environment-Specific API Behaviors**

### **Local Development Environment**
- Uses `server.py` as the main entry point
- SQLite database with file-based access
- ChromaDB for vector embeddings
- Simplified error handling for development purposes
- Adds Flask debug information in responses when `FLASK_DEBUG=true`

### **PythonAnywhere Production Environment**
- Uses `wsgi.py` and `api_server.py` as entry points
- PostgreSQL database with connection pooling
- Qdrant for vector embeddings
- Enhanced error handling with detailed logging
- Session tracking with unique session IDs for each operation
- Additional endpoints for system monitoring and troubleshooting

#### **PythonAnywhere-Specific Request Handling**
- **Batch Processing**: The `/update-database` endpoint processes bookmarks in batches, with the ability to pause and resume
- **Progress Tracking**: Operations track progress and can be resumed after interruption
- **Enhanced Logging**: Detailed logs with operation IDs for traceability
- **File Handling**: Improved file validation and storage with secure temporary directories
- **Status Monitoring**: Additional endpoints for checking system health and database connectivity

---

## **5. Future API Enhancements**
🚀 **Planned Improvements:**
- **Integration with Twitter API** for live bookmark updates.
- **Enhanced monitoring endpoints** for better production observability.
- **User authentication system** for personalized bookmark collections.
- **API versioning** to ensure backward compatibility.

**This document serves as the complete API reference for the system. Any new API additions should be documented here.** 🚀

## **4. PythonAnywhere-Specific API Configuration**

### **Environment-Specific Endpoints**
The PythonAnywhere deployment includes modified API endpoints that handle PostgreSQL and Qdrant integration:

#### **POST `/upload-bookmarks`** - Enhanced File Upload
- **Additional Validations**:
  ```json
  {
    "max_file_size": "16MB",
    "allowed_types": ["application/json"],
    "temp_storage": "temp_uploads/"
  }
  ```
- **Enhanced Response Format**:
  ```json
  {
    "message": "File uploaded successfully",
    "details": {
        "original_name": "bookmarks.json",
        "final_path": "/path/to/database/twitter_bookmarks.json",
        "backup_created": true,
        "timestamp": "2025-02-08T04:05:00Z"
    }
  }
  ```

#### **POST `/update-database`** - PostgreSQL Updates
- **Process Details**:
  ```json
  {
    "message": "Database updated successfully",
    "details": {
        "new_bookmarks": 50,
        "updated_bookmarks": 10,
        "errors": 0,
        "total_processed": 60,
        "unique_ids": 60
    }
  }
  ```

### **Error Handling**
PythonAnywhere-specific error responses include additional context:
```json
{
    "error": "Database operation failed",
    "details": {
        "operation": "bookmark_update",
        "stage": "vector_store_sync",
        "error_code": "QDRANT_SYNC_ERROR",
        "timestamp": "2025-02-08T04:05:00Z"
    },
    "traceback": "Detailed error trace"
}
```

### **Rate Limiting**
PythonAnywhere-specific rate limits:
- Upload endpoint: 10 requests/hour
- Database update: 5 requests/hour
- Search operations: 100 requests/minute

### **Logging**
Enhanced logging configuration:
- Location: `/home/username/logs/`
- Format: Detailed timestamps and request context
- Rotation: Daily with 7-day retention
- Error tracking: Integrated with PythonAnywhere's error console