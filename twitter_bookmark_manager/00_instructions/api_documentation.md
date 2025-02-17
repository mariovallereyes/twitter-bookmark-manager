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

## **2. Backend Interactions & Dependencies**
Each API endpoint interacts with multiple system components:
| Endpoint | Module Interaction |
|----------|------------------|
| `/search` | `search.py` (keyword & vector-based search) |
| `/api/chat` | `engine.py` (AI-powered chat) |
| `/upload-bookmarks` | `update_bookmarks.py`, `process_categories.py` |
| `/update-database` | `vector_store.py`, `db.py` |

---

## **3. Authentication & Security**
Currently, the system **does not require authentication** for API requests. Future enhancements could include:
- **OAuth-based authentication** for user-based access.
- **Rate limiting & API keys** for controlled access.
- **Secure handling of user data** when Twitter API integration is added.

---

## **4. Future API Enhancements**
🚀 **Planned Improvements:**
- **Integration with Twitter API** for live bookmark updates.
- **WebSocket support** for real-time chat updates.
- **User-based authentication & private bookmark collections.**

**This document serves as the complete API reference for the system. Any new API additions should be documented here.** 🚀