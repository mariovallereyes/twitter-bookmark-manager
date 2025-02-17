# User Flow - Twitter Bookmarks Manager

## **1. Initial Setup - User Prepares Bookmark Data**
- The user **exports bookmarks from Twitter** as a JSON file.
- They **upload the JSON file** via the web interface.
- The system **validates and parses** the JSON structure.
- A **backup of the existing database** is created before processing new bookmarks.

## **2. Bookmark Ingestion - System Processes the Data**
- The uploaded JSON file is **compared with existing records** to detect new bookmarks.
- **New bookmarks** are **added to the database** while duplicates are ignored.
- The system **extracts metadata**, including:

  - Tweet text
  - Author information
  - Timestamps
  - Media links
- **AI categorization** is applied to classify bookmarks into different topics.
- **Vector embeddings** are generated for each bookmark and stored in **ChromaDB** for future **semantic search**.

## **3. User Searches for Bookmarks**
- Users can search bookmarks in multiple ways:
  - **Keyword-based text search** (SQL search).
  - **Semantic search** (AI-powered vector retrieval).
  - **Category-based filtering** (e.g., "Show me all AI-related bookmarks").
  - **Author-based search** (e.g., "Find tweets by Elon Musk").
- The system **ranks results** based on relevance.
- The user **views detailed bookmark information** (tweet content, date, author, media).

## **4. AI-Powered Chat Interaction**
- The user can **ask questions** about their bookmarks (e.g., "Show me tweets about AI ethics").
- The system:
  - **Retrieves relevant bookmarks** using semantic search.
  - **Generates a response** using AI (Mistral-7B + RAG).
  - **Maintains chat history** for context-aware conversations.
- The user can **refine their queries** or continue interacting conversationally.

## **5. Database Update & Maintenance**
- The user can **upload new bookmark JSON files** periodically.
- The system **automatically processes new bookmarks** while preserving past data.
- Users can **manually trigger a vector store rebuild** (if needed).
- Logs are **generated for debugging** in case of failures.

## **6. Frontend User Interactions**
- **Search Page (`web/templates/index.html`):** Users type queries, filter by category, or search by author.
- **Upload Modal (`web/templates/base.html`):** Users select a new JSON file for ingestion.
- **Progress Tracking UI (`web/static/js/progress.js`):** Users see real-time updates while bookmarks are processed.
- **Chat Interface (`web/templates/chat.html`):** Users ask AI-powered queries to find bookmarks.

## **Future Enhancements & Adjustments**
This document will be updated as the project evolves and as more details about interactions and backend processes become clearer.

## **Next Steps**
- Validate exact interactions in `web/templates/base.html` and frontend scripts in `web/static/js/`.
- Confirm the ingestion process in `database/update_bookmarks.py`.
- Determine how past bookmarks are managed and how duplicates are handled.
- Validate real-time UI feedback mechanisms during ingestion and search.

---
**This document will be continuously updated based on project development.**
