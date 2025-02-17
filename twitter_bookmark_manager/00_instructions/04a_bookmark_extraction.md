# Automatic Bookmark Extraction - Twitter Bookmarks Manager

## **1. Overview**
This document provides a **detailed technical breakdown** of the **Automatic Bookmark Extraction System**. The goal is to allow users to retrieve their Twitter bookmarks **automatically across all browsers**, eliminating the need for manual JSON downloads. This system will:

- **Detect the user's browser** and select the appropriate method for extraction.
- **Use a custom Chrome extension** (if applicable) to extract and process bookmarks.
- **Support alternative methods** (JavaScript injection, Playwright/Selenium automation) for non-Chrome users.
- **Send extracted bookmark data to the backend** for processing and integration.

This documentation includes **full explanations, technical flow, and code extracted from the existing Chrome extension** to guide development.

---

## **2. System Architecture & Data Flow**

### **Step 1: Browser Detection**
- When a user **clicks "Retrieve Bookmarks"**, the system:
  - **Detects the user's browser** (`navigator.userAgent`).
  - **Selects the appropriate extraction method**:
    - **Chrome** → Uses **a custom extension** to extract bookmarks.
    - **Firefox/Edge/Safari** → Uses **a JavaScript-based injection method**.
    - **Backend Automation** (Selenium/Playwright) for **server-side extractions**.

### **Step 2: Bookmark Extraction Process**
1. **Navigates to `https://twitter.com/i/bookmarks`**.
2. **Fetches bookmarked tweets** (text, author, timestamp, media links).
3. **Iterates through all pages** to collect **full bookmark history**.
4. **Converts extracted bookmarks into a JSON format**.
5. **Sends JSON directly to the backend (`/api/upload-bookmarks`)** for processing.

### **Step 3: Backend Processing**
- Extracted bookmarks **are received via the API endpoint in `web/server.py`**.
- The system **validates, categorizes, and stores the bookmarks**.
- A copy of the raw JSON is stored in `database/json_history/` with format `twitter_bookmarks_YYYYMMDD.json`
- Bookmarks are **indexed in the vector database (ChromaDB)** for semantic search.
- The system maintains a historical record of all bookmark exports for:
  - Version comparison
  - Data recovery
  - Trend analysis
  - Backup purposes

### **Step 4: Data Retention & History**
- Each bookmark export is automatically archived in `database/json_history/`
- Files are named with the format `twitter_bookmarks_YYYYMMDD.json`
- The system maintains daily snapshots for historical tracking
- This enables:
  - Comparing bookmark changes over time
  - Recovering from failed imports
  - Analyzing bookmark trends
  - Maintaining data integrity

### **4. Final Integration with UI (`web/templates/base.html` Pop-up Behavior)**
The existing manual JSON upload flow will be adapted to support fully automated retrieval and processing.

#### **New User Flow:**
- User clicks "Upload Bookmarks" or "Update Database" in `web/templates/base.html`
- A pop-up launches, providing the user with a single button to:
  - Retrieve bookmarks automatically based on the chosen browser extraction method
  - Once bookmarks are extracted, they are automatically uploaded to `/api/upload-bookmarks`
  - The same pop-up shows real-time progress updates during the process
- The extracted JSON is used by existing scripts (`database/update_bookmarks.py`, `core/process_categories.py`) to update databases and categorize entries
- Error logging will be implemented on the server to track issues in any step of the process

This ensures that users only need to click one button to retrieve and update their bookmarks, making the system fully automated.

🚀 This documentation ensures Cursor (IDE AI Assistant) understands the full pipeline and implementation details!

---

## **3. Implementation Strategies by Browser**

### **1️⃣ Chrome: Using a Custom Extension**
- Chrome extensions **run scripts inside the browser** and can interact with `twitter.com` directly.
- The extracted extension code can be **modified** to:
  - Automatically **fetch bookmarks when triggered**.
  - **Store the JSON locally** instead of downloading.
  - **Send the JSON directly to the backend**.

#### **Code Snippet (Modified for Our Use Case)**
```javascript
// Content script to extract bookmarks automatically
(async function() {
    let bookmarks = [];
    let cursor = null;
    do {
        let response = await fetch(`https://twitter.com/i/api/graphql/YOUR_QUERY_ID/BookmarksTimeline?cursor=${cursor}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer YOUR_ACCESS_TOKEN`,
                'x-csrf-token': 'YOUR_CSRF_TOKEN',
                'Content-Type': 'application/json'
            },
        });
        let data = await response.json();
        bookmarks.push(...data.data.bookmark_timeline_v2.timeline.instructions[0].entries);
        cursor = data.data.bookmark_timeline_v2.timeline.instructions[0].entries.slice(-1)[0]?.content.value;
    } while (cursor);
    
    // Send extracted bookmarks to backend
    fetch('https://your-backend.com/upload-bookmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarks })
    });
})();

### **2️⃣ Firefox, Edge, Safari: JavaScript-Based Extraction**
Some browsers do not support Chrome extensions.
Instead, we can inject JavaScript into the browser when the user visits Twitter.
This method works only when the user is logged into Twitter.

#### **JavaScript Injection Approach**
```javascript

// Injects script into Twitter's bookmark page
(async function() {
    if (window.location.href.includes("twitter.com/i/bookmarks")) {
        let tweets = [];
        document.querySelectorAll("article").forEach(tweet => {
            tweets.push({
                text: tweet.innerText,
                author: tweet.querySelector("a[href*='/status/']")?.innerText,
                link: tweet.querySelector("a[href*='/status/']")?.href,
                timestamp: new Date().toISOString()
            });
        });
        
        // Send extracted bookmarks to backend
        fetch('https://your-backend.com/upload-bookmarks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bookmarks: tweets })
        });
    }
})();