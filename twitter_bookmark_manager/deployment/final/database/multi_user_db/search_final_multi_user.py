"""
Multi-user version of BookmarkSearch for PythonAnywhere.
Adds user filtering to all search operations.
"""

import logging
import os
import psycopg2
from psycopg2 import sql
from datetime import datetime
import random
import json
import time
from sqlalchemy import text
import traceback
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search_pa_multi_user")

# Import user context
from auth.user_context_final import with_user_context

class BookmarkSearchMultiUser:
    """
    Bookmark search implementation for PythonAnywhere with multi-user support.
    Adds user_id filtering to all database queries.
    """

    def __init__(self, conn, user_id=1):
        self.conn = conn
        self.user_id = user_id  # Default to system user if not specified
        logger.info(f"Created BookmarkSearchMultiUser with user_id: {self.user_id}")

    def _get_cursor(self):
        """
        Get a cursor from the connection, handling both psycopg2 and SQLAlchemy connections.
        Returns a cursor object and a flag indicating if it's SQLAlchemy.
        """
        # Check if this is an SQLAlchemy connection
        if hasattr(self.conn, 'execute'):
            logger.info("Using SQLAlchemy connection")
            return None, True
        else:
            # This is a psycopg2 connection
            logger.info("Using psycopg2 connection")
            return self.conn.cursor(), False

    @with_user_context
    def search(self, query=None, user=None, category_ids=None, limit=100, user_id=1):
        """
        Search for bookmarks with multi-user support.
        Filters bookmarks by user_id in addition to other criteria.
        """
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        # Ensure user_id is an integer
        try:
            self.user_id = int(self.user_id)
        except (TypeError, ValueError):
            logger.error(f"Invalid user_id: {self.user_id}, type: {type(self.user_id)}")
            self.user_id = 1  # Default to system user
        
        logger.info(f"SEARCH DEBUG - Using user_id: {self.user_id} (type: {type(self.user_id)})")
        
        cursor, is_sqlalchemy = self._get_cursor()
        results = []
        
        try:
            # Start building the query
            sql_query = """
            SELECT b.id, b.bookmark_id, b.text, b.author, b.created_at, b.author_id 
            FROM bookmarks b 
            WHERE b.user_id = %s
            """
            params = [self.user_id]
            
            # Add text search condition if query is provided
            if query and query.strip():
                sql_query += " AND b.text ILIKE %s"
                params.append(f"%{query}%")
                
            # Add user filter if provided
            if user and user.strip():
                sql_query += " AND b.author ILIKE %s"
                params.append(f"%{user}%")
                
            # Add category filter if provided
            if category_ids and len(category_ids) > 0:
                placeholder = ','.join(['%s'] * len(category_ids))
                sql_query += f"""
                AND b.id IN (
                    SELECT bookmark_id 
                    FROM bookmark_categories 
                    WHERE category_id IN ({placeholder}))
                """
                params.extend(category_ids)
                
            # Add order and limit
            sql_query += " ORDER BY b.created_at DESC LIMIT %s"
            params.append(limit)
            
            # CRITICAL DEBUGGING: Log full SQL query and parameters
            param_str = ', '.join([f"'{p}'" if isinstance(p, str) else str(p) for p in params])
            logger.info(f"SEARCH DEBUG - Executing exact SQL query: {sql_query} with params: [{param_str}]")
            logger.info(f"SEARCH DEBUG - User ID: {self.user_id}, Type: {type(self.user_id)}")
            
            # Direct debugging query to check bookmarks table
            if is_sqlalchemy:
                check_sql = text("SELECT COUNT(*) FROM bookmarks")
                result = self.conn.execute(check_sql)
                count = result.scalar()
                logger.info(f"SEARCH DEBUG - Total bookmarks in database: {count}")
                
                check_sql = text("SELECT COUNT(*) FROM bookmarks WHERE user_id = :user_id")
                result = self.conn.execute(check_sql, {"user_id": self.user_id})
                count = result.scalar()
                logger.info(f"SEARCH DEBUG - Bookmarks for user {self.user_id}: {count}")
            else:
                # For psycopg2, use the cursor directly
                debug_cursor = self.conn.cursor()
                debug_cursor.execute("SELECT COUNT(*) FROM bookmarks")
                count = debug_cursor.fetchone()[0]
                logger.info(f"SEARCH DEBUG - Total bookmarks in database: {count}")
                
                debug_cursor.execute("SELECT COUNT(*) FROM bookmarks WHERE user_id = %s", (self.user_id,))
                count = debug_cursor.fetchone()[0]
                logger.info(f"SEARCH DEBUG - Bookmarks for user {self.user_id}: {count}")
            
            # Execute the query
            if is_sqlalchemy:
                # For SQLAlchemy, use text() to create a SQL expression
                stmt = text(sql_query)
                result_proxy = self.conn.execute(stmt, params)
                rows = result_proxy.fetchall()
            else:
                # For psycopg2, use the cursor directly
                cursor.execute(sql_query, params)
                rows = cursor.fetchall()
            
            logger.info(f"SEARCH DEBUG - Search returned {len(rows)} results")
            
            # Debug raw row data
            if len(rows) > 0:
                logger.info(f"SEARCH DEBUG - First row sample: {rows[0]}")
            else:
                logger.info("SEARCH DEBUG - No rows returned")
            
            # Process results
            for row in rows:
                bookmark = {
                    'id': row[0],
                    'bookmark_id': row[1],  # Include the Twitter bookmark_id
                    'text': row[2],
                    'author': row[3],
                    'author_username': row[3].replace('@', '') if row[3] else '',
                    'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[4], 'strftime') else row[4],
                    'author_id': row[5]
                }
                
                # Get categories for this bookmark
                categories = self._get_bookmark_categories(bookmark['id'], is_sqlalchemy, cursor)
                bookmark['categories'] = [cat['name'] for cat in categories]
                results.append(bookmark)
                
        except Exception as e:
            logger.error(f"Error in search: {e}")
            logger.error(traceback.format_exc())
            
        return results
        
    def _get_bookmark_categories(self, bookmark_id, is_sqlalchemy, cursor=None):
        """Get categories for a bookmark, handling both connection types"""
        categories = []
        cat_query = """
        SELECT c.id, c.name 
        FROM categories c 
        JOIN bookmark_categories bc ON c.id = bc.category_id 
        WHERE bc.bookmark_id = %s AND c.user_id = %s
        """
        
        try:
            if is_sqlalchemy:
                stmt = text(cat_query)
                result_proxy = self.conn.execute(stmt, [bookmark_id, self.user_id])
                cat_rows = result_proxy.fetchall()
            else:
                cursor.execute(cat_query, (bookmark_id, self.user_id))
                cat_rows = cursor.fetchall()
                
            for cat_row in cat_rows:
                categories.append({
                    'id': cat_row[0],
                    'name': cat_row[1]
                })
        except Exception as e:
            logger.error(f"Error getting categories for bookmark {bookmark_id}: {e}")
            
        return categories

    @with_user_context
    def get_recent(self, limit=5, user_id=1):
        """Get recent bookmarks for a user"""
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        cursor, is_sqlalchemy = self._get_cursor()
        results = []
        
        try:
            if is_sqlalchemy:
                query = text("""
                SELECT b.id, b.text, b.url, b.author, b.created_at, b.author_id
                FROM bookmarks b
                WHERE b.user_id = :user_id
                ORDER BY b.created_at DESC
                LIMIT :limit
                """)
                result = self.conn.execute(query, {"user_id": self.user_id, "limit": limit})
                rows = result.fetchall()
            else:
                query = """
                SELECT b.id, b.text, b.url, b.author, b.created_at, b.author_id
                FROM bookmarks b
                WHERE b.user_id = %s
                ORDER BY b.created_at DESC
                LIMIT %s
                """
                cursor.execute(query, (self.user_id, limit))
                rows = cursor.fetchall()
                
            for row in rows:
                bookmark = {
                    'id': row[0],
                    'text': row[1],
                    'url': row[2],
                    'author': row[3],
                    'author_username': row[3].replace('@', '') if row[3] else '',
                    'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[4], 'strftime') else row[4],
                    'author_id': row[5]
                }
                
                # Get categories for this bookmark
                categories = self._get_bookmark_categories(bookmark['id'], is_sqlalchemy, cursor)
                bookmark['categories'] = [cat['name'] for cat in categories]
                results.append(bookmark)
                
        except Exception as e:
            logger.error(f"Error getting recent bookmarks: {e}")
            logger.error(traceback.format_exc())
            # Return empty list on error
            
        return results
        
    @with_user_context
    def get_recent_bookmarks(self, limit=5, user_id=1):
        """Get recent bookmarks formatted for the template display"""
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        bookmarks = self.get_recent(limit=limit, user_id=user_id)
        
        # Return the bookmarks directly as they are already formatted for template display
        return bookmarks
            
    @with_user_context
    def get_bookmark_count(self, user_id=1):
        """Get the total number of bookmarks for a user."""
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        cursor, is_sqlalchemy = self._get_cursor()
        
        try:
            if is_sqlalchemy:
                query = text("SELECT COUNT(*) FROM bookmarks WHERE user_id = :user_id")
                result = self.conn.execute(query, {"user_id": self.user_id})
                count = result.scalar()
            else:
                query = "SELECT COUNT(*) FROM bookmarks WHERE user_id = %s"
                cursor.execute(query, (self.user_id,))
                count = cursor.fetchone()[0]
                
            return count
            
        except Exception as e:
            logger.error(f"Error getting bookmark count: {e}")
            logger.error(traceback.format_exc())
            return 0
            
    @with_user_context
    def get_categories(self, user_id=1):
        """Get all categories for a user with bookmark counts."""
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        # Ensure user_id is an integer
        try:
            self.user_id = int(self.user_id)
        except (TypeError, ValueError):
            logger.error(f"Invalid user_id: {self.user_id}, type: {type(self.user_id)}")
            self.user_id = 1  # Default to system user
            
        logger.info(f"CATEGORIES DEBUG - Using user_id: {self.user_id} (type: {type(self.user_id)})")
        
        cursor, is_sqlalchemy = self._get_cursor()
        results = []
        
        try:
            if is_sqlalchemy:
                query = text("""
                SELECT c.id, c.name, COUNT(bc.bookmark_id) as count 
                FROM categories c 
                LEFT JOIN bookmark_categories bc ON c.id = bc.category_id 
                WHERE c.user_id = :user_id 
                GROUP BY c.id, c.name 
                ORDER BY c.name
                """)
                
                logger.info(f"Executing get_categories query for user_id: {self.user_id}")
                result_proxy = self.conn.execute(query, {"user_id": self.user_id})
                rows = result_proxy.fetchall()
            else:
                query = """
                SELECT c.id, c.name, COUNT(bc.bookmark_id) as count 
                FROM categories c 
                LEFT JOIN bookmark_categories bc ON c.id = bc.category_id 
                WHERE c.user_id = %s 
                GROUP BY c.id, c.name 
                ORDER BY c.name
                """
                cursor.execute(query, (self.user_id,))
                rows = cursor.fetchall()
            
            logger.info(f"get_categories returned {len(rows)} categories")
            
            for row in rows:
                # Calculate percentage if there are bookmarks
                bookmark_count = self.get_bookmark_count(user_id)
                percentage = 0
                if bookmark_count > 0 and row[2] > 0:
                    percentage = int((row[2] / bookmark_count) * 100)
                
                category = {
                    'id': row[0],
                    'name': row[1],
                    'count': row[2],
                    'percentage': percentage
                }
                results.append(category)
                
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            logger.error(traceback.format_exc())
            
        return results
            
    @with_user_context
    def update_bookmark_categories(self, bookmark_id, category_ids, user_id=1):
        """Update the categories for a bookmark, ensuring user ownership."""
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        cursor, is_sqlalchemy = self._get_cursor()
        
        try:
            # First, verify that the bookmark belongs to the user
            if is_sqlalchemy:
                query = text("SELECT id FROM bookmarks WHERE id = :bookmark_id AND user_id = :user_id")
                result = self.conn.execute(query, {"bookmark_id": bookmark_id, "user_id": self.user_id})
                if not result.fetchone():
                    logger.warning(f"Bookmark {bookmark_id} does not belong to user {self.user_id}")
                    return {"error": "Access denied or bookmark not found"} 
            else:
                query = "SELECT id FROM bookmarks WHERE id = %s AND user_id = %s"
                cursor.execute(query, (bookmark_id, self.user_id))
                if not cursor.fetchone():
                    logger.warning(f"Bookmark {bookmark_id} does not belong to user {self.user_id}")
                    return {"error": "Access denied or bookmark not found"}
            
            # Clear existing categories for this bookmark
            if is_sqlalchemy:
                delete_query = text("DELETE FROM bookmark_categories WHERE bookmark_id = :bookmark_id")
                self.conn.execute(delete_query, {"bookmark_id": bookmark_id})
            else:
                delete_query = "DELETE FROM bookmark_categories WHERE bookmark_id = %s"
                cursor.execute(delete_query, (bookmark_id,))
            
            # Insert new categories if provided
            if category_ids and len(category_ids) > 0:
                # Verify that the categories belong to the user
                if is_sqlalchemy:
                    # Build a query with SQLAlchemy that can handle a list of category IDs
                    category_list = ",".join([str(cat_id) for cat_id in category_ids])
                    verify_query = text(f"""
                    SELECT id FROM categories 
                    WHERE id IN ({category_list}) AND user_id = :user_id
                    """)
                    result = self.conn.execute(verify_query, {"user_id": self.user_id})
                    valid_category_ids = [row[0] for row in result.fetchall()]
                else:
                    placeholders = ','.join(['%s'] * len(category_ids))
                    verify_query = f"""
                    SELECT id FROM categories 
                    WHERE id IN ({placeholders}) AND user_id = %s
                    """
                    params = list(category_ids) + [self.user_id]
                    cursor.execute(verify_query, params)
                    valid_category_ids = [row[0] for row in cursor.fetchall()]
                
                # Insert verified categories
                if valid_category_ids:
                    for cat_id in valid_category_ids:
                        if is_sqlalchemy:
                            insert_query = text("INSERT INTO bookmark_categories (bookmark_id, category_id) VALUES (:bookmark_id, :cat_id)")
                            self.conn.execute(insert_query, {"bookmark_id": bookmark_id, "cat_id": cat_id})
                        else:
                            cursor.execute(
                                "INSERT INTO bookmark_categories (bookmark_id, category_id) VALUES (%s, %s)",
                                (bookmark_id, cat_id)
                            )
            
            return {"success": True, "message": "Categories updated"}
            
        except Exception as e:
            logger.error(f"Error updating bookmark categories: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)} 

    def _get_categories_for_bookmark(self, bookmark_id, is_sqlalchemy=False, cursor=None):
        """Get categories for a bookmark"""
        categories = []
        cat_query = """
        SELECT c.id, c.name 
        FROM categories c 
        JOIN bookmark_categories bc ON c.id = bc.category_id 
        WHERE bc.bookmark_id = :bookmark_id AND c.user_id = :user_id
        """
        
        try:
            if is_sqlalchemy:
                stmt = text(cat_query)
                result_proxy = self.conn.execute(stmt, {"bookmark_id": bookmark_id, "user_id": self.user_id})
                cat_rows = result_proxy.fetchall()
            else:
                cursor.execute(cat_query, (bookmark_id, self.user_id))
                cat_rows = cursor.fetchall()
                
            for cat_row in cat_rows:
                categories.append({
                    'id': cat_row[0],
                    'name': cat_row[1]
                })
        except Exception as e:
            logger.error(f"Error getting categories for bookmark {bookmark_id}: {e}")
            
        return categories

    def get_categories(self):
        """Get all categories"""
        results = []
        try:
            stmt = text("""
            SELECT id, name, description
            FROM categories
            WHERE user_id = :user_id
            ORDER BY name
            """)
            result = self.conn.execute(stmt, {"user_id": self.user_id})
            rows = result.fetchall()
            
            for row in rows:
                category = {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2]
                }
                results.append(category)
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            logger.error(traceback.format_exc())
            
        return results

    def get_bookmarks_by_category(self, category_name, limit=50):
        """Get bookmarks for a specific category"""
        results = []
        try:
            stmt = text("""
            SELECT b.id, b.bookmark_id, b.text, b.author, b.created_at, b.author_id
            FROM bookmarks b
            JOIN bookmark_categories bc ON b.id = bc.bookmark_id
            JOIN categories c ON bc.category_id = c.id
            WHERE c.name = :category_name AND b.user_id = :user_id
            ORDER BY b.created_at DESC
            LIMIT :limit
            """)
            result_proxy = self.conn.execute(stmt, {"category_name": category_name, "user_id": self.user_id, "limit": limit})
            rows = result_proxy.fetchall()
            
            for row in rows:
                bookmark = {
                    'id': row[0],
                    'bookmark_id': row[1],
                    'text': row[2],
                    'author': row[3],
                    'author_username': row[3].replace('@', '') if row[3] else '',
                    'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[4], 'strftime') else row[4],
                    'author_id': row[5]
                }
                results.append(bookmark)
        except Exception as e:
            logger.error(f"Error getting bookmarks by category: {e}")
            logger.error(traceback.format_exc())
            
        return results

    def get_bookmark(self, bookmark_id):
        """Get a bookmark by ID"""
        try:
            stmt = text("""
            SELECT id, bookmark_id, text, author, created_at, author_id
            FROM bookmarks
            WHERE id = :bookmark_id AND user_id = :user_id
            """)
            result = self.conn.execute(stmt, {"bookmark_id": bookmark_id, "user_id": self.user_id})
            row = result.fetchone()
            
            if row:
                bookmark = {
                    'id': row[0],
                    'bookmark_id': row[1],
                    'text': row[2],
                    'author': row[3],
                    'author_username': row[3].replace('@', '') if row[3] else '',
                    'created_at': row[4].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row[4], 'strftime') else row[4],
                    'author_id': row[5]
                }
                return bookmark
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting bookmark: {e}")
            logger.error(traceback.format_exc())
            return None

    def delete_bookmark(self, bookmark_id):
        """Delete a bookmark"""
        try:
            # First delete categories
            stmt = text("DELETE FROM bookmark_categories WHERE bookmark_id = :bookmark_id")
            self.conn.execute(stmt, {"bookmark_id": bookmark_id})
            
            # Then delete the bookmark itself
            stmt = text("DELETE FROM bookmarks WHERE id = :bookmark_id")
            self.conn.execute(stmt, {"bookmark_id": bookmark_id})
            
            return {"success": True, "message": "Bookmark deleted"}
        except Exception as e:
            logger.error(f"Error deleting bookmark: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def add_bookmark_to_category(self, bookmark_id, cat_id):
        """Add a bookmark to a category"""
        try:
            stmt = text("""
            INSERT INTO bookmark_categories (bookmark_id, category_id)
            VALUES (:bookmark_id, :cat_id)
            ON CONFLICT DO NOTHING
            """)
            self.conn.execute(stmt, {"bookmark_id": bookmark_id, "cat_id": cat_id})
            
            return {"success": True, "message": "Bookmark added to category"}
        except Exception as e:
            logger.error(f"Error adding bookmark to category: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)} 

    def get_all_bookmarks_for_user(self, user_id):
        """
        Retrieve all bookmarks for a specific user from the database.
        
        Args:
            user_id (int): The user ID to fetch bookmarks for
            
        Returns:
            list: List of bookmark dictionaries with keys id, text, tweet_content, author, created_at
        """
        try:
            logger.info(f"Fetching all bookmarks for user {user_id}")
            
            # Connect to database
            conn = self.get_db_connection()
            if not conn:
                logger.error(f"Failed to connect to database when fetching bookmarks for user {user_id}")
                return []
            
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Query to get all bookmarks for user
            query = """
                SELECT bookmark_id, text, tweet_content, author, created_at
                FROM bookmarks
                WHERE user_id = %s
            """
            
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            bookmarks = []
            for row in results:
                # Combine text and tweet_content for better search
                text = row['text'] or ''
                tweet_content = row['tweet_content'] or ''
                
                bookmark = {
                    'id': row['bookmark_id'],
                    'text': text,
                    'tweet_content': tweet_content,
                    'author': row['author'] or '',
                    'created_at': row['created_at'] or ''
                }
                bookmarks.append(bookmark)
            
            logger.info(f"Found {len(bookmarks)} bookmarks for user {user_id}")
            return bookmarks
            
        except Exception as e:
            logger.error(f"Error fetching bookmarks for user {user_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return [] 