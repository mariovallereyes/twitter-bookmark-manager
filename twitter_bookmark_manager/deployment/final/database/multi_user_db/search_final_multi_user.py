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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search_pa_multi_user")

# Import user context
from deployment.final.auth.user_context_final import with_user_context

class BookmarkSearchMultiUser:
    """
    Bookmark search implementation for PythonAnywhere with multi-user support.
    Adds user_id filtering to all database queries.
    """

    def __init__(self, conn, user_id=1):
        self.conn = conn
        self.user_id = user_id  # Default to system user if not specified
        logger.info(f"Created BookmarkSearchMultiUser with user_id: {self.user_id}")

    @with_user_context
    def search(self, query=None, user=None, category_ids=None, limit=100, user_id=1):
        """
        Search for bookmarks with multi-user support.
        Filters bookmarks by user_id in addition to other criteria.
        """
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        cursor = self.conn.cursor()
        results = []
        
        try:
            # Start building the query
            sql_query = """
            SELECT b.id, b.text, b.author, b.created_at 
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
                sql_query += """
                AND b.id IN (
                    SELECT bookmark_id 
                    FROM bookmark_categories 
                    WHERE category_id IN ({}))
                """.format(','.join(['%s'] * len(category_ids)))
                params.extend(category_ids)
                
            # Add order and limit
            sql_query += " ORDER BY b.created_at DESC LIMIT %s"
            params.append(limit)
            
            # Execute the query
            cursor.execute(sql_query, params)
            
            # Process results
            for row in cursor.fetchall():
                bookmark = {
                    'id': row[0],
                    'text': row[1],
                    'author': row[2],
                    'created_at': row[3].strftime('%Y-%m-%d %H:%M:%S') if row[3] else None
                }
                
                # Get categories for this bookmark
                cursor.execute("""
                SELECT c.id, c.name 
                FROM categories c 
                JOIN bookmark_categories bc ON c.id = bc.category_id 
                WHERE bc.bookmark_id = %s AND c.user_id = %s
                """, (bookmark['id'], self.user_id))
                
                categories = []
                for cat_row in cursor.fetchall():
                    categories.append({
                        'id': cat_row[0],
                        'name': cat_row[1]
                    })
                
                bookmark['categories'] = categories
                results.append(bookmark)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
            
    @with_user_context
    def get_recent(self, limit=10, user_id=1):
        """
        Get recent bookmarks with multi-user support.
        Filters by user_id.
        """
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        cursor = self.conn.cursor()
        results = []
        
        try:
            # Get recent bookmarks for this user
            cursor.execute("""
            SELECT b.id, b.text, b.author, b.created_at 
            FROM bookmarks b 
            WHERE b.user_id = %s
            ORDER BY b.created_at DESC 
            LIMIT %s
            """, (self.user_id, limit))
            
            # Process results
            for row in cursor.fetchall():
                bookmark = {
                    'id': row[0],
                    'text': row[1],
                    'author': row[2],
                    'created_at': row[3].strftime('%Y-%m-%d %H:%M:%S') if row[3] else None
                }
                
                # Get categories for this bookmark
                cursor.execute("""
                SELECT c.id, c.name 
                FROM categories c 
                JOIN bookmark_categories bc ON c.id = bc.category_id 
                WHERE bc.bookmark_id = %s AND c.user_id = %s
                """, (bookmark['id'], self.user_id))
                
                categories = []
                for cat_row in cursor.fetchall():
                    categories.append({
                        'id': cat_row[0],
                        'name': cat_row[1]
                    })
                
                bookmark['categories'] = categories
                results.append(bookmark)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in get_recent: {e}")
            return []
            
    @with_user_context
    def get_bookmark_count(self, user_id=1):
        """Get the total number of bookmarks for a user"""
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM bookmarks WHERE user_id = %s", (self.user_id,))
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"Error in get_bookmark_count: {e}")
            return 0
            
    @with_user_context
    def get_categories(self, user_id=1):
        """Get all categories for a user"""
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        cursor = self.conn.cursor()
        categories = []
        
        try:
            cursor.execute("""
            SELECT c.id, c.name, COUNT(bc.bookmark_id) as count
            FROM categories c
            LEFT JOIN bookmark_categories bc ON c.id = bc.category_id
            WHERE c.user_id = %s
            GROUP BY c.id, c.name
            ORDER BY count DESC, c.name
            """, (self.user_id,))
            
            for row in cursor.fetchall():
                categories.append({
                    'id': row[0],
                    'name': row[1],
                    'count': row[2]
                })
                
            return categories
            
        except Exception as e:
            logger.error(f"Error in get_categories: {e}")
            return []
            
    @with_user_context
    def update_bookmark_categories(self, bookmark_id, category_ids, user_id=1):
        """
        Update categories for a bookmark.
        Ensures the bookmark belongs to the current user.
        """
        # Use the provided user_id or default from the class
        self.user_id = user_id
        
        cursor = self.conn.cursor()
        try:
            # First verify this bookmark belongs to the user
            cursor.execute(
                "SELECT COUNT(*) FROM bookmarks WHERE id = %s AND user_id = %s", 
                (bookmark_id, self.user_id)
            )
            
            if cursor.fetchone()[0] == 0:
                logger.warning(f"Attempted to update categories for bookmark {bookmark_id} that doesn't belong to user {self.user_id}")
                return False
            
            # Delete existing categories
            cursor.execute(
                "DELETE FROM bookmark_categories WHERE bookmark_id = %s", 
                (bookmark_id,)
            )
            
            # Add new categories
            if category_ids and len(category_ids) > 0:
                # Verify all categories belong to this user
                placeholder = ','.join(['%s'] * len(category_ids))
                cursor.execute(
                    f"SELECT COUNT(*) FROM categories WHERE id IN ({placeholder}) AND user_id = %s",
                    category_ids + [self.user_id]
                )
                
                if cursor.fetchone()[0] != len(category_ids):
                    logger.warning(f"Attempted to use categories that don't belong to user {self.user_id}")
                    self.conn.rollback()
                    return False
                
                # Insert new categories
                for category_id in category_ids:
                    cursor.execute(
                        "INSERT INTO bookmark_categories (bookmark_id, category_id) VALUES (%s, %s)",
                        (bookmark_id, category_id)
                    )
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error in update_bookmark_categories: {e}")
            self.conn.rollback()
            return False 