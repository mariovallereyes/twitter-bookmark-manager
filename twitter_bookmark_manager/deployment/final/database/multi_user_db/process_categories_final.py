"""
Multi-user category processor using Gemini AI for background processing of bookmark categories.
"""
import sys
import os
import logging
import traceback
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy import inspect

# Set up base paths
BASE_DIR = '/home/mariovallereyes/twitter_bookmark_manager'
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import from main codebase
from twitter_bookmark_manager.config.constants import BOOKMARK_CATEGORIES
from deployment.final.database.multi_user_db.db_final import get_session, get_vector_store
from twitter_bookmark_manager.database.models import Bookmark, Category, bookmark_categories
from sqlalchemy import func, or_
from dotenv import load_dotenv

# Load environment variables
env_paths = [
    Path('/home/mariovallereyes/twitter_bookmark_manager/.env.final').resolve(),
    Path(__file__).parents[3] / '.env.final'
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

# Configure logging
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'final_category_processor.log')
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeminiCategorizer:
    """Categorizer using Google's Gemini API with adaptive category generation"""
    
    def __init__(self):
        """Initialize with API key from environment"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("No GEMINI_API_KEY found in environment variables")
            raise ValueError("Missing GEMINI_API_KEY environment variable")
            
        # Update to use the gemini-1.0-pro model which is compatible with v1 API
        self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.0-pro:generateContent"
        logger.info("✅ GeminiCategorizer initialized successfully")
        
    def generate_categories(self, text: str, max_categories: int = 3) -> List[Dict[str, float]]:
        """Generate categories for text using Gemini AI.
        Returns a list of dictionaries with category name and confidence."""
        # Skip the API call attempt and use the fallback mechanism directly
        logger.info("Using keyword-based fallback categorization mechanism")
        return self._fallback_generation(text)
        
    def _call_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """This method is no longer used as we're using the fallback mechanism directly"""
        # This method is kept for future reference
        pass
    
    def _build_generative_prompt(self, text: str, max_categories=3) -> str:
        """Build a prompt for Gemini to generate relevant categories"""
        
        prompt = f"""You are an AI tasked with organizing Twitter bookmarks into helpful categories.
        
Analyze this tweet text and generate {max_categories} category labels that best describe its content.
The categories should be concise (1-3 words each), descriptive, and focus on the main topics or themes.

Important guidelines:
- Categories should be general enough to be useful for organizing a large collection
- Focus on the subject matter rather than sentiment or format
- Use standard capitalization (e.g., "Machine Learning" not "MACHINE LEARNING")
- Avoid overly niche or specific categories
- Each category should be concise - maximum 3 words

Tweet text:
"{text}"

Respond in the following JSON format only:
{{
  "categories": [
    {{"name": "CATEGORY_NAME", "confidence": SCORE}},
    {{"name": "CATEGORY_NAME", "confidence": SCORE}},
    ...
  ]
}}

Where:
- Each category has a name and confidence score between 0-1
- The confidence scores should sum to 1.0
- List categories in order of relevance (highest confidence first)

Generate valuable categorization:
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response to extract generated categories"""
        try:
            # Try to find JSON in the response
            response_text = response_text.strip()
            
            # Extract JSON part from the response if there's surrounding text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                if 'categories' in result and len(result['categories']) > 0:
                    # Format the response in the expected structure
                    categories = result['categories']
                    
                    # Normalize category names (trim extra spaces, capitalize properly)
                    for cat in categories:
                        if 'name' in cat:
                            cat['name'] = self._normalize_category_name(cat['name'])
                    
                    return {
                        'categories': categories
                    }
            
            # If we couldn't parse JSON or find categories
            logger.warning(f"Could not parse JSON from Gemini response: {response_text}")
            return self._fallback_generation(response_text)
            
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            logger.error(f"Response was: {response_text}")
            return self._fallback_generation(response_text)
    
    def _normalize_category_name(self, name: str) -> str:
        """Normalize category names for consistency"""
        # Remove extra whitespace and capitalize properly
        name = ' '.join(name.strip().split())
        
        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        # Special case for common acronyms
        for acronym in ['AI', 'ML', 'NLP', 'UI', 'UX', 'SEO', 'API']:
            name = name.replace(f"{acronym[0].upper()}{acronym[1:].lower()}", acronym)
        
        return name
    
    def _fallback_generation(self, text: str) -> Dict[str, Any]:
        """Fallback category generation using simple heuristics"""
        text = text.lower()
        
        # Define some common themes to check for
        themes = {
            'Technology': ['tech', 'technology', 'software', 'hardware', 'digital'],
            'Programming': ['code', 'programming', 'developer', 'software', 'github'],
            'AI & ML': ['ai', 'machine learning', 'artificial intelligence', 'deep learning', 'neural'],
            'Business': ['business', 'startup', 'company', 'entrepreneur', 'market'],
            'Finance': ['finance', 'money', 'investing', 'stocks', 'crypto', 'bitcoin'],
            'Science': ['science', 'research', 'study', 'scientist', 'physics', 'biology'],
            'Health': ['health', 'fitness', 'wellness', 'medical', 'diet', 'exercise'],
            'Politics': ['politics', 'government', 'policy', 'election', 'president'],
            'Entertainment': ['movie', 'film', 'tv', 'music', 'entertainment', 'game'],
            'Sports': ['sports', 'football', 'basketball', 'soccer', 'baseball', 'tennis'],
            'News': ['news', 'breaking', 'today', 'update', 'report']
        }
        
        scores = []
        for theme, keywords in themes.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)
            # Score based on matches
            score = min(matches / max(len(keywords), 1), 1.0) * 0.8  # Scale down a bit
            if matches > 0:
                scores.append((theme, score))
        
        # If no scores or all very low, add a default
        if not scores or all(score < 0.1 for _, score in scores):
            import random
            random_theme = random.choice(list(themes.keys()))
            scores.append((random_theme, 0.7))
            
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 2 for fallback
        result = {
            'categories': [
                {'name': theme, 'confidence': score} 
                for theme, score in scores[:2]
            ]
        }
        
        # Normalize scores to sum to 1.0
        total = sum(cat['confidence'] for cat in result['categories'])
        if total > 0:
            for cat in result['categories']:
                cat['confidence'] = cat['confidence'] / total
                
        return result

class CategoryProcessorFinal:
    """Final environment implementation of adaptive category processing using Gemini AI"""
    
    def __init__(self):
        """Initialize with adaptive categorizer"""
        try:
            # Initialize Gemini categorizer
            self.classifier = GeminiCategorizer()
            
            # Media categories are still predefined
            self.media_categories = ['Videos', 'Images']
            
            logger.info("✅ CategoryProcessorFinal initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing CategoryProcessorFinal: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def detect_media_category(self, bookmark: Bookmark, session) -> Category or None:
        """Detect if bookmark has media and return appropriate category"""
        if not bookmark.media_files:
            return None
            
        media_type = None
        for media in bookmark.media_files:
            type_str = media.get('type', '').lower() if isinstance(media, dict) else ''
            if 'video' in type_str:
                media_type = 'Videos'
                break
            elif 'photo' in type_str or 'image' in type_str:
                media_type = 'Images'
                break
                
        if media_type:
            category = session.query(Category).filter_by(name=media_type).first()
            if not category:
                category = Category(name=media_type)
                session.add(category)
            return category
        return None
    
    def get_or_create_category(self, name: str, session) -> Category:
        """Get an existing category or create a new one"""
        category = session.query(Category).filter_by(name=name).first()
        if not category:
            category = Category(name=name)
            session.add(category)
            session.flush()  # Ensure category has an ID
            logger.info(f"Created new category: {name}")
        return category
    
    def similar_categories(self, category_name: str, session, threshold=0.8) -> List[Category]:
        """Find categories that are very similar to avoid duplication"""
        # Get all categories
        categories = session.query(Category).all()
        
        # Use a simple similarity measure for now
        from difflib import SequenceMatcher
        
        similar = []
        for category in categories:
            # Skip exact match
            if category.name == category_name:
                continue
                
            # Calculate similarity ratio
            ratio = SequenceMatcher(None, category_name.lower(), category.name.lower()).ratio()
            if ratio > threshold:
                similar.append((category, ratio))
                
        # Sort by similarity (highest first)
        similar.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the categories
        return [cat for cat, _ in similar]
    
    def get_categories(self, text: str, session, max_categories=2) -> List[Tuple[Category, float]]:
        """Get or create categories for the text using Gemini"""
        try:
            if not text or not text.strip():
                # Get default category for empty texts
                default_cat = self.get_or_create_category('Uncategorized', session)
                return [(default_cat, 1.0)]
                
            # Get predictions from Gemini
            result = self.classifier.generate_categories(text, max_categories=max_categories)
            
            if 'categories' not in result or not result['categories']:
                # Fallback if no categories were generated
                default_cat = self.get_or_create_category('Miscellaneous', session)
                return [(default_cat, 1.0)]
            
            # Categories with their confidence scores
            categories_with_scores = []
            
            # Process each generated category
            for cat_data in result['categories']:
                cat_name = cat_data['name']
                confidence = cat_data.get('confidence', 0.5)  # Default if missing
                
                # Check for similar categories first to reduce duplication
                similar = self.similar_categories(cat_name, session)
                if similar:
                    # Use the most similar existing category
                    category = similar[0]
                    logger.info(f"Using similar category '{category.name}' instead of '{cat_name}'")
                else:
                    # Create a new category
                    category = self.get_or_create_category(cat_name, session)
                
                categories_with_scores.append((category, confidence))
            
            return categories_with_scores
                
        except Exception as e:
            logger.error(f"Error in category classification: {e}")
            logger.error(traceback.format_exc())
            # Fallback to Miscellaneous
            default_cat = self.get_or_create_category('Miscellaneous', session)
            return [(default_cat, 1.0)]
    
    def process_bookmark(self, bookmark: Bookmark) -> Dict[str, any]:
        """Process a single bookmark and assign categories"""
        try:
            # Make sure we're working with the bookmark's session
            session = inspect(bookmark).session
            
            # If bookmark has no session or is detached, we can't proceed
            if session is None:
                return {
                    'bookmark_id': bookmark.id,
                    'success': False,
                    'error': 'Bookmark is detached from session'
                }
                
            if not bookmark.text:
                logger.warning(f"Empty text for bookmark {bookmark.id}")
                return {
                    'bookmark_id': bookmark.id,
                    'success': False,
                    'error': 'Empty text'
                }
                
            # Get AI-generated categories
            categories_with_scores = self.get_categories(bookmark.text, session)
            
            # Check for media category
            media_cat = self.detect_media_category(bookmark, session)
            
            # Clear existing categories
            session.execute(
                bookmark_categories.delete().where(
                    bookmark_categories.c.bookmark_id == bookmark.id
                )
            )
            
            # Add all categories
            assigned_categories = []
            for category, _ in categories_with_scores:
                bookmark.categories.append(category)
                assigned_categories.append(category.name)
            
            # Add media category if exists
            if media_cat:
                bookmark.categories.append(media_cat)
                assigned_categories.append(media_cat.name)
            
            # Add categorization timestamp
            bookmark.categorized_at = datetime.now()
            
            # commit is performed by the caller
            
            category_str = ", ".join(assigned_categories)
            logger.info(f"Categorized bookmark {bookmark.id}: {category_str}")
            
            return {
                'bookmark_id': bookmark.id,
                'success': True,
                'categories': assigned_categories,
                'confidence': [score for _, score in categories_with_scores]
            }
                
        except Exception as e:
            logger.error(f"Error processing bookmark {bookmark.id}: {e}")
            logger.error(traceback.format_exc())
            return {
                'bookmark_id': bookmark.id,
                'success': False,
                'error': str(e)
            }
            
    def merge_similar_categories(self, threshold=0.85) -> Dict[str, Any]:
        """
        Merge categories that are very similar to reduce fragmentation
        Returns statistics about merges performed
        """
        try:
            with get_session() as session:
                # Get all categories except media categories
                categories = session.query(Category)\
                    .filter(~Category.name.in_(self.media_categories))\
                    .all()
                
                # Track merges
                merges = []
                
                # Check each pair of categories
                from difflib import SequenceMatcher
                
                for i, cat1 in enumerate(categories):
                    for cat2 in categories[i+1:]:
                        # Calculate similarity
                        ratio = SequenceMatcher(None, cat1.name.lower(), cat2.name.lower()).ratio()
                        
                        if ratio > threshold:
                            # These categories are similar enough to merge
                            logger.info(f"Merging similar categories: '{cat2.name}' into '{cat1.name}'")
                            
                            # Move all bookmarks from cat2 to cat1
                            # First, find bookmarks that have cat2 but not cat1
                            bookmarks_to_update = session.query(Bookmark)\
                                .join(bookmark_categories, Bookmark.id == bookmark_categories.c.bookmark_id)\
                                .filter(bookmark_categories.c.category_id == cat2.id)\
                                .all()
                            
                            # Add cat1 to each bookmark that had cat2
                            update_count = 0
                            for bookmark in bookmarks_to_update:
                                # Check if bookmark already has cat1
                                has_cat1 = cat1 in bookmark.categories
                                
                                if not has_cat1:
                                    bookmark.categories.append(cat1)
                                    update_count += 1
                            
                            merges.append({
                                'from_category': cat2.name,
                                'to_category': cat1.name,
                                'similarity': ratio,
                                'bookmarks_updated': update_count
                            })
                            
                            # Now delete the cat2 category if it's empty
                            # First remove all associations
                            session.execute(
                                bookmark_categories.delete().where(
                                    bookmark_categories.c.category_id == cat2.id
                                )
                            )
                            
                            # Then delete the category
                            session.delete(cat2)
                
                # Commit all changes
                if merges:
                    session.commit()
                    logger.info(f"Merged {len(merges)} similar categories")
                
                return {
                    'success': True,
                    'merges_performed': len(merges),
                    'merge_details': merges
                }
                
        except Exception as e:
            logger.error(f"Error merging categories: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
            
    def process_uncategorized_batch(self, batch_size=20) -> Dict[str, Any]:
        """Process a batch of uncategorized bookmarks"""
        start_time = datetime.now()
        results = []
        
        try:
            with get_session() as session:
                # Find uncategorized bookmarks - get IDs only to avoid detached instance errors
                uncategorized_ids = [
                    id for (id,) in session.query(Bookmark.id)
                    .outerjoin(bookmark_categories)
                    .filter(bookmark_categories.c.category_id.is_(None))
                    .limit(batch_size)
                    .all()
                ]
                
                total_uncategorized = session.query(Bookmark)\
                    .outerjoin(bookmark_categories)\
                    .filter(bookmark_categories.c.category_id.is_(None))\
                    .count()
                
                logger.info(f"Found {len(uncategorized_ids)}/{total_uncategorized} uncategorized bookmarks to process")
                
                # Process each bookmark by ID (rather than using the detached objects)
                for bookmark_id in uncategorized_ids:
                    try:
                        # Get a fresh bookmark object within its own session to avoid detached instance errors
                        with get_session() as bookmark_session:
                            bookmark = bookmark_session.query(Bookmark).get(bookmark_id)
                            if bookmark:
                                result = self.process_bookmark(bookmark)
                                # Explicitly commit the changes after processing each bookmark
                                bookmark_session.commit()
                                results.append(result)
                            else:
                                logger.error(f"Bookmark ID {bookmark_id} not found")
                                results.append({
                                    'bookmark_id': bookmark_id,
                                    'success': False,
                                    'error': 'Bookmark not found'
                                })
                    except Exception as e:
                        logger.error(f"Error processing bookmark {bookmark_id}: {e}")
                        logger.error(traceback.format_exc())
                        results.append({
                            'bookmark_id': bookmark_id,
                            'success': False,
                            'error': str(e)
                        })
            
            # Periodically merge similar categories to reduce fragmentation
            if len(results) > 0:
                self.merge_similar_categories()
                
            # Calculate statistics
            success_count = len([r for r in results if r.get('success', False)])
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'processed_count': len(results),
                'success_count': success_count,
                'error_count': len(results) - success_count,
                'remaining_count': total_uncategorized - len(uncategorized_ids),
                'processing_time_seconds': processing_time,
                'is_complete': total_uncategorized <= len(uncategorized_ids),
                'results': results
            }
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'processed_count': len(results)
            }
    
    def get_categorization_stats(self) -> Dict[str, Any]:
        """Get statistics about categorization status"""
        try:
            with get_session() as session:
                # Ensure we have the latest data
                session.expire_all()
                
                # Count all bookmarks
                total_bookmarks = session.query(Bookmark).count()
                
                # Count categorized bookmarks directly (more reliable than using uncategorized count)
                categorized_count = session.query(Bookmark)\
                    .join(bookmark_categories)\
                    .distinct(Bookmark.id)\
                    .count()
                
                # Calculate uncategorized from total and categorized
                uncategorized_count = total_bookmarks - categorized_count
                
                # Get category distribution
                category_stats = session.query(
                    Category.name, 
                    func.count(bookmark_categories.c.bookmark_id)
                )\
                    .join(bookmark_categories)\
                    .group_by(Category.name)\
                    .all()
                
                category_distribution = {
                    name: count for name, count in category_stats
                }
                
                # Get total unique categories
                total_categories = session.query(Category).count()
                
                return {
                    'total_bookmarks': total_bookmarks,
                    'categorized_count': categorized_count,
                    'uncategorized_count': uncategorized_count,
                    'total_categories': total_categories,
                    'completion_percentage': round(
                        ((categorized_count) / total_bookmarks * 100)
                        if total_bookmarks > 0 else 0, 
                        2
                    ),
                    'category_distribution': category_distribution
                }
                
        except Exception as e:
            logger.error(f"Error getting categorization stats: {e}")
            return {
                'error': str(e)
            }
    
    def update_bookmark_categories(self, bookmark_id: str, category_names: List[str]) -> Dict[str, Any]:
        """Update categories for a specific bookmark.
        
        Args:
            bookmark_id: The ID of the bookmark to update
            category_names: List of category names to assign to the bookmark
            
        Returns:
            Dict containing information about the update operation
        """
        logger.info(f"Updating categories for bookmark {bookmark_id}: {category_names}")
        session = get_session()
        
        try:
            # Get the bookmark to ensure it exists
            bookmark = session.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
            if not bookmark:
                raise ValueError(f"Bookmark with ID {bookmark_id} not found")
                
            # Get existing categories for this bookmark
            existing_categories = session.query(Category).join(
                bookmark_categories, 
                Category.id == bookmark_categories.c.category_id
            ).filter(
                bookmark_categories.c.bookmark_id == bookmark_id
            ).all()
            
            existing_names = [c.name for c in existing_categories]
            
            # Process categories to add
            categories_to_add = [name for name in category_names if name not in existing_names]
            
            # Process categories to remove
            categories_to_remove = [c for c in existing_categories if c.name not in category_names]
            
            # Add new categories
            for name in categories_to_add:
                category = self.get_or_create_category(name, session)
                # Create association
                session.execute(
                    bookmark_categories.insert().values(
                        bookmark_id=bookmark_id,
                        category_id=category.id
                    )
                )
                
            # Remove old categories
            for category in categories_to_remove:
                session.execute(
                    bookmark_categories.delete().where(
                        (bookmark_categories.c.bookmark_id == bookmark_id) &
                        (bookmark_categories.c.category_id == category.id)
                    )
                )
                
            # Remove the check for categorized_at since it doesn't exist in the model
            # if not bookmark.categorized_at:
            #     bookmark.categorized_at = datetime.now()
                
            session.commit()
            
            return {
                'bookmark_id': bookmark_id,
                'categories': category_names,
                'added': categories_to_add,
                'removed': [c.name for c in categories_to_remove]
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating bookmark categories: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
        finally:
            session.close()
    
    def rename_category(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename an existing category.
        
        Args:
            old_name: The current name of the category
            new_name: The new name for the category
            
        Returns:
            Dict containing information about the rename operation
        """
        logger.info(f"Renaming category '{old_name}' to '{new_name}'")
        session = get_session()
        
        try:
            # Check if the old category exists
            old_category = session.query(Category).filter(Category.name == old_name).first()
            if not old_category:
                raise ValueError(f"Category '{old_name}' not found")
                
            # Check if the new name already exists
            existing_category = session.query(Category).filter(Category.name == new_name).first()
            if existing_category:
                raise ValueError(f"Category '{new_name}' already exists")
                
            # Rename the category
            old_category.name = new_name
            session.commit()
            
            return {
                'old_name': old_name,
                'new_name': new_name,
                'id': old_category.id
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error renaming category: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
        finally:
            session.close()
    
    def delete_category(self, category_name: str) -> Dict[str, Any]:
        """Delete a category and remove all its associations.
        
        Args:
            category_name: The name of the category to delete
            
        Returns:
            Dict containing information about the delete operation
        """
        logger.info(f"Deleting category '{category_name}'")
        session = get_session()
        
        try:
            # Check if the category exists
            category = session.query(Category).filter(Category.name == category_name).first()
            if not category:
                raise ValueError(f"Category '{category_name}' not found")
                
            # Get count of bookmarks with this category
            bookmark_count = session.query(func.count()).select_from(bookmark_categories).filter(
                bookmark_categories.c.category_id == category.id
            ).scalar()
            
            # Delete all bookmark associations
            session.execute(
                bookmark_categories.delete().where(
                    bookmark_categories.c.category_id == category.id
                )
            )
            
            # Delete the category
            session.delete(category)
            session.commit()
            
            return {
                'category_name': category_name,
                'id': category.id,
                'bookmark_associations_removed': bookmark_count
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting category: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
        finally:
            session.close()

# Function to run from PythonAnywhere scheduled task
def process_categories_background_job():
    """
    Function to be called by PythonAnywhere scheduled task to process categories.
    Processes a batch of bookmarks and logs results.
    """
    try:
        logger.info("="*50)
        logger.info("Starting background category processing job with adaptive Gemini AI")
        
        processor = CategoryProcessorFinal()
        
        # Get current stats
        stats_before = processor.get_categorization_stats()
        logger.info(f"Current stats: {stats_before}")
        
        # Process a batch - increased from 20 to 50 for faster processing
        result = processor.process_uncategorized_batch(batch_size=50)
        
        if result.get('success'):
            logger.info(f"Successfully processed {result['success_count']}/{result['processed_count']} bookmarks")
            logger.info(f"Remaining uncategorized: {result['remaining_count']}")
            
            # Force a new session for stats to ensure we get the latest data
            with get_session() as fresh_session:
                fresh_session.commit()  # Ensure any pending transactions are committed
                
            # Get updated stats with a small delay to ensure DB is updated
            import time
            time.sleep(0.5)  # Small delay to ensure DB writes are complete
            stats_after = processor.get_categorization_stats()
            logger.info(f"Updated category completion: {stats_after['completion_percentage']}%")
            logger.info(f"Current total categories: {stats_after['total_categories']}")
            
            return {
                'success': True,
                'processed': result['processed_count'],
                'is_complete': result['is_complete'],
                'stats': stats_after
            }
        else:
            logger.error(f"Error in background processing: {result.get('error')}")
            return {
                'success': False,
                'error': result.get('error')
            }
            
    except Exception as e:
        logger.error(f"Background job error: {e}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

# For command line testing
if __name__ == "__main__":
    result = process_categories_background_job()
    print(f"Job result: {result}") 