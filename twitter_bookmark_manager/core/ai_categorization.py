from transformers import pipeline
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Tuple
from config.constants import BOOKMARK_CATEGORIES
from database.db import get_db_session, get_vector_store
import numpy as np
from database.models import Bookmark  # Add this import at the top

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookmarkCategorizer:
    def __init__(self):
        """Initialize the categorizer with required models"""
        try:
            # Zero-shot classifier for categorization
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU. Use 0 for GPU
            )
            
            # Sentence transformer for embeddings
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load categories
            self.categories = [cat['name'] for cat in BOOKMARK_CATEGORIES]
            self.category_descriptions = {cat['name']: cat['description'] 
                                       for cat in BOOKMARK_CATEGORIES}
            
            logger.info("BookmarkCategorizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BookmarkCategorizer: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text"""
        try:
            if not text.strip():
                raise ValueError("Empty text provided for embedding generation")
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def categorize_text(self, text: str) -> Tuple[str, float]:
        """Categorize text into predefined categories"""
        try:
            if not text.strip():
                raise ValueError("Empty text provided for categorization")
            
            # Improved hypothesis template
            result = self.classifier(
                text,
                candidate_labels=self.categories,
                hypothesis_template="This is a {}" # Simpler template often works better
            )
            category = result['labels'][0]
            confidence = result['scores'][0]
            return category, confidence
        except Exception as e:
            logger.error(f"Error categorizing text: {e}")
            raise

    def process_bookmark(self, bookmark_id: str, text: str) -> Dict[str, Any]:
        """Process a single bookmark: categorize and generate embedding"""
        try:
            # Generate embedding
            embedding = self.generate_embedding(text)
            
            # Categorize text
            category, confidence = self.categorize_text(text)
            
            # Store in vector database
            vector_store = get_vector_store()
            vector_store.add_embeddings(
                ids=[bookmark_id],
                embeddings=[embedding],
                metadatas=[{
                    'text': text,
                    'category': category,
                    'confidence': confidence
                }],
                texts=[text]
            )
            
            # Store category in SQLite
            with get_db_session() as session:
                # Update bookmark category
                bookmark = session.query(Bookmark).get(bookmark_id)
                if bookmark:
                    bookmark.category = category
                    bookmark.category_confidence = confidence
                    session.commit()
            
            return {
                'bookmark_id': bookmark_id,
                'category': category,
                'confidence': confidence,
                'embedding_generated': True
            }
            
        except Exception as e:
            logger.error(f"Error processing bookmark {bookmark_id}: {e}")
            raise

    def batch_process_bookmarks(self, bookmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple bookmarks in batch"""
        results = []
        for bookmark in bookmarks:
            try:
                result = self.process_bookmark(
                    bookmark_id=bookmark['id'],
                    text=bookmark['text']
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch processing for bookmark {bookmark['id']}: {e}")
                results.append({
                    'bookmark_id': bookmark['id'],
                    'error': str(e)
                })
        return results