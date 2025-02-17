# /tests/test_bookmark_extractor.py

import unittest
import json
import requests
from pathlib import Path
from core.universal_bookmark_extractor import BookmarkExtractor
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestBookmarkExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = BookmarkExtractor()
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.temp_file = self.temp_dir / "test_bookmarks.json"

    def test_session_handling(self):
        """Test if session handling logic is present and correct"""
        script = self.extractor.get_extraction_script()
        
        # Check session token extraction with updated code
        self.assertIn("const cookieString = document.cookie", script)
        self.assertIn("const tokenCookie = cookieString.split('; ').find(row => row.startsWith('ct0='))", script)
        self.assertIn("x-csrf-token", script)
        self.assertIn("x-twitter-active-user", script)
        self.assertIn("OAuth2Session", script)

    def test_api_extraction_method(self):
        """Test Twitter internal API extraction method"""
        script = self.extractor.get_extraction_script()
        
        # Check API endpoint and parameters
        self.assertIn("bookmark_timeline", script)
        self.assertIn("variables", script)
        self.assertIn("features", script)
        
        # Check pagination handling
        self.assertIn("cursor", script)
        self.assertIn("count", script)

    def test_data_structure(self):
        """Test bookmark data structure"""
        script = self.extractor.get_extraction_script()
        
        # Check all required fields from API response
        required_fields = [
            "tweet_id",
            "tweet_url",
            "full_text",
            "created_at",
            "user",
            "media",
            "bookmark_date"
        ]
        
        for field in required_fields:
            self.assertIn(field, script)

    def test_progress_updates(self):
        """Test progress update functionality"""
        script = self.extractor.get_extraction_script()
        
        # Check API progress updates
        self.assertIn("Starting bookmark extraction", script)
        self.assertIn("Fetching bookmarks page", script)
        self.assertIn("Processed", script)
        self.assertIn("Finished extracting", script)

    def test_error_handling(self):
        """Test error handling"""
        script = self.extractor.get_extraction_script()
        
        # Check API error handling
        self.assertIn("try {", script)
        self.assertIn("catch (error)", script)
        self.assertIn("if (!response.ok)", script)
        self.assertIn("response.status === 429", script)  # Rate limit check
        self.assertIn("response.status === 401", script)  # Auth check

    def test_response_processing(self):
        """Test API response processing"""
        script = self.extractor.get_extraction_script()
        
        # Check response handling
        self.assertIn("instructions", script)
        self.assertIn("entries", script)
        self.assertIn("content", script)
        self.assertIn("itemContent", script)
        self.assertIn("tweet_results", script)
        self.assertIn("result", script)

    def test_injection_wrapper(self):
        """Test injection wrapper functionality"""
        wrapper = self.extractor.get_injection_wrapper()
        
        # Check wrapper structure
        self.assertIsInstance(wrapper, dict)
        self.assertIn('status', wrapper)
        self.assertIn('script', wrapper)
        self.assertIn('message', wrapper)
        
        # Check success case
        self.assertEqual(wrapper['status'], 'success')
        self.assertIsInstance(wrapper['script'], str)
        self.assertEqual(wrapper['message'], 'Extraction script ready')

    def test_live_extraction(self):
        """Test actual bookmark extraction in browser"""
        try:
            print("\nStarting Chrome for testing...")
            
            options = webdriver.ChromeOptions()
            # Use your actual Chrome profile
            options.add_argument("user-data-dir=C:\\Users\\mario\\AppData\\Local\\Google\\Chrome\\User Data")
            options.add_argument("profile-directory=Default")
            
            # Basic options
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            # Create service with increased timeout
            service = webdriver.ChromeService(
                service_args=['--verbose'],
                service_log_path=str(self.temp_dir / "chromedriver.log")
            )
            
            try:
                from selenium.webdriver.common.selenium_manager import SeleniumManager
                # Set longer timeout for driver creation
                SeleniumManager.TIMEOUT = 300  # 5 minutes
                
                print("Attempting to start Chrome with your profile...")
                driver = webdriver.Chrome(
                    options=options,
                    service=service
                )
                driver.set_page_load_timeout(60)  # 60 seconds page load timeout
                print("Chrome started successfully!")
                
                print("Navigating to Twitter bookmarks...")
                driver.get("https://twitter.com/i/bookmarks")
                
                # Wait for bookmarks to load with longer timeout
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "article"))
                )
                
                print("Starting bookmark extraction...")
                result = driver.execute_script(self.extractor.get_extraction_script())
                
                # Verify extraction worked
                self.assertIsInstance(result, list)
                if len(result) > 0:
                    print(f"\nSuccessfully extracted {len(result)} bookmarks")
                    self.assertIn('tweet_url', result[0])
                    self.assertIn('full_text', result[0])
                    
                    # Save results for inspection
                    with open(self.temp_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    print(f"Results saved to: {self.temp_file}")
                    
            except Exception as e:
                print(f"\nChrome error: {str(e)}")
                if 'driver' in locals():
                    print(f"Current URL: {driver.current_url}")
                raise
                
            finally:
                if 'driver' in locals():
                    try:
                        driver.quit()
                    except:
                        pass
                    
        except Exception as e:
            self.fail(f"Live extraction failed: {str(e)}")

    def tearDown(self):
        """Clean up any test artifacts"""
        import shutil
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

if __name__ == '__main__':
    unittest.main()