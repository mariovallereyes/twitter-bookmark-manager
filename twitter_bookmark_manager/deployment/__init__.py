"""
Deployment package for Twitter Bookmark Manager
""" 

# Make deployment a proper package 

# Deployment package initialization
# This file makes the deployment directory a proper Python package

import sys
import os

# Create a twitter_bookmark_manager module for compatibility
# This allows imports of "twitter_bookmark_manager" to work even when
# railway uses the deployment/final directory as the root
if 'twitter_bookmark_manager' not in sys.modules:
    import types
    twitter_bookmark_manager = types.ModuleType('twitter_bookmark_manager')
    sys.modules['twitter_bookmark_manager'] = twitter_bookmark_manager
    print("Created twitter_bookmark_manager module for compatibility") 