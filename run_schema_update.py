#!/usr/bin/env python3
"""
Helper script to run the schema update directly.
"""
import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import and run
from twitter_bookmark_manager.deployment.pythonanywhere.database.db_schema_update import update_database_schema

if __name__ == "__main__":
    result = update_database_schema()
    print(f"Schema update result: {result}") 