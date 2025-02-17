import os
import subprocess
import sys

def main():
    # Get the directory where the exe will be
    if getattr(sys, 'frozen', False):
        # Running as compiled
        script_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to batch file
    bat_path = os.path.join(script_dir, 'start_bookmarks.bat')
    
    # Run the batch file
    subprocess.Popen([bat_path], creationflags=subprocess.CREATE_NO_WINDOW)

if __name__ == "__main__":
    main()