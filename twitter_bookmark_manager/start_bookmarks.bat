@echo off
echo Starting Bilbeny's Bookmarks...

:: Set the project root directory
set PROJECT_ROOT=%~dp0

:: Activate virtual environment and start server (with error suppression)
start /min "BookmarksServer" cmd /c "cd %PROJECT_ROOT% && call venv\Scripts\activate && python web\server.py 2>nul"

:: Wait for server to be ready
:CHECKSERVER
timeout /t 1 /nobreak > nul
curl -s http://localhost:5000 > nul
if errorlevel 1 (
    echo Waiting for server to start...
    goto CHECKSERVER
) else (
    echo Server is ready, opening browser...
)

:: Start Chrome with a specific title for our window
start "Bilbeny's Bookmarks" chrome --new-window "http://localhost:5000"

:: Wait for and monitor specific Chrome window
:LOOP
timeout /t 2 /nobreak > nul
tasklist /FI "WINDOWTITLE eq Bilbeny's Bookmarks*" | find "chrome.exe" > nul
if errorlevel 1 (
    echo Bookmark manager closed, stopping server...
    taskkill /FI "WindowTitle eq BookmarksServer" /T /F > nul
    exit
) else (
    goto LOOP
)