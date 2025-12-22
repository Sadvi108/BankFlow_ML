@echo off
echo ========================================
echo Starting Simple Receipt Tester Server
echo ========================================
echo.
cd /d "%~dp0"
echo Directory: %CD%
echo.
echo Starting server on http://localhost:8081
echo.
echo Once you see "Server running", open your browser to:
echo http://localhost:8081
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe simple_http_server.py
pause
