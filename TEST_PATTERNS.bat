@echo off
echo ========================================
echo Testing Pattern Fixes (No Server Needed)
echo ========================================
echo.
cd /d "%~dp0"
echo Running pattern matching test...
echo.
C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe test_100_percent_accuracy.py
echo.
echo ========================================
echo Test complete! Check results above.
echo ========================================
pause
