
import sys
import shutil

print(f"Python version: {sys.version}")

try:
    import pdfplumber
    print("pdfplumber is installed")
except ImportError:
    print("pdfplumber is NOT installed")

try:
    import pytesseract
    print("pytesseract is installed")
    try:
        print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    except Exception as e:
        print(f"Tesseract binary not found or error: {e}")
except ImportError:
    print("pytesseract is NOT installed")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("OpenCV is NOT installed")

tesseract_cmd = shutil.which("tesseract")
print(f"Tesseract command path: {tesseract_cmd}")
