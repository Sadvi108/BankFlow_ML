import cv2
import os
from pathlib import Path
import sys

# Fix encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def debug_image(filename):
    path = Path("data/uploads") / filename
    print(f"Checking {path}")
    
    if not path.exists():
        print("❌ File does not exist")
        return
        
    print(f"Size: {path.stat().st_size} bytes")
    
    # Try cv2
    img = cv2.imread(str(path))
    if img is None:
        print("❌ cv2.imread failed (returned None)")
    else:
        print(f"✅ cv2 loaded image: {img.shape}")
        
    # Try PIL as fallback
    try:
        from PIL import Image
        pil_img = Image.open(path)
        print(f"✅ PIL loaded image: {pil_img.size}, mode={pil_img.mode}")
    except Exception as e:
        print(f"❌ PIL failed: {e}")

if __name__ == "__main__":
    debug_image("287156c2-2b81-4d79-b34b-abe459a09b07.jpg")
