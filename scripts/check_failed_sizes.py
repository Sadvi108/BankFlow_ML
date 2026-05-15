import os
from pathlib import Path
import sys

# Fix encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_failed_images():
    uploads_dir = Path("data/uploads")
    
    failed_files = [
        "287156c2-2b81-4d79-b34b-abe459a09b07.jpg",
        "508132bb-62a3-4554-8fc6-875a68cfcc01.jpg",
        "68f6c6ba-7856-4614-845c-4b983004e376.jpg",
        "748a9bca-5a73-4d1d-9bdb-eebc18312a84.jpg",
        "b681bef9-7201-47a7-bf85-d538fbb60156.jpg",
        "c0ad1ab2-ee87-4eee-8471-6c6de8d804ef.jpg",
        "c195f6da-3781-461b-9d1f-39090e4e5c04.jpg",
        "c2da7433-4aad-47f8-8079-a97620c7f473.jpg",
        "cfd733d8-f991-4b8b-8bbb-8f29ee8a078e.jpg",
        "d4a7863f-63d8-430d-88f2-e9cffd6d5405.jpg",
        "d7456d72-3098-4e3b-b6e7-8042f6153008.jpg",
        "d7f516a0-d925-4a34-adf7-890e21beff3a.jpg",
        "dfa5dc6b-13f5-4aa5-9823-fb3a48028313.jpg",
        "e5c6e179-e576-4a8e-a7ce-476520539a25.jpg",
        "ec21ab89-06ce-4737-923c-0d1eeeffc675.jpg",
        "f24c8acd-0a5b-435d-a924-c60a92079ed6.jpg"
    ]
    
    print(f"{'Filename':<45} | {'Size (bytes)':<12} | {'Status'}")
    print("-" * 80)
    
    for filename in failed_files:
        path = uploads_dir / filename
        if not path.exists():
            print(f"{filename:<45} | {'MISSING':<12} | ❌")
            continue
            
        size = path.stat().st_size
        status = "❌ Too small" if size < 1000 else "✅ OK size"
        print(f"{filename:<45} | {size:<12} | {status}")

if __name__ == "__main__":
    check_failed_images()
