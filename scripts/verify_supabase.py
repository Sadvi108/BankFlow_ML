from app.db import get_db
import os
from dotenv import load_dotenv

load_dotenv()

def verify_connection():
    print("Testing Supabase connection...")
    db = get_db()
    if not db:
        print("❌ Failed to initialize Supabase client. Check .env file.")
        return

    print(f"✅ Client initialized with URL: {os.getenv('SUPABASE_URL')}")
    
    try:
        print("Attempting to fetch from 'Receipts' table...")
        rows = db.read_annotations()
        print(f"✅ Connection successful! Found {len(rows)} records.")
        if len(rows) > 0:
            print("Latest record ID:", rows[-1].get('id'))
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("\nPossible causes:")
        print("1. The 'Receipts' table does not exist yet.")
        print("2. Row Level Security (RLS) policies might be blocking access.")
        print("3. Network connectivity issues.")

if __name__ == "__main__":
    verify_connection()
