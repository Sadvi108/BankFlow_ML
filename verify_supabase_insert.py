from app.db import get_db
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

def verify_insert():
    print("Testing Supabase INSERT with specific columns...")
    db = get_db()
    if not db:
        print("❌ Failed to initialize Supabase client.")
        return

    # Create a dummy entry
    test_id = uuid.uuid4().hex
    entry = {
        "id": test_id,
        "filename": "test_debug.jpg",
        "bank": {"name": "TEST BANK"},
        "fields": {
            "transaction_id": "REF123456",
            "amount": "100.00",
            "date": "2024-01-01"
        }
    }

    # Construct the record exactly as the app does
    record = {
        "id": entry.get("id"),
        "data": entry,
        "Bank_Name": "TEST BANK",
        "Reference_Id": "REF123456",
        "amount": "100.00",
        "date": "2024-01-01"
    }

    print(f"Attempting to insert record with keys: {list(record.keys())}")
    
    try:
        db.client.from_(db.table_name).insert(record).execute()
        print("✅ Insert successful! Check Supabase for:")
        print(f"   ID: {test_id}")
        print(f"   Bank_Name: TEST BANK")
        print(f"   Reference_Id: REF123456")
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        print("   This confirms the column names are incorrect or there's a permission issue.")

if __name__ == "__main__":
    verify_insert()
