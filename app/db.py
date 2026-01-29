import os
import json
from typing import Dict, List, Optional, Any
try:
    from postgrest import SyncPostgrestClient as PostgrestClient
except ImportError:
    try:
        from postgrest import PostgrestClient
    except ImportError:
        PostgrestClient = None

class SupabaseDB:
    def __init__(self, url: str, key: str):
        if not PostgrestClient:
            raise ImportError("postgrest library is not installed")
        
        # Ensure URL points to the REST endpoint
        if not url.endswith("/rest/v1") and not url.endswith("/rest/v1/"):
            if url.endswith("/"):
                url = f"{url}rest/v1"
            else:
                url = f"{url}/rest/v1"
                
        self.client = PostgrestClient(url, headers={"apikey": key, "Authorization": f"Bearer {key}"})
        self.table_name = "Receipts"

    def append_annotation(self, entry: Dict) -> None:
        """
        Inserts a new record into Supabase.
        Attempts to write to specific columns (bank_name, reference_id, amount, date)
        in addition to the 'data' JSONB column.
        """
        # Base record with ID and JSON data
        record = {
            "id": entry.get("id"),
            "data": entry
        }
        
        # Extract fields for dedicated columns
        bank_name = entry.get("bank", {}).get("name")
        fields = entry.get("fields", {})
        
        # Determine the best reference ID to use
        # Priority: transaction_id -> reference_number -> invoice_number
        ref_id = fields.get("transaction_id") or fields.get("reference_number") or fields.get("invoice_number")
        
        # Add optional columns
        if bank_name:
            record["Bank_Name"] = bank_name
        if ref_id:
            record["Reference_Id"] = ref_id
        if fields.get("amount"):
            record["amount"] = fields.get("amount")
        if fields.get("date"):
            record["date"] = fields.get("date")

        try:
            # Try inserting with all columns
            self.client.from_(self.table_name).insert(record).execute()
        except Exception as e:
            error_msg = str(e)
            # If error is likely due to missing columns (400 Bad Request), retry with just basic fields
            if "400" in error_msg or "column" in error_msg.lower():
                print(f"Insert with columns failed ({e}). Retrying with only 'id' and 'data'...")
                try:
                    basic_record = {
                        "id": entry.get("id"),
                        "data": entry
                    }
                    self.client.from_(self.table_name).insert(basic_record).execute()
                except Exception as e2:
                    print(f"Retry failed: {e2}")
                    raise e2
            else:
                print(f"Error inserting into Supabase: {e}")
                raise e

    def read_annotations(self) -> List[Dict]:
        try:
            response = self.client.from_(self.table_name).select("*").execute()
            results = []
            for row in response.data:
                # If we use the 'data' JSONB column pattern
                if "data" in row:
                    item = row["data"]
                    # Ensure ID is preserved/consistent
                    if "id" not in item and "id" in row:
                        item["id"] = row["id"]
                    results.append(item)
                else:
                    # If the table structure matches the object directly
                    results.append(row)
            return results
        except Exception as e:
            print(f"Error reading from Supabase: {e}")
            return []

    def update_annotation(self, item_id: str, updates: Dict) -> bool:
        try:
            # Fetch current data to merge
            # Note: This is not atomic. For atomic updates, we'd need a Postgres function or deeper JSONB support.
            response = self.client.from_(self.table_name).select("data").eq("id", item_id).execute()
            if not response.data:
                return False
            
            current_data = response.data[0].get("data", {})
            current_data.update(updates)
            
            self.client.from_(self.table_name).update({"data": current_data}).eq("id", item_id).execute()
            return True
        except Exception as e:
            print(f"Error updating Supabase: {e}")
            return False

    def summary(self) -> Dict:
        rows = self.read_annotations()
        total = len(rows)
        banks = {}
        for r in rows:
            b = (r.get("bank", {}) or {}).get("name")
            if b:
                banks[b] = banks.get(b, 0) + 1
        return {"total": total, "per_bank": banks}

def get_db() -> Optional[SupabaseDB]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if url and key:
        return SupabaseDB(url, key)
    return None
