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
        Assumes a table named 'receipts' exists with columns:
        - id (uuid/text)
        - data (jsonb)
        """
        record = {
            "id": entry.get("id"),
            "data": entry
        }
        # Optionally add other columns if they exist in the schema and we want to index them
        # e.g. record["bank_name"] = entry.get("bank", {}).get("name")
        
        try:
            self.client.from_(self.table_name).insert(record).execute()
        except Exception as e:
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
