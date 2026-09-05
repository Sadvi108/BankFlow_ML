import json
import os
import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from threading import RLock

class HistoryManager:
    """Manages extraction history using a local JSON file."""
    
    def __init__(self, history_file: str = "data/history.json"):
        self._lock = RLock()
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.history_file.exists():
            with open(self.history_file, 'w') as f:
                json.dump([], f)

    def add_entry(self, entry: Dict[str, Any]) -> str:
        with self._lock:
            return self._add_entry(entry)

    def _add_entry(self, entry: Dict[str, Any]) -> str:
        """Add a new extraction entry and return its ID."""
        history = self.get_all()
        
        # Standardize entry
        entry_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        new_entry = {
            "id": entry_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "filename": entry.get("filename", "unknown"),
            "bank_name": entry.get("bank_name", "Unknown"),
            "reference_id": entry.get("reference_id", ""),
            "amount": entry.get("amount", ""),
            "date": entry.get("date", ""),
            "confidence": entry.get("confidence", 0.0),
            "status": "success" if entry.get("reference_id") else "warning",
            "notes": "",
            # A receipt carries several references and three distinct money
            # figures. Storing only the elected primary and a single amount
            # threw the rest away before the CSV export could ever see them.
            "references": entry.get("references", []),
            "fee": entry.get("fee"),
            "total_debit": entry.get("total_debit"),
            "beneficiary_bank": entry.get("beneficiary_bank"),
            "needs_review": entry.get("needs_review", False),
        }
        
        history.insert(0, new_entry) # Most recent first
        # Limit to last 50 entries
        history = history[:50]
        
        self._save(history)
        return entry_id

    def get_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._get_all()

    def _get_all(self) -> List[Dict[str, Any]]:
        """Get all history entries."""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        with self._lock:
            return self._update_entry(entry_id, updates)

    def _update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing entry."""
        history = self.get_all()
        found = False
        
        for entry in history:
            if entry["id"] == entry_id:
                for key, value in updates.items():
                    if key in entry:
                        entry[key] = value
                found = True
                break
        
        if found:
            self._save(history)
        return found

    def _save(self, history: List[Dict[str, Any]]):
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

# Global instance
history_manager = HistoryManager()
