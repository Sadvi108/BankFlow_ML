# START SERVER - Manual Instructions

## Quick Start

Open a **new terminal/command prompt** and run:

```bash
cd c:\Users\User\Documents\trae_projects\CLA_Training
python simple_server.py
```

Then open your browser to: **http://localhost:8081**

---

## What You'll See

The upload interface where you can:
1. Click "Choose File" 
2. Select a receipt from the `Receipts` folder
3. Click "Upload and Extract"
4. See the extracted transaction IDs, bank name, amount, etc.

---

## Testing All Receipts

Upload each of the 23 receipts one by one and note:
- ✅ Which ones successfully extract IDs
- ❌ Which ones fail or show no IDs
- Share the failures with me and I'll fix them

---

## Alternative: Command Line Test

If the server won't start, run this instead:

```bash
python quick_receipt_test.py
```

This will test all 23 receipts automatically and show results.
