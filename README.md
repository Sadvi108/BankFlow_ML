# CLA_Training — Bank Receipt OCR MVP

This project scaffolds a minimal FastAPI backend and a simple upload UI to classify Malaysian bank receipts and extract key fields (transaction/reference number, amount, date). It is designed as a starting point that can be extended with cloud OCR (AWS Textract / Google Document AI / Azure Form Recognizer), layout-aware ML models (LayoutLM/Donut), and admin verification flows.

## Features (MVP)
- Upload PDF or image (PNG/JPG) bank receipt
- Convert PDF to image (first page) via PyMuPDF
- OCR via `pytesseract` if Tesseract is available; otherwise graceful fallback
- Simple bank classifier using keyword matching (Maybank, CIMB, Public Bank, RHB, Hong Leong, AmBank, Bank Islam, BSN)
- Regex-based field extraction for common patterns (Ref No / Transaction ID / Amount / Date)
- Minimal HTML UI that shows extracted JSON results

## Quick Start

1. Ensure Python 3.9+ is installed.
2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Optional: Install Tesseract OCR (for local OCR)
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - After install, verify:
     ```powershell
     tesseract --version
     ```
4. Run the dev server:
   ```powershell
   python -m uvicorn app.main:app --reload
   ```
5. Open http://127.0.0.1:8000/ and upload a bank receipt to see extraction results.

## Project Structure
```
app/
  main.py        # FastAPI app routes and HTML UI
  ocr.py         # OCR utilities (pytesseract) with graceful fallback
  utils.py       # File handling and PDF-to-image conversion
  classify.py    # Bank keyword-based classifier
  extract.py     # Regex-based field extraction
templates/
  upload.html    # Minimal upload page (HTML + fetch)
data/
  uploads/       # Uploaded files
  processed/     # Processed images
```

## Notes and Next Steps
- Cloud OCR: Integrate Textract / Document AI / Form Recognizer for higher accuracy and bounding boxes.
- Layout-aware models: Add LayoutLMv3 / Donut fine-tuning with Label Studio annotations.
- Admin UI: Build human-in-the-loop verification page with bounding box highlights and edit/confirm/reject flow.
- Storage/DB: Wire up PostgreSQL + S3 for metadata, audit logs, and object storage.
- Security & Monitoring: Add auth (Keycloak/Auth0), metrics, logging, and error reporting.

## License
Internal use. Adapt as needed.