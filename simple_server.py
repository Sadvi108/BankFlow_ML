"""
Simple FastAPI server for testing the simplified web interface
"""

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
import logging
from datetime import datetime
import re
import io
import csv
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from app.history_manager import history_manager
from app.db import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DB
db = get_db()
if db:
    logger.info("Supabase connection initialized")
else:
    logger.warning("Supabase connection not available")

# Initialize app and templates
app = FastAPI(
    title="Bank Receipt Extractor",
    description="Simple bank receipt processing",
    version="1.0.0"
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory=str(Path("templates")))

# Data storage
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Import the robust V3 pattern matcher and layout-aware extractor
from app.ultimate_patterns_v3 import extract_all_fields_v3
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.layout_aware_extractor import layout_extractor

# Initialize enhanced OCR pipeline
ocr_pipeline = EnhancedOCRPipeline()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the simplified upload page"""
    return templates.TemplateResponse("simple_upload.html", {"request": request})

@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.post("/extract", response_class=JSONResponse)
async def extract_receipt(file: UploadFile = File(...)):
    """Extract transaction details from uploaded receipt"""
    try:
        # Validate file type (accept by extension or known content types)
        allowed_exts = {".pdf", ".jpg", ".jpeg", ".png"}
        file_ext = Path(file.filename).suffix.lower() if file.filename else ""
        
        # Ensure content_type is not None
        content_type = file.content_type or "application/octet-stream"
        
        content_ok = (
            (content_type.startswith("image/") or content_type == "application/pdf")
            or (content_type == "application/octet-stream" and file_ext in allowed_exts)
            or (file_ext in allowed_exts)
        )
        if not content_ok:
            raise HTTPException(status_code=400, detail="Only image and PDF files are supported")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = file_ext or ".jpg"
        filename = f"{file_id}{file_ext}"
        file_path = UPLOAD_DIR / filename
        
        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        # 1. OCR PROCESSING (Get text + tokens)
        try:
            # For images, we want the tokens for layout awareness
            if content_type.startswith("image/") or (content_type == "application/octet-stream" and file_ext in {".jpg", ".jpeg", ".png"}):
                import cv2
                import numpy as np
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                ocr_result = ocr_pipeline.extract_text_with_confidence(image)
            else:
                # For PDFs, use temp file to avoid large buffer
                import tempfile
                # Create a temp file and close it immediately so other processes/libraries can open it
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                # File is closed here
                
                try:
                    # Try embedded text first (fast path)
                    embedded_text = ""
                    try:
                        import fitz
                        doc = fitz.open(tmp_path)
                        parts = []
                        for page in doc:
                            parts.append(page.get_text("text"))
                        doc.close()
                        embedded_text = " ".join(parts).strip()
                    except Exception:
                        embedded_text = ""

                    if len(embedded_text) >= 50:
                        ocr_result = {"text": embedded_text, "tokens": [], "confidence": 0.7}
                    else:
                        # Primary pipeline
                        ocr_result = ocr_pipeline.process_file(tmp_path)
                        text_primary = (ocr_result.get('text') or "").strip()
                        if len(text_primary) < 50:
                            import fitz
                            import numpy as np
                            import cv2
                            from PIL import Image
                            doc = fitz.open(tmp_path)
                            texts = []
                            for page in doc:
                                t = page.get_text("text")
                                if t:
                                    texts.append(t)
                            if sum(len(t) for t in texts) < 50:
                                for page in doc:
                                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                                    png_bytes = pix.tobytes("png")
                                    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                                    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                                    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
                                    res_a = ocr_pipeline.extract_text_with_confidence(arr)
                                    res_b = ocr_pipeline.extract_text_with_confidence(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
                                    res = res_a if len((res_a.get("text") or "")) >= len((res_b.get("text") or "")) else res_b
                                    if res.get("text"):
                                        texts.append(res["text"])
                            doc.close()
                            fallback_text = " ".join(texts).strip()
                            if fallback_text:
                                ocr_result = {"text": fallback_text, "tokens": [], "confidence": 0.6}
                            if len(fallback_text) < 50:
                                try:
                                    import pdfplumber
                                    with pdfplumber.open(tmp_path) as pdf:
                                        pl_texts = []
                                        for page in pdf.pages:
                                            pl_texts.append(page.extract_text() or "")
                                    pl_text = " ".join(pl_texts).strip()
                                    if pl_text and len(pl_text) > len(fallback_text):
                                        ocr_result = {"text": pl_text, "tokens": [], "confidence": 0.6}
                                except Exception:
                                    pass
                        import fitz
                        import numpy as np
                        import cv2
                        from PIL import Image
                        doc = fitz.open(tmp_path)
                        texts = []
                        for page in doc:
                            t = page.get_text("text")
                            if t:
                                texts.append(t)
                        if sum(len(t) for t in texts) < 50:
                            for page in doc:
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                                png_bytes = pix.tobytes("png")
                                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                                arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                                thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
                                res_a = ocr_pipeline.extract_text_with_confidence(arr)
                                res_b = ocr_pipeline.extract_text_with_confidence(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
                                res = res_a if len((res_a.get("text") or "")) >= len((res_b.get("text") or "")) else res_b
                                if res.get("text"):
                                    texts.append(res["text"])
                        doc.close()
                        fallback_text = " ".join(texts).strip()
                        if fallback_text:
                            ocr_result = {"text": fallback_text, "tokens": [], "confidence": 0.6}
                        if len(fallback_text) < 50:
                            try:
                                import pdfplumber
                                with pdfplumber.open(tmp_path) as pdf:
                                    pl_texts = []
                                    for page in pdf.pages:
                                        pl_texts.append(page.extract_text() or "")
                                pl_text = " ".join(pl_texts).strip()
                                if pl_text and len(pl_text) > len(fallback_text):
                                    ocr_result = {"text": pl_text, "tokens": [], "confidence": 0.6}
                                    logger.info(f"pdfplumber fallback success: {len(pl_text)} chars")
                            except Exception as e:
                                logger.warning(f"pdfplumber fallback failed: {e}")
                                pass
                except Exception as e:
                    # If pipeline fails, attempt direct text extraction then image OCR
                    logger.warning(f"Primary PDF OCR failed, attempting fallbacks: {e}")
                    import fitz
                    import numpy as np
                    import cv2
                    from PIL import Image
                    fallback_text = ""
                    try:
                        doc = fitz.open(tmp_path)
                        texts = []
                        for page in doc:
                            t = page.get_text("text")
                            if t:
                                texts.append(t)
                        if sum(len(t) for t in texts) < 50:
                            for page in doc:
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                                png_bytes = pix.tobytes("png")
                                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                                arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                                thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
                                res_a = ocr_pipeline.extract_text_with_confidence(arr)
                                res_b = ocr_pipeline.extract_text_with_confidence(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
                                res = res_a if len((res_a.get("text") or "")) >= len((res_b.get("text") or "")) else res_b
                                if res.get("text"):
                                    texts.append(res["text"])
                        doc.close()
                        fallback_text = " ".join(texts).strip()
                    except Exception:
                        pass
                    if fallback_text:
                        ocr_result = {"text": fallback_text, "tokens": [], "confidence": 0.6}
                    else:
                        try:
                            import pdfplumber
                            with pdfplumber.open(tmp_path) as pdf:
                                pl_texts = []
                                for page in pdf.pages:
                                    pl_texts.append(page.extract_text() or "")
                            pl_text = " ".join(pl_texts).strip()
                            if pl_text:
                                ocr_result = {"text": pl_text, "tokens": [], "confidence": 0.6}
                            else:
                                ocr_result = {"text": "", "tokens": [], "confidence": 0.0}
                        except Exception:
                            ocr_result = {"text": "", "tokens": [], "confidence": 0.0}
                finally:
                    # Clean up temp file
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            ocr_result = {"text": "", "tokens": [], "confidence": 0.0}

        # Check minimal text
        if not ocr_result.get('text') or len(ocr_result['text'].strip()) < 5:
            raise HTTPException(status_code=422, detail="Could not extract text")
            
        # 2. PATTERN-BASED EXTRACTION (To get Bank Name first)
        pattern_results = extract_all_fields_v3(ocr_result.get('text', ''))
        bank_name = pattern_results.get('bank_name', 'Unknown')

        # 3. LAYOUT-AWARE EXTRACTION (Prioritize layout, use patterns as fallback)
        logger.info(f"Extracting layout... (Tokens: {len(ocr_result.get('tokens', []))})")
        layout_results = layout_extractor.extract(ocr_result, bank_name)
        
        logger.info(f"Layout extraction success: {layout_results.get('success')}")
        
        # 4. MERGE RESULTS
        # Layout extractor is the authority on Reference IDs
        # Pattern matcher is the authority on Date/Amount/Bank (if layout fails)
        
        final_id = layout_results.get("reference_id") if layout_results.get("success") else pattern_results.get("transaction_id")
        final_bank = layout_results.get("bank_name") if layout_results.get("bank_name") != "Unknown" else pattern_results.get("bank_name")
        
        results = {
            "bank": final_bank,
            "bank_name": final_bank,
            "transaction_id": final_id,
            "reference_number": final_id,
            "all_ids": layout_results.get("candidates", [final_id]) if layout_results.get("success") else [final_id],
            "date": pattern_results.get("date"),
            "amount": pattern_results.get("amount"),
            "ocr_confidence": ocr_result.get("confidence", 0.5),
            "method": "layout_aware_local" if layout_results.get("success") else "pattern_fallback",
            "raw_text": ocr_result.get("text", "")
        }
        
        logger.info(f"Extraction completed: {results}")
        
        # 5. SAVE TO HISTORY
        entry = {
            "id": file_id,
            "filename": file.filename,
            "bank_name": results["bank_name"],
            "reference_id": results["transaction_id"],
            "amount": results["amount"],
            "date": results["date"],
            "confidence": results["ocr_confidence"] * 100
        }
        entry_id = history_manager.add_entry(entry)
        results["entry_id"] = entry_id

        # 6. SAVE TO SUPABASE
        if db:
            try:
                # Format entry for Supabase
                supabase_entry = {
                    "id": file_id,
                    "bank": {"name": results["bank_name"]},
                    "fields": {
                        "transaction_id": results["transaction_id"],
                        "reference_number": results["reference_number"],
                        "amount": results["amount"],
                        "date": results["date"]
                    },
                    "confidence": results["ocr_confidence"] * 100,
                    "filename": file.filename,
                    "timestamp": datetime.now().isoformat()
                }
                db.append_annotation(supabase_entry)
                logger.info(f"Saved to Supabase: {file_id}")
            except Exception as e:
                logger.error(f"Failed to save to Supabase: {e}")

        return {
            "success": True,
            "data": results,
            "extraction": results,
            "message": "Extraction completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/history")
async def get_history():
    """Get extraction history"""
    return {"success": True, "history": history_manager.get_all()}

@app.post("/history/update/{entry_id}")
async def update_history(entry_id: str, request: Request):
    """Update a history entry"""
    updates = await request.json()
    success = history_manager.update_entry(entry_id, updates)
    return {"success": success}

@app.get("/export")
async def export_history():
    """Export history to CSV"""
    history = history_manager.get_all()
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "filename", "bank_name", "reference_id", "amount", "date", "status"])
    writer.writeheader()
    
    for entry in history:
        writer.writerow({
            "timestamp": entry.get("timestamp"),
            "filename": entry.get("filename"),
            "bank_name": entry.get("bank_name"),
            "reference_id": entry.get("reference_id"),
            "amount": entry.get("amount"),
            "date": entry.get("date"),
            "status": entry.get("status")
        })
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=extraction_history.csv"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Bank Receipt Extractor"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run("simple_server:app", host="0.0.0.0", port=port, reload=True)
