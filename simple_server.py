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
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline, is_text_garbage
from app.layout_aware_extractor import layout_extractor
from app.result_merger import merge as merge_results

# Initialize enhanced OCR pipeline
ocr_pipeline = EnhancedOCRPipeline()


def _pdf_to_text(tmp_path: str) -> Dict[str, Any]:
    """Single source of truth for PDF -> text. Tries:
       1. fitz embedded text (rejected if CID-encoded / garbage)
       2. Enhanced OCR pipeline (rasterize + Tesseract)
       3. pdfplumber as final fallback.
    """
    # 1. Embedded text
    try:
        import fitz
        doc = fitz.open(tmp_path)
        parts = [page.get_text("text") for page in doc]
        doc.close()
        embedded = " ".join(p for p in parts if p).strip()
        if embedded and not is_text_garbage(embedded):
            return {"text": embedded, "tokens": [], "confidence": 0.7, "source": "embedded"}
    except Exception as e:
        logger.warning(f"fitz embedded text failed: {e}")

    # 2. Rasterize + OCR via enhanced pipeline
    try:
        ocr_res = ocr_pipeline.process_file(tmp_path)
        text = (ocr_res.get("text") or "").strip()
        if text and not is_text_garbage(text):
            conf = ocr_res.get("confidence", 0)
            if conf > 1:  # pipeline returns 0-100
                conf = conf / 100.0
            return {"text": text, "tokens": ocr_res.get("tokens", []),
                    "confidence": conf, "source": ocr_res.get("method", "ocr")}
    except Exception as e:
        logger.warning(f"OCR pipeline failed on PDF: {e}")

    # 3. pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(tmp_path) as pdf:
            pl_text = " ".join((p.extract_text() or "") for p in pdf.pages).strip()
        if pl_text:
            return {"text": pl_text, "tokens": [], "confidence": 0.4, "source": "pdfplumber"}
    except Exception as e:
        logger.warning(f"pdfplumber fallback failed: {e}")

    return {"text": "", "tokens": [], "confidence": 0.0, "source": "none"}


def _render_pdf_page_to_png(pdf_bytes: bytes, page_index: int = 0, zoom: float = 2.0) -> Optional[bytes]:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) <= page_index:
            doc.close()
            return None
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png = pix.tobytes("png")
        doc.close()
        return png
    except Exception as e:
        logger.warning(f"PDF render to PNG failed: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the simplified upload page"""
    return templates.TemplateResponse("simple_upload.html", {"request": request})

@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

def _build_response(file_id: str, filename: str, merged: Dict[str, Any],
                    raw_text: str, ocr_confidence: float, llm_used: bool) -> Dict[str, Any]:
    """Persist to history/Supabase, build the JSON response shape the UI expects."""
    results = {
        "bank": merged["bank_name"],
        "bank_name": merged["bank_name"],
        "transaction_id": merged["transaction_id"],
        "reference_number": merged["reference_number"],
        "all_ids": merged.get("all_ids") or [merged["transaction_id"]],
        "date": merged["date"],
        "amount": merged["amount"],
        "ocr_confidence": ocr_confidence,
        "overall_confidence": merged["overall_confidence"],
        "field_confidence": merged["field_confidence"],
        "field_source": merged["field_source"],
        "method": merged["method"],
        "needs_review": merged["needs_review"],
        "llm_used": llm_used,
        "raw_text": raw_text,
    }

    try:
        entry_id = history_manager.add_entry({
            "id": file_id,
            "filename": filename,
            "bank_name": results["bank_name"],
            "reference_id": results["transaction_id"],
            "amount": results["amount"],
            "date": results["date"],
            "confidence": round(results["overall_confidence"] * 100, 1),
        })
        results["entry_id"] = entry_id
    except Exception as e:
        logger.warning(f"History save failed: {e}")

    if db:
        try:
            db.append_annotation({
                "id": file_id,
                "bank": {"name": results["bank_name"]},
                "fields": {
                    "transaction_id": results["transaction_id"],
                    "reference_number": results["reference_number"],
                    "amount": results["amount"],
                    "date": results["date"],
                },
                "confidence": round(results["overall_confidence"] * 100, 1),
                "method": results["method"],
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.warning(f"Supabase save failed: {e}")

    return {
        "success": True,
        "data": results,
        "extraction": results,
        "message": "Extraction completed",
    }


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
        
        # 1. OCR — single path for images, helper for PDFs
        is_image = content_type.startswith("image/") or (
            content_type == "application/octet-stream" and file_ext in {".jpg", ".jpeg", ".png"}
        )
        image_png_for_llm: Optional[bytes] = None  # bytes to send to Gemini if needed

        try:
            if is_image:
                import cv2
                import numpy as np
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                ocr_result = ocr_pipeline.extract_text_with_confidence(image)
                image_png_for_llm = content  # raw image bytes are LLM-ready
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    ocr_result = _pdf_to_text(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            ocr_result = {"text": "", "tokens": [], "confidence": 0.0}

        if not ocr_result.get('text') or len(ocr_result['text'].strip()) < 5:
            raise HTTPException(status_code=422, detail="Could not extract text from receipt")

        # 2. PATTERN + LAYOUT extraction
        text = ocr_result.get('text', '')
        pattern_results = extract_all_fields_v3(text)
        bank_name = pattern_results.get('bank_name', 'Unknown')

        layout_results = layout_extractor.extract(ocr_result, bank_name)
        logger.info(f"Layout success={layout_results.get('success')} bank={bank_name}")

        local_id = layout_results.get("reference_id") if layout_results.get("success") else pattern_results.get("transaction_id")
        local_bank = (
            layout_results.get("bank_name")
            if layout_results.get("success") and layout_results.get("bank_name") not in (None, "Unknown")
            else pattern_results.get("bank_name", "Unknown")
        )
        local_dict = {
            "bank_name": local_bank,
            "transaction_id": local_id,
            "date": pattern_results.get("date"),
            "amount": pattern_results.get("amount"),
        }

        # 3. Validate locally (no API). Merger flags needs_review when fields fail.
        merged = merge_results(local_dict, None)
        llm_used = False
        # carry layout candidates for transparency
        merged["all_ids"] = (
            layout_results.get("candidates")
            if layout_results.get("success") else [merged["transaction_id"]]
        ) or [merged["transaction_id"]]

        return _build_response(
            file_id, file.filename, merged,
            raw_text=text,
            ocr_confidence=ocr_result.get("confidence", 0.5),
            llm_used=llm_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
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
