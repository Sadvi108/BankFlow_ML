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

# Initialize DB. Persistence is optional: SUPABASE_URL/KEY without the postgrest
# package raises at import, which would take the whole service down on boot.
try:
    db = get_db()
except Exception as e:
    db = None
    logger.warning(f"Supabase unavailable, continuing without persistence: {e}")

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
from app.ibg.extractor import extract_ibg_fields
from app import ocr_vision

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

    # 2b. Apple Vision. The pipeline above needs a `tesseract` binary; without
    #     it every scanned receipt failed with 422 and no text ever reached the
    #     extractors. Vision ships with macOS, so this recovers those uploads.
    if ocr_vision.available():
        try:
            vision_res = ocr_vision.pdf_to_text(tmp_path)
            text = (vision_res.get("text") or "").strip()
            if text and not is_text_garbage(text):
                logger.info("Apple Vision OCR recovered %d chars (conf %.2f)",
                            len(text), vision_res.get("confidence", 0.0))
                return {"text": text, "tokens": [],
                        "confidence": vision_res.get("confidence", 0.7),
                        "source": "vision"}
        except Exception as e:
            logger.warning(f"Apple Vision OCR failed on PDF: {e}")

    # 3. pdfplumber. The garbage check matters most HERE: this is the last
    #    fallback, and pdfplumber happily returns the CID codepoints of a
    #    font-subset PDF -- `(cid:3)(cid:9)...` -- which is far longer than the
    #    caller's 5-character guard. Without this check that garbage was
    #    accepted as real text, every extractor found nothing in it, and the
    #    upload returned success with all fields blank and no error shown.
    #    Falling through to "" instead makes the caller raise an honest 422.
    try:
        import pdfplumber
        with pdfplumber.open(tmp_path) as pdf:
            pl_text = " ".join((p.extract_text() or "") for p in pdf.pages).strip()
        if pl_text and not is_text_garbage(pl_text):
            return {"text": pl_text, "tokens": [], "confidence": 0.4, "source": "pdfplumber"}
        if pl_text:
            logger.warning("pdfplumber returned %d chars of unusable text "
                           "(CID-encoded or mojibake); discarding", len(pl_text))
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
    return templates.TemplateResponse(request, "simple_upload.html", {})

@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse(request, "train.html", {})

def _build_response(file_id: str, filename: str, merged: Dict[str, Any],
                    raw_text: str, ocr_confidence: float, llm_used: bool,
                    ibg: Optional[Dict[str, Any]] = None,
                    text_source: str = "unknown") -> Dict[str, Any]:
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
        # Which rung of the PDF ladder actually produced the text (embedded /
        # ocr / vision / pdfplumber). When fields come back empty this is the
        # first thing worth knowing, and it is not otherwise recoverable from
        # a deployed instance.
        "text_source": text_source,
    }

    # Everything the IBG engine found, added alongside the legacy keys rather
    # than replacing them, so existing consumers keep working unchanged.
    if ibg:
        results["references"] = ibg.get("references", [])
        results["references_by_role"] = ibg.get("references_by_role", {})
        results["reference_count"] = ibg.get("reference_count", 0)
        results["fee"] = ibg["fee"]["value"]
        results["total_debit"] = ibg["total_debit"]["value"]
        results["beneficiary_bank"] = ibg["beneficiary_bank"]["value"]
        results["payer"] = ibg["payer"]["value"]
        results["beneficiary"] = ibg["beneficiary"]["value"]
        results["payment_mode"] = ibg["payment_mode"]["value"]
        results["review_reasons"] = ibg.get("review_reasons", [])
        results["fields_absent"] = ibg.get("fields_absent", [])
        results["field_detail"] = dict(
            (k, ibg[k]) for k in
            ("reference_id", "bank_name", "beneficiary_bank",
             "transaction_date", "amount", "fee", "total_debit",
             "payer", "beneficiary", "payment_mode")
        )

    try:
        entry_id = history_manager.add_entry({
            "id": file_id,
            "filename": filename,
            "bank_name": results["bank_name"],
            "reference_id": results["transaction_id"],
            "amount": results["amount"],
            "date": results["date"],
            "confidence": round(results["overall_confidence"] * 100, 1),
            # Carried so the CSV export and the history view can show every
            # reference, not just the elected primary.
            "references": results.get("references", []),
            "fee": results.get("fee"),
            "total_debit": results.get("total_debit"),
            "beneficiary_bank": results.get("beneficiary_bank"),
            "needs_review": results.get("needs_review"),
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
                ocr_result.setdefault("source", "ocr")  # images always go through OCR
                # Same Tesseract-absent recovery as the PDF path: a photo of a
                # receipt is the most common upload and produced no text at all
                # without a tesseract binary.
                if len((ocr_result.get("text") or "").strip()) < 5 and ocr_vision.available():
                    vision_res = ocr_vision.ndarray_to_text(image)
                    if (vision_res.get("text") or "").strip():
                        logger.info("Apple Vision OCR recovered image text (conf %.2f)",
                                    vision_res.get("confidence", 0.0))
                        ocr_result = {"text": vision_res["text"], "tokens": [],
                                      "confidence": vision_res.get("confidence", 0.7),
                                      "source": "vision"}
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
        # Text lifted from a PDF's own text layer has no OCR misreads, so the
        # digit-repair pass must not run over it.
        ocr_used = ocr_result.get("source") not in ("embedded", "pdfplumber")
        pattern_results = extract_all_fields_v3(text, ocr_used=ocr_used)
        bank_name = pattern_results.get('bank_name', 'Unknown')

        # The IBG engine runs on every receipt. Where it resolves a field, it
        # wins: it ranks by which label introduced a value rather than by the
        # shape of the token, which is what stopped account numbers and payer
        # free-text being reported as the bank's reference. Where it finds
        # nothing, the legacy V3 answer stands, so non-interbank receipts are
        # unaffected.
        try:
            ibg_results = extract_ibg_fields(text, ocr_used=ocr_used)
        except Exception as e:
            logger.error(f"IBG extraction failed, falling back to V3: {e}",
                         exc_info=True)
            ibg_results = None

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

        if ibg_results:
            # The reference is AUTHORITATIVE, including when it is None.
            # The legacy engine ranks candidates by token shape, which is
            # precisely what let a debit account number be reported as the
            # transaction reference. If the label-driven engine found no
            # bank-issued reference, the honest answer is "none on this
            # document" -- not the best-looking number on the page.
            local_dict["transaction_id"] = ibg_results["reference_id"]["value"]

            # For the other fields a missing IBG answer means "did not
            # resolve", so the legacy value is still worth keeping.
            for legacy_key, ibg_key in (("bank_name", "bank_name"),
                                        ("date", "transaction_date"),
                                        ("amount", "amount")):
                if ibg_results[ibg_key]["value"] is not None:
                    local_dict[legacy_key] = ibg_results[ibg_key]["value"]

        # 3. Validate locally (no API). Merger flags needs_review when fields fail.
        merged = merge_results(local_dict, None)
        llm_used = False
        # carry layout candidates for transparency
        merged["all_ids"] = (
            layout_results.get("candidates")
            if layout_results.get("success") else [merged["transaction_id"]]
        ) or [merged["transaction_id"]]

        # The IBG engine knows why it is unsure; prefer its verdict.
        if ibg_results:
            merged["needs_review"] = ibg_results["needs_review"]
            merged["overall_confidence"] = ibg_results["overall_confidence"]

        return _build_response(
            file_id, file.filename, merged,
            raw_text=text,
            ocr_confidence=ocr_result.get("confidence", 0.5),
            llm_used=llm_used,
            ibg=ibg_results,
            text_source=ocr_result.get("source", "unknown"),
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
    # Existing columns kept in place and order so anything already consuming
    # this file keeps working; the reference and money breakdown is appended.
    writer = csv.DictWriter(output, fieldnames=[
        "timestamp", "filename", "bank_name", "reference_id", "amount", "date",
        "status", "fee", "total_debit", "beneficiary_bank",
        "bank_references", "payer_references", "needs_review",
    ])
    writer.writeheader()

    for entry in history:
        refs = entry.get("references") or []
        bank_refs = [r["value"] for r in refs
                     if r.get("role", "").startswith("bank_")]
        payer_refs = [r["value"] for r in refs
                      if r.get("role") == "payer_supplied"]
        writer.writerow({
            "timestamp": entry.get("timestamp"),
            "filename": entry.get("filename"),
            "bank_name": entry.get("bank_name"),
            "reference_id": entry.get("reference_id"),
            "amount": entry.get("amount"),
            "date": entry.get("date"),
            "status": entry.get("status"),
            "fee": entry.get("fee"),
            "total_debit": entry.get("total_debit"),
            "beneficiary_bank": entry.get("beneficiary_bank"),
            "bank_references": " | ".join(bank_refs),
            "payer_references": " | ".join(payer_refs),
            "needs_review": entry.get("needs_review"),
        })
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=extraction_history.csv"}
    )

def _ocr_diagnostics() -> Dict[str, Any]:
    """What the OCR stack actually looks like from inside the container.

    A missing `tesseract` binary and an OOM-killed worker both surface in the
    browser as a failed upload with no useful message, and neither is
    reproducible locally. Render's free instances have no shell, so this is the
    only way to tell the two apart. Every probe is individually guarded --
    /health is the deploy's health check and must never raise.
    """
    import shutil

    diag: Dict[str, Any] = {}

    diag["tesseract_path"] = shutil.which("tesseract")
    try:
        import pytesseract
        diag["tesseract_version"] = str(pytesseract.get_tesseract_version())
    except Exception as e:
        diag["tesseract_version"] = f"unavailable: {e}"
    try:
        import pytesseract
        diag["languages"] = pytesseract.get_languages(config="")
    except Exception as e:
        diag["languages"] = f"unavailable: {e}"

    try:
        diag["vision_available"] = ocr_vision.available()
    except Exception as e:
        diag["vision_available"] = f"unavailable: {e}"

    # Resident size vs the cgroup limit. Three sequential Tesseract passes over
    # a full-resolution phone photo is the memory peak in this app; if headroom
    # here is small, the worker is being killed mid-upload rather than erroring.
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    diag["rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                    break
    except Exception:
        diag["rss_mb"] = None

    for limit_file in ("/sys/fs/cgroup/memory.max",
                       "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(limit_file) as f:
                raw = f.read().strip()
            if raw and raw != "max":
                diag["memory_limit_mb"] = round(int(raw) / 1024 / 1024, 1)
                break
        except Exception:
            continue

    return diag


@app.get("/health")
async def health_check():
    """Health check endpoint (also the Render healthCheckPath)."""
    payload = {"status": "healthy", "service": "Bank Receipt Extractor"}
    try:
        payload["ocr"] = _ocr_diagnostics()
    except Exception as e:  # never let diagnostics fail the health check
        payload["ocr"] = {"error": str(e)}
    return payload

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run("simple_server:app", host="0.0.0.0", port=port, reload=True)
