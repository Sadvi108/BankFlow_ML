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
import time

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
    """Extract every PDF page once, OCRing only pages without usable text.

    A PDF may be fully digital, fully scanned, or a mixture of both.  The old
    aggregate check returned as soon as *any* usable embedded text existed,
    silently dropping scanned pages.  On fully scanned documents it then tried
    pdfplumber and an unavailable Tesseract executable before repeating every
    page through Apple Vision.  Page-level routing fixes both the missing-ID
    and latency failure modes.
    """
    embedded_pages = {}
    scan_pages = []
    page_count = 0

    # Inspect the text layer once and classify each page independently.
    try:
        import fitz
        doc = fitz.open(tmp_path)
        try:
            page_count = len(doc)
            for index in range(page_count):
                page = doc.load_page(index)
                page_text = (page.get_text("text") or "").strip()
                if not is_text_garbage(page_text):
                    embedded_pages[index] = page_text
                elif page_text or page.get_images(full=True) or page.get_drawings():
                    # Non-empty garbage includes CID-encoded text. An empty
                    # page needs OCR when it contains an image or outlined
                    # vector glyphs. Some banking portals export every letter
                    # as a drawing, so an image-only check loses the whole ID.
                    scan_pages.append(index)
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"fitz embedded text failed: {e}")

    def _joined(page_map):
        return "\n".join(
            "=== PAGE %d ===\n%s" % (index + 1, page_map[index])
            for index in sorted(page_map) if page_map[index].strip()
        ).strip()

    if embedded_pages and not scan_pages:
        return {
            "text": _joined(embedded_pages),
            "tokens": [],
            "confidence": 0.7,
            "source": "embedded",
            "pages_processed": len(embedded_pages),
            "page_count": page_count,
        }

    ocr_result = None
    ocr_source = None
    ocr_pages = {}
    core_ocr = ocr_pipeline.ocr_pipeline

    def _ocr_page_map(candidate, targets):
        page_texts = candidate.get("page_texts") or []
        if page_texts:
            return {
                int(item["page"]) - 1: item["text"].strip()
                for item in page_texts
                if item.get("page") is not None
                and int(item["page"]) - 1 in targets
                and not is_text_garbage(item.get("text"))
            }
        aggregate = re.sub(
            r"(?m)^=== PAGE \d+ ===$", "", candidate.get("text") or "")
        if len(targets) == 1 and not is_text_garbage(aggregate):
            return {targets[0]: aggregate.strip()}
        return {}

    # Use Tesseract only when its external executable exists. Preprocessing a
    # large page twice before discovering that it is absent was a major source
    # of slow scanned-receipt requests on macOS deployments.
    if scan_pages and core_ocr.tesseract_available():
        try:
            candidate = core_ocr.process_pdf_pages(tmp_path, scan_pages)
            ocr_pages = _ocr_page_map(candidate, scan_pages)
            if ocr_pages:
                ocr_result = candidate
                ocr_source = "ocr"
        except Exception as e:
            logger.warning(f"Tesseract OCR failed on PDF pages: {e}")

    # Retry only unread pages, even when Tesseract recovered some of a PDF.
    # One successful page must not suppress recovery of a later page's IDs.
    pending_pages = [index for index in scan_pages if index not in ocr_pages]
    if pending_pages and ocr_vision.available():
        try:
            candidate = ocr_vision.pdf_to_text(
                tmp_path, page_indices=pending_pages)
            recovered = _ocr_page_map(candidate, pending_pages)
            if recovered:
                ocr_pages.update(recovered)
                if ocr_result is None:
                    ocr_result = candidate
                    ocr_source = "vision"
                else:
                    ocr_source = "ocr_vision"
        except Exception as e:
            logger.warning(f"Apple Vision OCR failed on PDF pages: {e}")

    if ocr_result is not None:
        combined_pages = dict(embedded_pages)
        combined_pages.update(ocr_pages)
        combined = _joined(combined_pages)
        if combined and not is_text_garbage(combined):
            conf = ocr_result.get("confidence", 0.0)
            if conf > 1:
                conf = conf / 100.0
            unread_pages = [index + 1 for index in scan_pages
                            if index not in ocr_pages]
            return {
                "text": combined,
                "tokens": ocr_result.get("tokens", []),
                "confidence": conf,
                "source": ("hybrid_%s" % ocr_source
                           if embedded_pages else ocr_source),
                "pages_processed": len(combined_pages),
                "page_count": page_count,
                "ocr_pages": [index + 1 for index in scan_pages],
                "unread_pages": unread_pages,
                "page_methods": ocr_result.get("page_methods", []),
                "page_passes": ocr_result.get("page_passes", []),
                "passes_used": ocr_result.get("passes_used"),
                "processing_time_ms": ocr_result.get("processing_time_ms"),
            }

    # Preserve readable digital pages even if OCR could not recover a scanned
    # companion page. The response carries the exact unread page numbers so it
    # cannot masquerade as a complete extraction downstream.
    if embedded_pages:
        return {
            "text": _joined(embedded_pages),
            "tokens": [],
            "confidence": 0.5,
            "source": "embedded_partial",
            "pages_processed": len(embedded_pages),
            "page_count": page_count,
            "ocr_pages": [index + 1 for index in scan_pages],
            "unread_pages": [index + 1 for index in scan_pages],
        }

    # Final fallback for malformed PDFs that fitz could not inspect. The
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
                    text_source: str = "unknown",
                    processing_time_ms: Optional[float] = None,
                    timings: Optional[Dict[str, float]] = None,
                    ocr_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist to history/Supabase, build the JSON response shape the UI expects."""
    def _unique_values(values):
        unique = []
        seen = set()
        for value in values:
            # Layout candidates are evidence objects ({id, score}); the portal
            # contract is intentionally an array of identifier strings.
            if isinstance(value, dict):
                value = value.get("id") or value.get("value") or value.get("text")
            value = str(value).strip() if value is not None else ""
            # The legacy scalar removes spaces and uppercases an ID. Do not
            # count its original labelled spelling as a second reference.
            key = re.sub(r"\s+", "", value).upper()
            if value and key not in seen:
                seen.add(key)
                unique.append(value)
        return unique

    initial_ids = _unique_values(
        [merged.get("transaction_id")] + list(merged.get("all_ids") or []))
    results = {
        "bank": merged["bank_name"],
        "bank_name": merged["bank_name"],
        "transaction_id": merged["transaction_id"],
        "reference_number": merged["reference_number"],
        "primary_reference_id": merged["transaction_id"],
        "all_ids": initial_ids,
        # Explicit names for portal consumers. `all_ids` remains as a backwards
        # compatible alias, while these make it impossible to mistake a scalar
        # primary for the complete reference set.
        "reference_ids": initial_ids,
        "all_reference_ids": initial_ids,
        "bank_reference_ids": _unique_values([merged.get("transaction_id")]),
        "payer_reference_ids": [],
        "reference_count": len(initial_ids),
        "references": [],
        "references_by_role": {
            "bank_primary": [],
            "bank_secondary": [],
            "payer_supplied": [],
        },
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
        "processing_time_ms": processing_time_ms,
        "timings": timings or {},
        "ocr_details": ocr_details or {},
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

        # Populate receipt_ref / other_ref / customer_ref for downstream ERP / company portals
        all_ref_objs = ibg.get("references", [])
        receipt_ref = None
        other_ref = None
        customer_ref = None

        # 1. Customer Ref
        for r in all_ref_objs:
            lbl = r.get("label", "").lower()
            if not customer_ref and "customer" in lbl:
                customer_ref = r.get("value")

        # 2. Receipt / Invoice / Table Remittance Ref
        for r in all_ref_objs:
            lbl = r.get("label", "").lower()
            val = r.get("value")
            if val == merged["transaction_id"] or any(x in lbl for x in ("utr", "paynet", "duitnow", "rpp", "channel", "batch")):
                continue
            if not receipt_ref and any(k in lbl for k in ("receipt", "invoice", "document", "remittance", "reference no", "reference", "bill of lading", "b/l")):
                receipt_ref = val

        # 3. Other / Payer / Payment Details Ref
        for r in all_ref_objs:
            lbl = r.get("label", "").lower()
            val = r.get("value")
            if val == merged["transaction_id"] or val == receipt_ref:
                continue
            if not other_ref and any(k in lbl for k in ("other", "payment details", "recipient", "remark", "2nd party", "your reference", "customer")):
                other_ref = val

        # Fallbacks
        non_primary = [r.get("value") for r in all_ref_objs if r.get("value") != merged["transaction_id"]]
        if not receipt_ref and non_primary:
            receipt_ref = non_primary[0]
        if not other_ref and customer_ref and customer_ref != receipt_ref:
            other_ref = customer_ref
        elif not other_ref and len(non_primary) > 1:
            for cand in non_primary:
                if cand != receipt_ref:
                    other_ref = cand
                    break

        results["receipt_ref"] = receipt_ref
        results["receipt_reference"] = receipt_ref
        results["other_ref"] = other_ref
        results["other_reference"] = other_ref
        results["customer_ref"] = customer_ref
        results["customer_reference"] = customer_ref

        all_ids_list = _unique_values(
            [merged.get("transaction_id")]
            + [r.get("value") for r in all_ref_objs])
        bank_ids = _unique_values(
            [merged.get("transaction_id")]
            + [r.get("value") for r in all_ref_objs
               if r.get("role") in ("bank_primary", "bank_secondary")])
        payer_ids = _unique_values(
            [r.get("value") for r in all_ref_objs
             if r.get("role") == "payer_supplied"])
        results["all_ids"] = all_ids_list
        results["reference_ids"] = all_ids_list
        results["all_reference_ids"] = all_ids_list
        results["bank_reference_ids"] = bank_ids
        results["payer_reference_ids"] = payer_ids
        results["reference_count"] = len(all_ids_list)

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
    request_started = time.perf_counter()
    timings: Dict[str, float] = {}
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
        timings["upload_ms"] = round(
            (time.perf_counter() - request_started) * 1000, 1)
        
        logger.info(f"File saved: {file_path}")
        
        # 1. OCR — single path for images, helper for PDFs
        is_image = content_type.startswith("image/") or (
            content_type == "application/octet-stream" and file_ext in {".jpg", ".jpeg", ".png"}
        )

        ocr_started = time.perf_counter()
        try:
            if is_image:
                import cv2
                import numpy as np
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if (not ocr_pipeline.ocr_pipeline.tesseract_available()
                        and ocr_vision.available()):
                    # Do not spend two full preprocessing passes merely to
                    # discover that pytesseract has no executable to invoke.
                    vision_res = ocr_vision.ndarray_to_text(image)
                    ocr_result = {"text": vision_res.get("text", ""),
                                  "tokens": [],
                                  "confidence": vision_res.get("confidence", 0.0),
                                  "source": "vision"}
                else:
                    ocr_result = ocr_pipeline.extract_text_with_confidence(image)
                    ocr_result.setdefault("source", "ocr")
                    if (len((ocr_result.get("text") or "").strip()) < 5
                            and ocr_vision.available()):
                        vision_res = ocr_vision.ndarray_to_text(image)
                        if (vision_res.get("text") or "").strip():
                            logger.info("Apple Vision OCR recovered image text (conf %.2f)",
                                        vision_res.get("confidence", 0.0))
                            ocr_result = {"text": vision_res["text"], "tokens": [],
                                          "confidence": vision_res.get("confidence", 0.7),
                                          "source": "vision"}
            else:
                # The upload was already written to ``file_path`` above. The
                # old path wrote the same PDF to a second temporary file before
                # opening it, adding I/O and storage pressure to every request.
                ocr_result = _pdf_to_text(str(file_path))
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            ocr_result = {"text": "", "tokens": [], "confidence": 0.0}
        timings["ocr_ms"] = round(
            (time.perf_counter() - ocr_started) * 1000, 1)

        if not ocr_result.get('text') or len(ocr_result['text'].strip()) < 5:
            raise HTTPException(status_code=422, detail="Could not extract text from receipt")

        # 2. PATTERN + LAYOUT extraction
        extraction_started = time.perf_counter()
        text = ocr_result.get('text', '')
        # Text lifted from a PDF's own text layer has no OCR misreads, so the
        # digit-repair pass must not run over it.
        ocr_used = ocr_result.get("source") not in (
            "embedded", "embedded_partial", "pdfplumber",
            "pdf_text_extraction")
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
            [candidate.get("id") for candidate in
             layout_results.get("candidates", []) if candidate.get("id")]
            if layout_results.get("success") else [merged["transaction_id"]]
        ) or [merged["transaction_id"]]

        # The IBG engine knows why it is unsure; prefer its verdict.
        if ibg_results:
            merged["needs_review"] = ibg_results["needs_review"]
            merged["overall_confidence"] = ibg_results["overall_confidence"]
            if not ibg_results.get("references"):
                merged["needs_review"] = True
                ibg_results.setdefault("review_reasons", []).append(
                    "No reference could be read from this document")
        if ocr_result.get("unread_pages"):
            merged["needs_review"] = True
            if ibg_results is not None:
                ibg_results.setdefault("review_reasons", []).append(
                    "OCR could not read PDF page(s): %s" % ", ".join(
                        str(page) for page in ocr_result["unread_pages"]))
        if ibg_results and ocr_used:
            reference_keys = [re.sub(r"[^A-Z0-9]", "", ref["value"].upper())
                              for ref in ibg_results.get("references", [])]
            uncertain_reference = any(
                len(key) >= 3 and any(key in ref for ref in reference_keys)
                for token in ocr_result.get("tokens", [])
                if float(token.get("conf", 100)) < 40
                for key in [re.sub(r"[^A-Z0-9]", "", token.get("text", "").upper())]
            )
            if uncertain_reference:
                merged["needs_review"] = True
                ibg_results.setdefault("review_reasons", []).append(
                    "A reference contains text with low OCR confidence; verify its digits")

        timings["field_extraction_ms"] = round(
            (time.perf_counter() - extraction_started) * 1000, 1)
        timings["total_ms"] = round(
            (time.perf_counter() - request_started) * 1000, 1)
        ocr_details = {
            key: ocr_result.get(key) for key in
            ("method", "passes_used", "pages_processed", "page_methods",
             "page_passes", "processing_time_ms", "page_count", "ocr_pages",
             "unread_pages")
            if ocr_result.get(key) is not None
        }

        return _build_response(
            file_id, file.filename, merged,
            raw_text=text,
            ocr_confidence=ocr_result.get("confidence", 0.5),
            llm_used=llm_used,
            ibg=ibg_results,
            text_source=ocr_result.get("source", "unknown"),
            processing_time_ms=timings["total_ms"],
            timings=timings,
            ocr_details=ocr_details,
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
