from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid

from . import utils, ocr, classify, extract, dataset


# Initialize app and templates
app = FastAPI(title="Bank Receipt Extractor (MVP)")
templates = Jinja2Templates(directory=str(Path("templates")))


# Ensure data directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_DIR = DATA_DIR / "dataset"
DATASET_IMG_DIR = DATASET_DIR / "images"
for d in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
dataset.ensure_dirs()


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/extract")
async def extract_fields(file: UploadFile = File(...)):
    # Save uploaded file
    suffix = Path(file.filename).suffix.lower()
    file_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{file_id}{suffix or ''}"
    content = await file.read()
    saved_path.write_bytes(content)

    # Convert to image (first page if PDF)
    text = ""
    tokens = []
    processed_path = None
    try:
        if suffix == ".pdf" or (file.content_type and "pdf" in file.content_type.lower()):
            # First try to extract text directly from PDF (no OCR)
            text, tokens = ocr.pdf_to_text_and_tokens(saved_path)
            if not text.strip():
                # Fallback: render to image and OCR
                image = utils.pdf_to_image(saved_path)
                image = ocr.auto_rotate(image)
                image = utils.preprocess_image(image)
                processed_path = PROCESSED_DIR / f"{file_id}.png"
                image.save(processed_path)
                text, tokens = ocr.image_to_text_and_tokens(image)
        else:
            image = utils.load_image(saved_path)
            image = ocr.auto_rotate(image)
            image = utils.preprocess_image(image)
            processed_path = PROCESSED_DIR / f"{file_id}.png"
            image.save(processed_path)
            text, tokens = ocr.image_to_text_and_tokens(image)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to process file: {str(e)}"}
        )

    # Bank classification
    bank_name, bank_conf = classify.bank_from_text(text)

    # Field extraction
    fields = extract.extract_fields(text=text, tokens=tokens, bank_hint=bank_name)

    result: Dict[str, Any] = {
        "bank": {"name": bank_name, "confidence": bank_conf},
        "fields": fields,
        "meta": {
            "filename": file.filename,
            "saved_path": str(saved_path),
            "processed_path": str(processed_path) if processed_path else None,
            "ocr_tokens": len(tokens),
        },
    }

    return JSONResponse(content=result)


@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})


@app.post("/upload_bulk")
async def upload_bulk(files: List[UploadFile] = File(...)):
    results = []
    errors = []
    for file in files:
        try:
            suffix = Path(file.filename).suffix.lower()
            file_id = uuid.uuid4().hex
            saved_path = UPLOAD_DIR / f"{file_id}{suffix or ''}"
            content = await file.read()
            saved_path.write_bytes(content)

            # Convert to image
            if suffix == ".pdf" or (file.content_type and "pdf" in file.content_type.lower()):
                image = utils.pdf_to_image(saved_path)
            else:
                image = utils.load_image(saved_path)

            # Preprocess
            image = ocr.auto_rotate(image)
            image = utils.preprocess_image(image)
            processed_path = PROCESSED_DIR / f"{file_id}.png"
            image.save(processed_path)

            # OCR
            text, tokens = ocr.image_to_text_and_tokens(image)

            # Classify and extract
            bank_name, bank_conf = classify.bank_from_text(text)
            fields = extract.extract_fields(text=text, tokens=tokens, bank_hint=bank_name)

            entry: Dict[str, Any] = {
                "id": file_id,
                "filename": file.filename,
                "bank": {"name": bank_name, "confidence": bank_conf},
                "fields": fields,
                "ocr_text": text,
                "meta": {
                    "saved_path": str(saved_path),
                    "processed_path": str(processed_path),
                    "ocr_tokens": len(tokens),
                },
                "ground_truth": {
                    "bank_name": None,
                    "reference_number": None,
                    "transaction_id": None,
                    "invoice_number": None,
                    "amount": None,
                    "date": None,
                },
            }
            dataset.append_annotation(entry)
            results.append({"id": file_id, "filename": file.filename, "bank": bank_name, "fields": fields})
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    return JSONResponse(content={"uploaded": len(results), "errors": errors, "items": results})


@app.get("/dataset/summary")
async def dataset_summary():
    return JSONResponse(content=dataset.summary())


@app.get("/dataset/items")
async def dataset_items(limit: Optional[int] = 100):
    rows = dataset.read_annotations()
    if limit:
        rows = rows[-limit:]
    return JSONResponse(content={"count": len(rows), "items": rows})


@app.post("/dataset/update")
async def dataset_update(payload: Dict[str, Any]):
    item_id = payload.get("id")
    if not item_id:
        return JSONResponse(status_code=400, content={"error": "Missing 'id' in payload"})
    gt_updates = {
        "ground_truth": {
            "bank_name": payload.get("bank_name"),
            "reference_number": payload.get("reference_number"),
            "transaction_id": payload.get("transaction_id"),
            "invoice_number": payload.get("invoice_number"),
            "amount": payload.get("amount"),
            "date": payload.get("date"),
        }
    }
    ok = dataset.update_annotation(item_id, gt_updates)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Annotation id not found"})
    return JSONResponse(content={"updated": True, "id": item_id})


@app.post("/ingest_receipts")
async def ingest_receipts(path: Optional[str] = None):
    """Process all files in the local Receipts folder and append to dataset."""
    base_dir = BASE_DIR
    receipts_dir = Path(path) if path else (base_dir / "Receipts")
    if not receipts_dir.exists() or not receipts_dir.is_dir():
        return JSONResponse(status_code=404, content={"error": f"Receipts folder not found: {receipts_dir}"})

    supported_exts = {".pdf", ".png", ".jpg", ".jpeg"}
    files = [p for p in receipts_dir.iterdir() if p.suffix.lower() in supported_exts]
    processed = []
    errors = []
    for p in files:
        try:
            file_id = uuid.uuid4().hex
            text = ""
            tokens = []
            processed_path = None
            dataset_img_path = None
            if p.suffix.lower() == ".pdf":
                # First try direct PDF text extraction
                text, tokens = ocr.pdf_to_text_and_tokens(p)
                if not text.strip():
                    # Fallback to rendering and OCR
                    image = utils.pdf_to_image(p)
                    image = ocr.auto_rotate(image)
                    image = utils.preprocess_image(image)
                    processed_path = PROCESSED_DIR / f"{file_id}.png"
                    image.save(processed_path)
                    dataset_img_path = DATASET_IMG_DIR / f"{file_id}.png"
                    dataset_img_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(dataset_img_path)
                    text, tokens = ocr.image_to_text_and_tokens(image)
            else:
                image = utils.load_image(p)
                image = ocr.auto_rotate(image)
                image = utils.preprocess_image(image)
                processed_path = PROCESSED_DIR / f"{file_id}.png"
                image.save(processed_path)
                dataset_img_path = DATASET_IMG_DIR / f"{file_id}.png"
                dataset_img_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(dataset_img_path)
                text, tokens = ocr.image_to_text_and_tokens(image)

            # Classify and extract
            bank_name, bank_conf = classify.bank_from_text(text)
            fields = extract.extract_fields(text=text, tokens=tokens, bank_hint=bank_name)

            entry: Dict[str, Any] = {
                "id": file_id,
                "filename": p.name,
                "bank": {"name": bank_name, "confidence": bank_conf},
                "fields": fields,
                "ocr_text": text,
                "meta": {
                    "source_path": str(p),
                    "processed_path": str(processed_path) if processed_path else None,
                    "dataset_image_path": str(dataset_img_path) if dataset_img_path else None,
                    "ocr_tokens": len(tokens),
                },
                "ground_truth": {
                    "bank_name": None,
                    "reference_number": None,
                    "transaction_id": None,
                    "invoice_number": None,
                    "amount": None,
                    "date": None,
                },
            }
            dataset.append_annotation(entry)
            processed.append({"id": file_id, "filename": p.name, "bank": bank_name, "fields": fields})
        except Exception as e:
            errors.append({"filename": p.name, "error": str(e)})

    return JSONResponse(content={"processed": len(processed), "errors": errors, "items": processed})