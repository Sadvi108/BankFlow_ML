"""OCR via Apple's Vision framework — a drop-in when Tesseract is absent.

Scanned receipts have no text layer, so the pipeline falls back to OCR. That
fallback was `pytesseract`, which shells out to a `tesseract` binary; on a
machine without it (and without Homebrew to install it) every scanned upload
died with a 422 and no text ever reached the extractors.

macOS ships a text recogniser in the Vision framework. It needs no system
package and no admin rights — only the pyobjc bindings, which are pure pip —
and on receipt-shaped documents it is at least as accurate as Tesseract.

`available()` is the guard: this module is macOS-only, so every caller must
check before use and keep its existing path for other platforms.
"""
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VISION_IMPORT_ERROR = None
try:  # pragma: no cover - platform dependent
    import Quartz
    import Vision
    from Foundation import NSURL
    _HAVE_VISION = True
except Exception as exc:  # pragma: no cover - platform dependent
    _HAVE_VISION = False
    _VISION_IMPORT_ERROR = exc


def available() -> bool:
    """True when Apple Vision OCR can be used on this machine."""
    return _HAVE_VISION


def why_unavailable() -> Optional[str]:
    return None if _HAVE_VISION else str(_VISION_IMPORT_ERROR)


def _recognize(image_url, languages, fast: bool) -> List[Dict[str, Any]]:
    """Run one Vision text-recognition pass over an image URL."""
    source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    if source is None:
        return []
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg_image is None:
        return []

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None)

    results: List[Dict[str, Any]] = []

    request = Vision.VNRecognizeTextRequest.alloc().init()
    # Accurate is materially better on the small, dense type banks use for
    # reference numbers; the speed difference is irrelevant for one page.
    request.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelFast if fast
        else Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)  # ids/refs are not words
    if languages:
        try:
            request.setRecognitionLanguages_(languages)
        except Exception:
            pass

    success, error = handler.performRequests_error_([request], None)
    if not success:
        logger.warning("Vision OCR request failed: %s", error)
        return []

    for observation in (request.results() or []):
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        best = candidates[0]
        box = observation.boundingBox()
        results.append({
            "text": best.string(),
            "confidence": float(best.confidence()),
            # Vision's origin is bottom-left, normalised 0..1.
            "x": float(box.origin.x),
            "y": float(box.origin.y),
            "w": float(box.size.width),
            "h": float(box.size.height),
        })
    return results


def _lines_to_text(lines: List[Dict[str, Any]]) -> str:
    """Reassemble observations into reading order.

    Vision returns each recognised line separately with a normalised box.
    Sorting top-to-bottom then left-to-right rebuilds the page. Getting this
    wrong matters: the extractors key on the order labels and values appear in.
    """
    if not lines:
        return ""
    # Match by vertical centres, not bottom edges. A digit value often has a
    # slightly shorter box than its label; sorting bottoms alone could place
    # the value ABOVE the label and make the field look empty. Join aligned
    # columns on one row so their original label/value relationship survives.
    ordered = sorted(lines, key=lambda l: (-(l["y"] + l["h"] / 2), l["x"]))
    rows = []
    for line in ordered:
        centre = line["y"] + line["h"] / 2
        if rows:
            anchor = rows[-1][0]
            anchor_centre = anchor["y"] + anchor["h"] / 2
            same_row = abs(centre - anchor_centre) <= 0.45 * min(
                line["h"], anchor["h"])
            if same_row:
                rows[-1].append(line)
                continue
        rows.append([line])
    return "\n".join("  ".join(item["text"] for item in sorted(
        row, key=lambda item: item["x"])) for row in rows)


def image_to_text(image_path: str,
                  languages: Optional[List[str]] = None,
                  fast: bool = False) -> Dict[str, Any]:
    """OCR one image file. Returns {text, confidence, lines}."""
    if not _HAVE_VISION:
        return {"text": "", "confidence": 0.0, "lines": []}

    url = NSURL.fileURLWithPath_(image_path)
    lines = _recognize(url, languages or ["en-US"], fast)
    text = _lines_to_text(lines)
    confidence = (sum(l["confidence"] for l in lines) / len(lines)) if lines else 0.0
    return {"text": text, "confidence": confidence, "lines": lines}


def ndarray_to_text(image, languages: Optional[List[str]] = None,
                    fast: bool = False) -> Dict[str, Any]:
    """OCR an OpenCV/numpy BGR image by round-tripping through a temp PNG.

    Vision wants a CGImage; writing a PNG is the least fragile way to get one
    and costs a few milliseconds on a page-sized image.
    """
    if not _HAVE_VISION:
        return {"text": "", "confidence": 0.0, "lines": []}
    try:
        import cv2
    except ImportError:
        return {"text": "", "confidence": 0.0, "lines": []}

    handle, path = tempfile.mkstemp(suffix=".png")
    os.close(handle)
    try:
        cv2.imwrite(path, image)
        return image_to_text(path, languages=languages, fast=fast)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def pdf_to_text(pdf_path: str, zoom: float = 3.0,
                max_pages: Optional[int] = None,
                page_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """Rasterise a PDF and OCR every page.

    zoom=3 renders at ~216 dpi. Receipt reference numbers are small and dense;
    at the default 72 dpi Vision merges digits.
    """
    if not _HAVE_VISION:
        return {"text": "", "confidence": 0.0, "pages": 0}
    try:
        import fitz
    except ImportError:
        return {"text": "", "confidence": 0.0, "pages": 0}

    doc = fitz.open(pdf_path)
    chunks: List[str] = []
    page_texts: List[Dict[str, Any]] = []
    confidences: List[float] = []
    if page_indices is None:
        targets = list(range(len(doc)))
    else:
        targets = sorted(set(
            int(index) for index in page_indices
            if 0 <= int(index) < len(doc)
        ))
    if max_pages is not None:
        targets = targets[:max_pages]
    try:
        for index in targets:
            page = doc.load_page(index)
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), alpha=False)
            handle, path = tempfile.mkstemp(suffix=".png")
            os.close(handle)
            try:
                pixmap.save(path)
                result = image_to_text(path)
                # Some scanners attach a /Rotate flag that leaves the printed
                # receipt sideways. Retry that known rotation only when the
                # first read contains no references. Keep the corrected view
                # only when it actually recovers an identifier.
                if page.rotation:
                    from app.ibg.reference_id import extract_references
                    if not extract_references(result.get('text', '')):
                        rotated = page.get_pixmap(
                            matrix=fitz.Matrix(zoom, zoom).prerotate(-page.rotation),
                            alpha=False)
                        rotated.save(path)
                        alternative = image_to_text(path)
                        if extract_references(alternative.get('text', '')):
                            result = alternative
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            if result["text"].strip():
                chunks.append("=== PAGE %d ===\n%s" % (
                    index + 1, result["text"]))
                page_texts.append({"page": index + 1, "text": result["text"]})
                confidences.append(result["confidence"])
    finally:
        doc.close()

    return {
        "text": "\n".join(chunks),
        "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "pages": len(targets),
        "pages_processed": len(page_texts),
        "page_texts": page_texts,
    }
