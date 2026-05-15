from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image
import os

try:
    import pytesseract
    # Try to set Windows default install path if not on PATH
    if os.name == "nt":
        default_win_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if os.path.exists(default_win_path):
            try:
                pytesseract.pytesseract.tesseract_cmd = default_win_path
            except Exception:
                pass
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False


def auto_rotate(image: Image.Image) -> Image.Image:
    """Use Tesseract OSD to auto-rotate images that are 90/180/270 off.

    If Tesseract isn't available, returns the original image.
    """
    if not TESSERACT_AVAILABLE:
        return image
    try:
        osd = pytesseract.image_to_osd(image)
        # OSD output contains a line like: "Rotate: 90"
        for line in osd.splitlines():
            line = line.strip()
            if line.startswith("Rotate:"):
                angle = int(line.split(":", 1)[1].strip())
                if angle % 360 != 0:
                    # PIL rotates counter-clockwise by default when using rotate()
                    return image.rotate(-angle, expand=True)
        return image
    except Exception:
        return image


def image_to_text_and_tokens(image: Image.Image) -> Tuple[str, List[Dict]]:
    """Run OCR on an image and return plain text and per-word bounding boxes.

    Tokens schema: [{text, left, top, width, height, conf}]
    If Tesseract is not available, returns empty tokens and raw text derived
    from a very naive fallback (currently empty string).
    """
    if TESSERACT_AVAILABLE:
        # Tesseract per-word data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        tokens: List[Dict] = []
        lines: List[str] = []
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw) if conf_raw != "-1" else -1.0
            except Exception:
                conf = -1.0
            if txt:
                tokens.append({
                    "text": txt,
                    "left": int(data.get("left", [0])[i]),
                    "top": int(data.get("top", [0])[i]),
                    "width": int(data.get("width", [0])[i]),
                    "height": int(data.get("height", [0])[i]),
                    "conf": conf,
                })
                lines.append(txt)
        text = " ".join(lines)
        return text, tokens
    else:
        # Fallback: no OCR configured
        return "", []


def pdf_to_text_and_tokens(pdf_path: Path, page_index: int = 0) -> Tuple[str, List[Dict]]:
    """Extract text and per-word tokens directly from a PDF (no OCR).

    Uses PyMuPDF to get textual content with bounding boxes when the PDF is not a scanned image.
    Returns (text, tokens). If textual content is unavailable, returns ("", []).
    """
    if not PYMUPDF_AVAILABLE:
        return "", []
    try:
        doc = fitz.open(str(pdf_path))
        if page_index < 0 or page_index >= len(doc):
            return "", []
        page = doc.load_page(page_index)
        # Get words: list of tuples (x0, y0, x1, y1, word, block_no, line_no, word_no)
        words = page.get_text("words") or []
        tokens: List[Dict] = []
        text_parts: List[str] = []
        for w in words:
            x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
            if not word:
                continue
            text_parts.append(word)
            tokens.append({
                "text": word,
                "left": int(x0),
                "top": int(y0),
                "width": int(x1 - x0),
                "height": int(y1 - y0),
                "conf": None,
            })
        text = " ".join(text_parts)
        return text, tokens
    except Exception:
        return "", []