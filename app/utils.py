from pathlib import Path
from typing import Optional
from PIL import Image
import io
import numpy as np
import cv2

import fitz  # PyMuPDF


def load_image(path: Path) -> Image.Image:
    """Load an image file into PIL.Image (RGB)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pdf_to_image(pdf_path: Path, page_index: int = 0, zoom: float = 2.0) -> Image.Image:
    """Render first page of PDF to a PIL Image using PyMuPDF.

    Args:
        pdf_path: path to PDF
        page_index: which page to render (default first)
        zoom: scaling factor (~ 2.0 ≈ 144 DPI if base is 72 DPI)
    Returns:
        PIL.Image of the rendered page
    """
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= len(doc):
        raise IndexError("PDF page index out of range")
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes(output="png")
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_pil(mat: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_image(img: Image.Image) -> Image.Image:
    """Basic preprocessing: grayscale, CLAHE, deskew, border trim.

    Designed to improve OCR robustness for scanned/taken photos.
    """
    mat = _pil_to_cv(img)

    # Convert to grayscale
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

    # CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Binarize (Otsu)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew using minimum area rect on foreground pixels
    coords = np.column_stack(np.where(bw > 0))
    angle = 0.0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[-1]
        # minAreaRect angle is in range [-90, 0)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        # Rotate around center
        (h, w) = bw.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        mat = cv2.warpAffine(mat, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        bw = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # Optional: trim white borders
    # Find bounding box of content
    ys, xs = np.where(bw == 0)  # black pixels (assuming text is darker)
    if xs.size > 0 and ys.size > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # Add margin
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(bw.shape[1], x_max + margin)
        y_max = min(bw.shape[0], y_max + margin)
        mat = mat[y_min:y_max, x_min:x_max]

    # Slight denoise + sharpen
    mat = cv2.GaussianBlur(mat, (0, 0), 0.8)
    mat = cv2.addWeighted(mat, 1.5, cv2.GaussianBlur(mat, (0, 0), 1.2), -0.5, 0)

    return _cv_to_pil(mat)