"""Bounded, adaptive OCR for bank receipts.

The production service is deliberately conservative with CPU and memory: each
page is rendered and released separately, orientation detection runs at most
once on weak reads, and expensive preprocessing only runs when needed.
"""

import cv2
import numpy as np
import pytesseract
import logging
from typing import Dict, Any, List
import re
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
import io
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRPipeline:
    """Advanced OCR pipeline for bank receipt processing"""
    
    def __init__(self):
        # Preprocessing parameters
        self.target_dpi = 300
        self.contrast_factor = 1.5
        self.sharpness_factor = 1.2
        self.brightness_factor = 1.1
        # A 3x PDF render is roughly 216 DPI and is a good accuracy/memory
        # trade-off for receipts. The old 4x render held every page in RAM.
        self.pdf_zoom = self._env_float("OCR_PDF_ZOOM", 3.0, 1.5, 4.0)
        self.max_image_dimension = self._env_int(
            "OCR_MAX_IMAGE_DIMENSION", 2800, 1600, 5000)
        # Two passes are sufficient for clean exports and normal phone photos.
        # A heavy third pass can be enabled for a deployment with more CPU.
        self.max_passes = 3 if os.getenv("OCR_ENABLE_HEAVY_PASS", "0") == "1" else 2

    @staticmethod
    def _env_float(name: str, default: float, minimum: float,
                   maximum: float) -> float:
        try:
            return max(minimum, min(maximum, float(os.getenv(name, default))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            return max(minimum, min(maximum, int(os.getenv(name, default))))
        except (TypeError, ValueError):
            return default

    def _cap_image_size(self, image: np.ndarray) -> np.ndarray:
        """Bound giant phone photos before CPU-heavy filters and OCR."""
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.max_image_dimension:
            return image
        scale = self.max_image_dimension / float(longest)
        logger.info("Downscaling OCR input from %sx%s to %.0f%%", width, height,
                    scale * 100)
        return cv2.resize(image, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    def _upscale_to_height(self, image: np.ndarray,
                           target_height: int) -> np.ndarray:
        """Upscale small text without exceeding the global dimension cap."""
        height, width = image.shape[:2]
        requested = target_height / float(height)
        bounded = self.max_image_dimension / float(max(height, width))
        scale = min(requested, bounded)
        if scale <= 1.0:
            return image
        return cv2.resize(image, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    def _render_pdf_page(self, page):
        pix = page.get_pixmap(matrix=fitz.Matrix(self.pdf_zoom, self.pdf_zoom),
                              alpha=False)
        nparr = np.frombuffer(pix.tobytes("png"), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._cap_image_size(image) if image is not None else None

    def _iter_pdf_images(self, pdf_path: str):
        """Yield one rendered page at a time so multi-page PDFs stay bounded."""
        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                image = self._render_pdf_page(doc.load_page(page_num))
                if image is not None:
                    logger.info("Rendered PDF page %s, shape=%s", page_num + 1,
                                image.shape)
                    yield page_num, image
        finally:
            doc.close()

    def process_pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """Compatibility helper; production processing uses the iterator."""
        try:
            return [image for _page_num, image in self._iter_pdf_images(pdf_path)]
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return []

    @staticmethod
    def _receipt_signal_count(text: str) -> int:
        """Count independent signs that OCR preserved receipt semantics."""
        upper = (text or "").upper()
        checks = (
            re.search(r"\b(?:REFERENCE|REF\.?\s*(?:NO|ID)|TRANSACTION\s+ID|UTR|UETR)\b", upper),
            re.search(r"\b(?:AMOUNT|TOTAL|MYR|RM|SGD|USD)\b", upper),
            re.search(r"\b(?:DATE|TIME)\b|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", upper),
            re.search(r"\b(?:BANK|TRANSFER|PAYMENT|GIRO|IBG|DUITNOW)\b", upper),
            re.search(r"\b(?:ACCOUNT|BENEFICIARY|RECIPIENT|PAYER|PAYEE)\b", upper),
        )
        return sum(bool(match) for match in checks)

    def _is_good_read(self, result: Dict[str, Any]) -> bool:
        text = (result.get("text") or "").strip()
        confidence = float(result.get("confidence") or 0.0)
        if len(text) < 20 or result.get("word_count", 0) < 5:
            return False
        # Preserve the old high-confidence fast path, with an additional safe
        # semantic path for a clearly readable receipt just below 0.85.
        return confidence >= 0.85 or (
            confidence >= 0.75 and self._receipt_signal_count(text) >= 3)

    def _result_score(self, result: Dict[str, Any]) -> float:
        """Prefer semantic completeness when two OCR passes are close."""
        text = result.get("text") or ""
        return (
            float(result.get("confidence") or 0.0)
            + 0.035 * self._receipt_signal_count(text)
            + 0.0001 * min(len(text), 400)
        )

    def extract_text_with_confidence(self, image: np.ndarray, skip_rotation: bool = False) -> Dict[str, Any]:
        """Extract text with lazy rotation and at most two normal passes."""
        started = time.perf_counter()
        try:
            if image is None or not getattr(image, "size", 0):
                raise ValueError("Empty image")
            working = self._cap_image_size(image)

            # PASS 1: Light Preprocessing (Resize + Gray + Contrast)
            light = self._run_tesseract(
                self.preprocess_image_light(working, skip_rotation=True), psm=6)
            light["method"] = "light"
            attempts = [("light", light)]
            if self._is_good_read(light):
                light["passes_used"] = 1
                light["processing_time_ms"] = round(
                    (time.perf_counter() - started) * 1000, 1)
                logger.info("Light OCR accepted (confidence %.2f)",
                            light["confidence"])
                return light

            if not skip_rotation:
                # OSD is itself a Tesseract invocation. Skip it entirely for a
                # clean first read; otherwise run it once before all fallbacks.
                working = self._auto_rotate_text(working)

            # PASS 2: uneven-lighting / camera-photo optimization.
            photo = self._run_tesseract(
                self.preprocess_for_photo(working, skip_rotation=True), psm=6)
            photo["method"] = "photo"
            attempts.append(("photo", photo))

            # PASS 3 is intentionally deployment-controlled; it is much more
            # CPU-intensive and only useful for a small class of noisy scans.
            if self.max_passes >= 3 and not self._is_good_read(photo):
                heavy = self._run_tesseract(
                    self.preprocess_image_heavy(working, skip_rotation=True), psm=11)
                heavy["method"] = "heavy"
                attempts.append(("heavy", heavy))

            best_method, best_result = max(
                attempts, key=lambda item: self._result_score(item[1]))
            best_result["method"] = best_method
            best_result["passes_used"] = len(attempts)
            best_result["processing_time_ms"] = round(
                (time.perf_counter() - started) * 1000, 1)
            logger.info("Best OCR mode: %s (confidence %.2f, passes %s)",
                        best_method, best_result["confidence"], len(attempts))
            return best_result

        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'tokens': [],
                'lines': [],
                'word_count': 0,
                'avg_word_confidence': 0.0,
                'processed_successfully': False,
                'error': str(e),
                'passes_used': 0,
                'processing_time_ms': round(
                    (time.perf_counter() - started) * 1000, 1),
            }

    def preprocess_image_light(self, image: np.ndarray, skip_rotation: bool = False) -> np.ndarray:
        """Light preprocessing: Resize, Gray, Optional Rotation, CLAHE"""
        # 1. Resize if too small (upscale for better OCR)
        height, width = image.shape[:2]
        if height < 1500:
            image = self._upscale_to_height(image, 1500)
            
        # 2. Gray
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 3. Rotate
        if not skip_rotation:
            gray = self._auto_rotate_text(gray)
            
        # 4. CLAHE (Contrast)
        enhanced = self._enhance_contrast(gray)
        return enhanced

    def preprocess_for_photo(self, image: np.ndarray, skip_rotation: bool = False) -> np.ndarray:
        """New optimized pipeline for Camera Images / Photos"""
        # 1. Resize carefully
        height, width = image.shape[:2]
        target_height = 2000
        if height < target_height:
            image = self._upscale_to_height(image, target_height)
            
        # 2. Convert to Gray
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 3. Rotate
        if not skip_rotation:
            gray = self._auto_rotate_text(gray)

        # 4. Sharpening (New step for thermal receipts)
        # Unsharp Masking: Original + (Original - Blurred) * Amount
        gaussian = cv2.GaussianBlur(gray, (0, 0), 3.0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

        # 5. Noise Removal (Bilateral Filter preserves edges better than Gaussian)
        # d=9, sigmaColor=75, sigmaSpace=75
        denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        # 6. Adaptive Thresholding with Large Block Size
        # Key for shadows/uneven lighting. Block size 31 or 41 is much better than 11.
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 15  # Block size 31, C=15 (removes more background noise)
        )
        
        # 7. Morphological Opening (Remove small salt noise)
        kernel = np.ones((2,2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 8. Optional Dilation to thicken text (Helps with dot matrix)
        # processed = cv2.dilate(processed, kernel, iterations=1)
        
        return processed

    def preprocess_image_heavy(self, image: np.ndarray, skip_rotation: bool = False) -> np.ndarray:
        """Heavy preprocessing: Light + Denoise + Sharpen + Threshold + Morph"""
        # Start with light processed image
        base = self.preprocess_image_light(image, skip_rotation)
        
        # Denoise
        denoised = self._denoise_image(base)
        # Sharpen
        sharpened = self._sharpen_image(denoised)
        # Threshold
        thresholded = self._adaptive_threshold(sharpened)
        # Morph
        morphed = self._morphological_operations(thresholded)
        
        return morphed

    def _run_tesseract(self, image: np.ndarray, psm: int = 6) -> Dict[str, Any]:
        """Helper to run tesseract on an image and parse results"""
        config = "--oem 1 --psm %d -c preserve_interword_spaces=1" % psm
        data = pytesseract.image_to_data(
            image, lang="eng", config=config,
            output_type=pytesseract.Output.DICT)
        
        lines = []
        tokens = []
        word_confidences = []
        current_line = []
        current_line_key = None
        
        item_count = len(data['text'])
        page_numbers = data.get('page_num') or [1] * item_count
        block_numbers = data.get('block_num') or [0] * item_count
        paragraph_numbers = data.get('par_num') or [0] * item_count
        for i in range(item_count):
            if float(data['conf'][i]) > 0 and data['text'][i].strip():
                word = data['text'][i].strip()
                conf = float(data['conf'][i])
                line_num = int(data['line_num'][i])
                line_key = (
                    int(page_numbers[i]),
                    int(block_numbers[i]),
                    int(paragraph_numbers[i]),
                    line_num,
                )
                
                token = {
                    'text': word,
                    'conf': conf,
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'line_num': line_num
                }
                tokens.append(token)
                
                if current_line_key is None or line_key == current_line_key:
                    current_line.append((word, conf))
                else:
                    if current_line:
                        lines.append(' '.join([w for w, _ in current_line]))
                        word_confidences.extend([c for _, c in current_line])
                    current_line = [(word, conf)]
                current_line_key = line_key
                    
        if current_line:
            lines.append(' '.join([w for w, _ in current_line]))
            word_confidences.extend([c for _, c in current_line])
            
        overall_conf = sum(word_confidences) / len(word_confidences) if word_confidences else 0
        
        return {
            'text': '\n'.join(lines),
            'confidence': overall_conf / 100.0,
            'lines': lines,
            'tokens': tokens,
            'word_count': len(word_confidences),
            'avg_word_confidence': overall_conf,
            'processed_successfully': True,
            'width': image.shape[1],
            'height': image.shape[0]
        }
    
    def preprocess_image(self, image: np.ndarray, skip_rotation: bool = False) -> np.ndarray:
        """Legacy method for backward compatibility - defaults to heavy for safety or light?"""
        # For compatibility with any direct calls, let's use heavy as it was the previous default
        # But we should really encourage using extract_text_with_confidence
        return self.preprocess_image_heavy(image, skip_rotation)

    def _auto_rotate_text(self, image: np.ndarray) -> np.ndarray:
        """Auto-rotate text to correct orientation."""
        try:
            # Use Tesseract's OSD (Orientation and Script Detection)
            osd = pytesseract.image_to_osd(image)
            rotation = int(re.search(r'Rotate: (\d+)', osd).group(1))
            
            if rotation != 0:
                logger.info(f"Auto-rotating image by {rotation} degrees")
                if rotation == 90:
                    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    return cv2.rotate(image, cv2.ROTATE_180)
                elif rotation == 270:
                    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except:
            # If OSD fails, return original
            pass
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Denoise image using Non-local Means Denoising."""
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image using unsharp masking."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding."""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    def _morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the image."""
        kernel = np.ones((2, 2), np.uint8)
        # Remove small noise
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        # Close small holes
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closing

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a file (image or PDF) and extract text"""
        started = time.perf_counter()
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                all_text = []
                confidences = []
                all_tokens = []
                page_methods = []
                page_passes = []
                pages_processed = 0
                page_top_offset = 0
                max_page_width = 1

                # Render and release one page at a time. Apart from bounding
                # memory, this preserves every page so a later-page reference
                # is never lost to an early exit.
                for page_num, image in self._iter_pdf_images(file_path):
                    result = self.extract_text_with_confidence(image)
                    if result.get('processed_successfully'):
                        all_text.append(f"=== PAGE {page_num + 1} ===")
                        all_text.append(result['text'])
                        confidences.append(result['confidence'])
                        page_methods.append(result.get('method', 'unknown'))
                        page_passes.append(int(result.get('passes_used') or 0))
                        pages_processed += 1
                        for token in result.get('tokens', []):
                            positioned = dict(token)
                            positioned['page'] = page_num + 1
                            positioned['top'] = positioned.get('top', 0) + page_top_offset
                            all_tokens.append(positioned)
                        max_page_width = max(
                            max_page_width, int(result.get('width') or image.shape[1]))
                        page_top_offset += int(result.get('height') or image.shape[0])

                if not pages_processed:
                    return {
                        'text': '', 'confidence': 0.0,
                        'error': 'Failed to process PDF',
                        'processed_successfully': False,
                        'pages_processed': 0,
                    }
                overall_confidence = sum(confidences) / len(confidences) if confidences else 0
                return {
                    'text': '\n'.join(all_text),
                    'confidence': overall_confidence * 100,  # Convert to percentage
                    'tokens': all_tokens,
                    'width': max_page_width,
                    'height': max(page_top_offset, 1),
                    'pages_processed': pages_processed,
                    'page_methods': page_methods,
                    'page_passes': page_passes,
                    'passes_used': sum(page_passes),
                    'method': 'ocr',
                    'processed_successfully': True,
                    'processing_time_ms': round(
                        (time.perf_counter() - started) * 1000, 1),
                }
            
            else:
                # Process image
                image = cv2.imread(file_path)
                if image is None:
                    return {
                        'text': '',
                        'confidence': 0.0,
                        'error': 'Failed to load image',
                        'processed_successfully': False
                    }
                
                result = self.extract_text_with_confidence(image)
                # Convert confidence to percentage for consistency
                result['confidence'] = result.get('confidence', 0) * 100
                return result
                
        except Exception as e:
            logger.error(f"File processing error: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'error': str(e),
                'processed_successfully': False
            }

    def extract_from_bytes(self, file_bytes: bytes, file_extension: str) -> Dict[str, Any]:
        """Extract text from file bytes"""
        try:
            if file_extension.lower() == '.pdf':
                pdf_stream = io.BytesIO(file_bytes)
                doc = fitz.open(stream=pdf_stream, filetype="pdf")
                all_text = []
                confidences = []
                pages_processed = 0
                try:
                    for page_num in range(len(doc)):
                        image = self._render_pdf_page(doc.load_page(page_num))
                        if image is None:
                            continue
                        result = self.extract_text_with_confidence(image)
                        if result.get('processed_successfully'):
                            all_text.append(f"=== PAGE {page_num + 1} ===")
                            all_text.append(result['text'])
                            confidences.append(result['confidence'])
                            pages_processed += 1
                finally:
                    doc.close()
                overall_confidence = sum(confidences) / len(confidences) if confidences else 0
                return {
                    'text': '\n'.join(all_text),
                    'confidence': overall_confidence * 100,
                    'pages_processed': pages_processed,
                    'method': 'ocr',
                    'processed_successfully': bool(pages_processed),
                }
            
            else:
                # Process image from bytes
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return {
                        'text': '',
                        'confidence': 0.0,
                        'error': 'Failed to decode image',
                        'processed_successfully': False
                    }
                
                result = self.extract_text_with_confidence(image)
                result['confidence'] = result.get('confidence', 0) * 100
                return result
                
        except Exception as e:
            logger.error(f"Bytes processing error: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'error': str(e),
                'processed_successfully': False
            }

# Global instance
ocr_pipeline = OCRPipeline()
