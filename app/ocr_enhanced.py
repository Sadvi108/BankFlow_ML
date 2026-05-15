from typing import List, Tuple, Dict, Optional
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import re
import os

try:
    import pytesseract
    # Try to set Windows default install path if not on PATH
    if os.name == "nt":
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
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


class EnhancedOCRProcessor:
    """Advanced OCR preprocessing and text extraction with multiple enhancement techniques."""
    
    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        self.pymupdf_available = PYMUPDF_AVAILABLE
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply comprehensive image preprocessing for optimal OCR results."""
        try:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Step 1: Auto-rotate if needed
            img_array = self._auto_rotate_cv2(img_array)
            
            # Step 2: Enhance contrast and brightness
            img_array = self._enhance_contrast_brightness(img_array)
            
            # Step 3: Denoise
            img_array = self._denoise_image(img_array)
            
            # Step 4: Sharpen
            img_array = self._sharpen_image(img_array)
            
            # Step 5: Adaptive thresholding for better text extraction
            img_array = self._adaptive_threshold(img_array)
            
            # Step 6: Morphological operations to enhance text
            img_array = self._morphological_operations(img_array)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(img_array)
            
            # Step 7: PIL-based enhancements
            processed_image = self._pil_enhancements(processed_image)
            
            return processed_image
            
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return image
    
    def _auto_rotate_cv2(self, img_array: np.ndarray) -> np.ndarray:
        """Auto-rotate image using text orientation detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            
            # Detect text orientation
            coords = np.column_stack(np.where(gray > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                # Rotate the image
                (h, w) = img_array.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_array = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return img_array
        except Exception:
            return img_array
    
    def _enhance_contrast_brightness(self, img_array: np.ndarray) -> np.ndarray:
        """Enhance contrast and brightness for better text visibility."""
        try:
            # Convert to LAB color space
            if len(img_array.shape) == 3:
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge back
                enhanced = cv2.merge([l, a, b])
                img_array = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            else:
                # For grayscale images
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_array = clahe.apply(img_array)
            
            return img_array
        except Exception:
            return img_array
    
    def _denoise_image(self, img_array: np.ndarray) -> np.ndarray:
        """Apply denoising to remove artifacts."""
        try:
            if len(img_array.shape) == 3:
                # Color image denoising
                return cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            else:
                # Grayscale denoising
                return cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
        except Exception:
            return img_array
    
    def _sharpen_image(self, img_array: np.ndarray) -> np.ndarray:
        """Sharpen the image to enhance text edges."""
        try:
            # Create sharpening kernel
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            
            if len(img_array.shape) == 3:
                # Apply to each channel
                sharpened = np.zeros_like(img_array)
                for i in range(img_array.shape[2]):
                    sharpened[:,:,i] = cv2.filter2D(img_array[:,:,i], -1, kernel)
                return sharpened
            else:
                return cv2.filter2D(img_array, -1, kernel)
        except Exception:
            return img_array
    
    def _adaptive_threshold(self, img_array: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction."""
        try:
            if len(img_array.shape) == 3:
                # Convert to grayscale first
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Convert back to RGB if original was color
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            else:
                img_array = thresh
            
            return img_array
        except Exception:
            return img_array
    
    def _morphological_operations(self, img_array: np.ndarray) -> np.ndarray:
        """Apply morphological operations to enhance text structure."""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Create kernels for different operations
            kernel_dilate = np.ones((2, 2), np.uint8)
            kernel_erode = np.ones((1, 1), np.uint8)
            
            # Apply dilation to make text thicker
            dilated = cv2.dilate(gray, kernel_dilate, iterations=1)
            
            # Apply erosion to remove small artifacts
            cleaned = cv2.erode(dilated, kernel_erode, iterations=1)
            
            # Convert back to RGB if original was color
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
            else:
                img_array = cleaned
            
            return img_array
        except Exception:
            return img_array
    
    def _pil_enhancements(self, image: Image.Image) -> Image.Image:
        """Apply PIL-based enhancements."""
        try:
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            return image
        except Exception:
            return image
    
    def extract_text_with_confidence(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        """Extract text with confidence scores and detailed token information."""
        if not self.tesseract_available:
            return "", []
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Configure Tesseract for optimal results
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}<>@#$%^&*+=_-~/\|"\' '
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            tokens: List[Dict] = []
            lines: List[str] = []
            current_line = []
            
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                conf_raw = data["conf"][i]
                
                try:
                    conf = float(conf_raw) if conf_raw != "-1" else -1.0
                except Exception:
                    conf = -1.0
                
                if txt:
                    token_info = {
                        "text": txt,
                        "left": int(data.get("left", [0])[i]),
                        "top": int(data.get("top", [0])[i]),
                        "width": int(data.get("width", [0])[i]),
                        "height": int(data.get("height", [0])[i]),
                        "conf": conf,
                        "line_num": int(data.get("line_num", [0])[i]),
                        "word_num": int(data.get("word_num", [0])[i]),
                        "block_num": int(data.get("block_num", [0])[i]),
                        "par_num": int(data.get("par_num", [0])[i])
                    }
                    tokens.append(token_info)
                    current_line.append(txt)
                
                # Check if this is the end of a line
                if i < len(data["text"]) - 1 and data["line_num"][i] != data["line_num"][i + 1]:
                    if current_line:
                        lines.append(" ".join(current_line))
                        current_line = []
            
            # Add the last line if exists
            if current_line:
                lines.append(" ".join(current_line))
            
            # Join lines with proper spacing
            text = "\n".join(lines)
            
            # Post-process text for better quality
            text = self._post_process_text(text)
            
            return text, tokens
            
        except Exception as e:
            print(f"Error in OCR extraction: {e}")
            return "", []
    
    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text for better quality."""
        try:
            # Fix common OCR errors
            corrections = {
                'O': '0',  # Letter O to zero
                'l': '1',  # Lowercase L to one
                'I': '1',  # Uppercase I to one
                'S': '5',  # Letter S to five (in numeric contexts)
                'B': '8',  # Letter B to eight (in numeric contexts)
                'G': '6',  # Letter G to six (in numeric contexts)
            }
            
            # Apply corrections only in numeric contexts
            lines = text.split('\n')
            processed_lines = []
            
            for line in lines:
                # Check if line contains numbers or transaction-related keywords
                if re.search(r'\d|Ref|Trx|Txn|Amount|RM|Date|Time', line, re.IGNORECASE):
                    for old, new in corrections.items():
                        line = re.sub(rf'(?<=[A-Z0-9]){old}(?=[A-Z0-9])', new, line)
                processed_lines.append(line)
            
            return '\n'.join(processed_lines)
            
        except Exception:
            return text
    
    def extract_from_pdf(self, pdf_path: Path, page_index: int = 0) -> Tuple[str, List[Dict]]:
        """Extract text from PDF with fallback to OCR if needed."""
        if not self.pymupdf_available:
            return "", []
        
        try:
            doc = fitz.open(str(pdf_path))
            if page_index < 0 or page_index >= len(doc):
                return "", []
            
            page = doc.load_page(page_index)
            
            # First try to extract text directly
            text = page.get_text()
            
            if text.strip():
                # Direct text extraction successful
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
                        "conf": 95.0,  # High confidence for direct PDF text
                        "line_num": 0,
                        "word_num": len(tokens),
                        "block_num": 0,
                        "par_num": 0
                    })
                
                text = " ".join(text_parts)
                text = self._post_process_text(text)
                return text, tokens
            
            else:
                # No direct text, try OCR on rendered image
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                from io import BytesIO
                img = Image.open(BytesIO(img_data))
                
                # Use enhanced OCR
                return self.extract_text_with_confidence(img)
                
        except Exception as e:
            print(f"Error in PDF extraction: {e}")
            return "", []


# Global instance for backward compatibility
def auto_rotate(image: Image.Image) -> Image.Image:
    """Legacy function for backward compatibility."""
    processor = EnhancedOCRProcessor()
    img_array = np.array(image)
    rotated_array = processor._auto_rotate_cv2(img_array)
    return Image.fromarray(rotated_array)


def image_to_text_and_tokens(image: Image.Image) -> Tuple[str, List[Dict]]:
    """Legacy function for backward compatibility."""
    processor = EnhancedOCRProcessor()
    return processor.extract_text_with_confidence(image)


def pdf_to_text_and_tokens(pdf_path: Path, page_index: int = 0) -> Tuple[str, List[Dict]]:
    """Legacy function for backward compatibility."""
    processor = EnhancedOCRProcessor()
    return processor.extract_from_pdf(pdf_path, page_index)