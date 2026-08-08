"""Gemini 1.5 Flash multimodal fallback for Malaysian bank receipts.

Used by simple_server.py when the local regex + layout pipeline cannot
fully validate all required fields. Returns a normalized dict that the
result_merger consumes.
"""
import os
import json
import re
import io
import logging
from typing import Any, Dict, Optional

import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You extract structured data from Malaysian bank transaction receipts.

Identify these fields and return STRICT JSON only (no prose, no markdown fences):

{
  "bank_name": "Maybank" | "CIMB" | "Public Bank" | "RHB" | "HSBC" | "UOB" | "AmBank" | "Hong Leong Bank" | "Citibank" | "Standard Chartered" | "AFFIN" | "DuitNow" | "Unknown",
  "transaction_id": "...",
  "date": "DD-MM-YYYY or DD/MM/YYYY or DD MMM YYYY",
  "amount": "NNN.NN (numeric only, no currency symbol)",
  "is_duitnow": true | false
}

RULES:
1. transaction_id is the SYSTEM-generated reference (labels: "Reference No", "Reference ID", "Transaction ID", "Transaction Reference", "No. Rujukan"). It is usually 10-25 alphanumeric chars, often containing the bank prefix (PBB, MBB, RHB, CIMB, etc.) and a date stamp.
2. NEVER return SST/GST/Service-Tax registration numbers (start with W or C, format like W10-1808-32000018). NEVER return account numbers (14-16 digit, often with slash).
3. NEVER return "Recipient Reference" or "Payment Description" — those are user-entered, not system-generated. Only use them if no system reference is visible.
4. If a field is not clearly readable, set it to null. Do not guess.
5. Amount: number only, two decimals if shown (e.g. "40.00", "1,250.50" -> "1250.50"). Strip "RM" / "MYR".
6. Date: keep the format printed on the receipt; do not reformat.

EXAMPLES OF VALID transaction_id: "PBB251031999999", "MBB210119805805", "RHB251031222222", "MYCN251031853500", "210119805805".
EXAMPLES TO REJECT: "W10-1808-32000018" (SST), "21000000000001" alone (account), "WHS32110059378" (recipient ref).

Return only the JSON object."""


class GeminiExtractor:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash-latest"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = None
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            logger.info("GOOGLE_API_KEY not set — Gemini fallback disabled.")

    @property
    def available(self) -> bool:
        return self.model is not None

    def extract_from_image_bytes(self, image_data: bytes) -> Dict[str, Any]:
        """Extract fields from raw image bytes (PNG/JPG)."""
        if not self.available:
            return {"success": False, "error": "GOOGLE_API_KEY missing"}
        try:
            img = Image.open(io.BytesIO(image_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            response = self.model.generate_content(
                [_SYSTEM_PROMPT, img],
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                },
            )
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"Gemini image extraction failed: {e}")
            return {"success": False, "error": str(e)}

    def extract_from_pdf_bytes(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Render first page to PNG, then run image extraction."""
        if not self.available:
            return {"success": False, "error": "GOOGLE_API_KEY missing"}
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) == 0:
                return {"success": False, "error": "Empty PDF"}
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            png_bytes = pix.tobytes("png")
            doc.close()
            return self.extract_from_image_bytes(png_bytes)
        except Exception as e:
            logger.error(f"Gemini PDF extraction failed: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _parse_response(raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {"success": False, "error": "Empty response"}
        text = raw.strip()
        # Strip accidental code fences
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Gemini returned non-JSON: {e}; raw={text[:200]}")
            return {"success": False, "error": "Invalid JSON from model"}

        tid = data.get("transaction_id")
        if isinstance(tid, str):
            data["transaction_id"] = tid.strip().upper().replace(" ", "")
        data["success"] = bool(data.get("transaction_id"))
        return data


gemini_extractor = GeminiExtractor()
