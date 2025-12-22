import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from PIL import Image
import io

logger = logging.getLogger(__name__)

class GeminiExtractor:
    """
    Multimodal extractor using Gemini 1.5 Flash to extract data directly from Malaysian bank receipt images.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found in environment. GeminiExtractor will not function.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
    def extract_from_image(self, image_data: bytes, filename: str = "receipt.png") -> Dict[str, Any]:
        """
        Extract fields from receipt image using multimodal Gemini 1.5 Flash.
        """
        if not self.api_key:
            return {"success": False, "error": "API Key missing"}

        try:
            # Prepare image part
            img = Image.open(io.BytesIO(image_data))
            
            # System prompt with instructions
            prompt = """
            You are an expert Malaysian Financial Document Parser. Your goal is to extract the unique Transaction Reference ID and other key details from the provided bank receipt image.

            RULES:
            1. IDENTIFY THE BANK: Recognize the bank (e.g., Maybank, CIMB, Public Bank, RHB, HSBC, UOB).
            2. FIND THE SYSTEM REFERENCE: Look for labels like "Reference ID", "Transaction ID", "Reference Number", or "Ref". 
               - PRIORITIZE IDs found in the header or near these labels.
               - IF uncertain, return all possible Reference ID candidates in the 'candidates' list with confidence scores.
            3. EXCLUDE TAX IDS: Do NOT extract SST/GST/Service Tax Registration Numbers. These usually start with "W" or "C" (e.g., W10-1808-32000018).
            4. EXCLUDE USER REFERENCES: Do NOT extract "Recipient Reference" or "Payment Description" unless the system-generated "Transaction Reference" is missing.
            5. LAYOUT AWARENESS: Reference IDs are typically at the top right or centered near the bank logo. Service Tax IDs are often in the footer.
            6. FORMAT: Extract the date (DD/MM/YYYY or similar) and amount (as a float).
            7. LANGUAGE: Support both English and Bahasa Melayu (e.g., "No. Rujukan" vs "Reference No.").

            Return ONLY a JSON object with the following structure:
            {
              "bank_name": "...",
              "reference_id": "...",
              "transaction_date": "...",
              "amount": 0.00,
              "candidates": [{"id": "...", "confidence": 0.9}, ...],
              "is_successful": true,
              "is_duitnow": boolean,
              "is_authentic": boolean
            }
            """

            # Call Gemini
            response = self.model.generate_content(
                [prompt, img],
                generation_config={
                    "response_mime_type": "application/json",
                }
            )
            
            # Parse response
            if response.text:
                result = json.loads(response.text)
                # Post-extraction normalization
                if result.get("reference_id"):
                    result["reference_id"] = result["reference_id"].strip().upper().replace(" ", "")
                return result
            else:
                return {"success": False, "error": "Empty response from Gemini"}

        except Exception as e:
            logger.error(f"Gemini extraction failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def is_valid_malaysian_ref(self, ref_id: str) -> bool:
        """
        Validation logic: check if the ID matches common Malaysian bank formats (10-15 alphanumeric).
        """
        if not ref_id:
            return False
        # Basic check: 8-25 chars, alphanumeric with some symbols allowed
        clean_id = ref_id.strip().upper().replace(" ", "")
        if len(clean_id) < 8 or len(clean_id) > 25:
            return False
            
        # Common SST patterns to reject
        sst_patterns = [r'^[WC]\d{2}-\d{4}-\d{8}$', r'^[WC]10']
        for pat in sst_patterns:
            if re.match(pat, clean_id):
                return False
                
        return True

# Singleton instance
gemini_extractor = GeminiExtractor()
