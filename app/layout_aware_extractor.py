import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from app.ultimate_patterns_v3 import ultimate_matcher_v3

logger = logging.getLogger(__name__)

class LayoutAwareExtractor:
    """
    Local extractor that uses OCR positional data to intelligently prioritize Reference IDs.
    Mimics Gemini's layout awareness without requiring an API key.
    """
    
    def __init__(self):
        self.sst_patterns = [
            r'^[WC]\d{2}-\d{4}-\d{8}$',
            r'^[WC]10',
            r'\bW10-\d{4}-\d{8}\b',
            r'\bSST\s*No\b'
        ]
        
    def extract(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Reference ID using both patterns and layout coordinates.
        """
        text = ocr_result.get('text', '')
        tokens = ocr_result.get('tokens', [])
        width = ocr_result.get('width', 1)
        height = ocr_result.get('height', 1)
        
        if not text:
            return {"success": False, "error": "No text extracted"}

        # 1. Generate Candidates using existing robust patterns
        all_matches = []
        for pattern_name, patterns in ultimate_matcher_v3.generic_patterns.items():
            if pattern_name == 'duitnow_patterns': continue
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    if isinstance(match.group(), str):
                        match_text = match.group(1) if match.groups() else match.group()
                        if ultimate_matcher_v3.is_valid_transaction_id(match_text):
                            all_matches.append({
                                'text': match_text.strip().upper().replace(" ", ""),
                                'start': match.start(),
                                'end': match.end(),
                                'pattern': pattern
                            })

        # Also check bank-specific patterns
        bank_name = ultimate_matcher_v3.detect_bank(text)
        if bank_name and bank_name in ultimate_matcher_v3.bank_patterns:
            for pattern in ultimate_matcher_v3.bank_patterns[bank_name]['patterns']:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    match_text = match.group(1) if match.groups() else match.group()
                    if ultimate_matcher_v3.is_valid_transaction_id(match_text):
                        all_matches.append({
                            'text': match_text.strip().upper().replace(" ", ""),
                            'start': match.start(),
                            'end': match.end(),
                            'pattern': pattern,
                            'bank_source': bank_name
                        })

        # 2. Score Candidates based on Layout
        scored_candidates = []
        seen = set()
        
        for match in all_matches:
            tid = match['text']
            if tid in seen: continue
            seen.add(tid)
            
            score = 100 # Base score
            
            # Find coordinates for this match
            match_tokens = self._find_tokens_for_match(match, tokens, text)
            if not match_tokens:
                scored_candidates.append({'id': tid, 'score': score})
                continue
                
            y_pos = match_tokens[0]['top'] / height
            x_pos = match_tokens[0]['left'] / width
            
            # RULE: Prioritize Header IDs (Top 40% of page)
            if y_pos < 0.4:
                score += 50
                logger.debug(f"Header boost for {tid}: +50")
            
            # RULE: Bonus for IDs on the right (frequent in CIMB/Maybank)
            if x_pos > 0.5:
                score += 20
                
            # RULE: Proximity to Reference labels
            if self._is_near_keyword(match_tokens, tokens, ["REFERENCE", "REF", "TRX", "TXN", "ID", "INVOICE"]):
                score += 150 # Increased boost from 80
                logger.debug(f"Keyword proximity boost for {tid}: +150")
                
            # RULE: Penalty for being near "Recipient" or "Payment Description" or "Terminal" or "Merchant"
            if self._is_near_keyword(match_tokens, tokens, ["RECIPIENT", "SAY", "DESCRIPTION", "MEMO", "REMARK", "TERMINAL", "MERCHANT", "MID", "TID", "TRACE", "BATCH", "APPR", "AUTH"]):
                score -= 100 # Increased penalty
                logger.debug(f"Non-system ref penalty for {tid}: -100")
                
            # RULE: Explicit SST exclusion
            is_sst = False
            for pat in self.sst_patterns:
                if re.search(pat, tid) or self._is_near_keyword(match_tokens, tokens, ["SST", "SERVICE TAX", "TAX ID"]):
                    is_sst = True
                    break
            
            if is_sst:
                score -= 150
                logger.debug(f"SST penalty for {tid}: -150")
                
            # RULE: Alphanumeric bias
            if not tid.isdigit():
                score += 30
                
            scored_candidates.append({'id': tid, 'score': score})

        # Sort by score
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. Finalize Output
        if not scored_candidates:
            return {"success": False, "error": "No candidates found"}
            
        best = scored_candidates[0]
        
        return {
            "success": True,
            "bank_name": bank_name or "Unknown",
            "reference_id": best['id'],
            "candidates": scored_candidates[:5],
            "is_successful": best['score'] > 50
        }

    def _find_tokens_for_match(self, match: Dict, tokens: List[Dict], full_text: str) -> List[Dict]:
        """Find the OCR tokens that correspond to the substring indices of a regex match."""
        # Simple proximity-based token finding
        # Since full_text is built from tokens, we can try to align them
        match_tokens = []
        match_str = full_text[match['start'] : match['end']]
        
        # This is a bit complex due to OCR joining/splitting, so we use a simplify approach:
        # Find tokens whose words appear in the match string and are roughly at the same index
        current_idx = 0
        for token in tokens:
            token_text = token['text']
            token_len = len(token_text)
            
            # Check if this token intersects with the match range
            if current_idx + token_len > match['start'] and current_idx < match['end']:
                match_tokens.append(token)
            
            current_idx += token_len + 1 # +1 for newline or space joining
            
        return match_tokens

    def _is_near_keyword(self, target_tokens: List[Dict], all_tokens: List[Dict], keywords: List[str]) -> bool:
        """Check if any of the target tokens are physically near (vertically or on the same line) any keyword."""
        if not target_tokens: return False
        
        target_y = sum(t['top'] for t in target_tokens) / len(target_tokens)
        target_x = sum(t['left'] for t in target_tokens) / len(target_tokens)
        
        for token in all_tokens:
            token_text = token['text'].upper()
            if any(kw in token_text for kw in keywords):
                # Check pixel distance (roughly)
                # On the same line? (diff in 'top' is small)
                if abs(token['top'] - target_y) < 50: # Same line
                    if abs(token['left'] - target_x) < 500: # Relatively close horizontally
                        return True
                # Just above?
                if 0 < (target_y - token['top']) < 150:
                    if abs(token['left'] - target_x) < 300:
                        return True
        return False

# Singleton instance
layout_extractor = LayoutAwareExtractor()
