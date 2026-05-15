#!/usr/bin/env python3
"""
Enhanced pattern matching for transaction ID extraction
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

class EnhancedPatternMatcher:
    """Advanced pattern matching for transaction ID extraction."""
    
    def __init__(self):
        # Bank-specific patterns with validation rules
        self.bank_patterns = {
            'Maybank': {
                'patterns': [
                    r'\bMaybank[\s:]*([A-Z0-9]{8,20})',
                    r'\bMaybank2u[\s:]*([A-Z0-9]{8,20})',
                    r'MYCN[0-9]{8,15}',  # Maybank specific prefix
                ],
                'validation': self.validate_maybank_id,
                'min_length': 8,
                'max_length': 20
            },
            'CIMB': {
                'patterns': [
                    r'\bCIMB[\s:]*([A-Z0-9]{6,18})',
                    r'\bCIMB Bank[\s:]*([A-Z0-9]{6,18})',
                    r'B10-[0-9]{4}-[0-9]{6,8}',  # CIMB specific format
                ],
                'validation': self.validate_cimb_id,
                'min_length': 6,
                'max_length': 18
            },
            'Public Bank': {
                'patterns': [
                    r'\bPublic Bank[\s:]*([A-Z0-9]{8,16})',
                    r'\bPBB[\s:]*([A-Z0-9]{8,16})',
                    r'[0-9]{8,12}PBB[0-9]{0,4}',  # PBB specific format
                ],
                'validation': self.validate_public_bank_id,
                'min_length': 8,
                'max_length': 16
            },
            'RHB': {
                'patterns': [
                    r'\bRHB[\s:]*([A-Z0-9]{6,15})',
                    r'\bRHB Bank[\s:]*([A-Z0-9]{6,15})',
                    r'[0-9]{6,10}RHB[0-9]{0,5}',  # RHB specific format
                ],
                'validation': self.validate_rhb_id,
                'min_length': 6,
                'max_length': 15
            },
            'HSBC': {
                'patterns': [
                    r'\bHSBC[\s:]*([A-Z0-9]{8,20})',
                    r'\bThe Hongkong and Shanghai Banking Corporation[\s:]*([A-Z0-9]{8,20})',
                    r'HSBC[0-9]{6,12}[A-Z]{0,4}',  # HSBC specific format
                ],
                'validation': self.validate_hsbc_id,
                'min_length': 8,
                'max_length': 20
            },
            'UOB': {
                'patterns': [
                    r'\bUOB[\s:]*([A-Z0-9]{6,16})',
                    r'\bUnited Overseas Bank[\s:]*([A-Z0-9]{6,16})',
                    r'UOB[0-9]{6,12}[A-Z]{0,4}',  # UOB specific format
                ],
                'validation': self.validate_uob_id,
                'min_length': 6,
                'max_length': 16
            },
            'Standard Chartered': {
                'patterns': [
                    r'\bStandard Chartered[\s:]*([A-Z0-9]{8,18})',
                    r'\bSCB[\s:]*([A-Z0-9]{8,18})',
                    r'SCB[0-9]{6,12}[A-Z]{0,6}',  # SCB specific format
                ],
                'validation': self.validate_scb_id,
                'min_length': 8,
                'max_length': 18
            },
            'Ambank': {
                'patterns': [
                    r'\bAmbank[\s:]*([A-Z0-9]{6,15})',
                    r'\bAmBank[\s:]*([A-Z0-9]{6,15})',
                    r'AM[0-9]{6,10}[A-Z]{0,5}',  # Ambank specific format
                ],
                'validation': self.validate_ambank_id,
                'min_length': 6,
                'max_length': 15
            },
            'Affin Bank': {
                'patterns': [
                    r'\bAffin Bank[\s:]*([A-Z0-9]{6,15})',
                    r'\bAffinBank[\s:]*([A-Z0-9]{6,15})',
                    r'AF[0-9]{6,10}[A-Z]{0,5}',  # Affin specific format
                ],
                'validation': self.validate_affin_id,
                'min_length': 6,
                'max_length': 15
            },
            'Hong Leong Bank': {
                'patterns': [
                    r'\bHong Leong Bank[\s:]*([A-Z0-9]{8,16})',
                    r'\bHLB[\s:]*([A-Z0-9]{8,16})',
                    r'HL[0-9]{6,12}[A-Z]{0,4}',  # HLB specific format
                ],
                'validation': self.validate_hlb_id,
                'min_length': 8,
                'max_length': 16
            },
            'Citibank': {
                'patterns': [
                    r'\bCitibank[\s:]*([A-Z0-9]{8,18})',
                    r'\bCiti[\s:]*([A-Z0-9]{8,18})',
                    r'CT[0-9]{6,12}[A-Z]{0,6}',  # Citi specific format
                ],
                'validation': self.validate_citi_id,
                'min_length': 8,
                'max_length': 18
            },
            'DuitNow': {
                'patterns': [
                    r'\bDuitNow[\s:]*([A-Z0-9]{10,25})',
                    r'\bDuitNow Reference[\s:]*([A-Z0-9]{8,25})',
                    r'DN[0-9]{8,15}[A-Z]{0,8}',  # DuitNow specific format
                ],
                'validation': self.validate_duitnow_id,
                'min_length': 10,
                'max_length': 25
            }
        }
        
        # General patterns for unknown banks
        self.general_patterns = [
            # Transaction reference patterns
            (r'\b(?:ref|reference|ref\.)(?!.*(?:number|no|id))[\s:]*([A-Z0-9]{6,20})', 'General Ref'),
            (r'\b(?:transaction|txn|trans)(?!.*(?:number|no|id))[\s:]*([A-Z0-9]{6,20})', 'General Transaction'),
            (r'\b(?:payment|transfer|pymt)(?!.*(?:ref|id))[\s:]*([A-Z0-9]{6,20})', 'General Payment'),
            
            # Invoice patterns
            (r'\b(?:invoice|inv)(?!.*(?:date|amount))[\s:]*([A-Z0-9]{8,20})', 'Invoice'),
            
            # Receipt patterns
            (r'\b(?:receipt|rcpt)(?!.*(?:date|no))[\s:]*([A-Z0-9]{6,18})', 'Receipt'),
            
            # Bank reference patterns
            (r'\b(?:bank|cust|customer)[\s:]*ref[\s:]*([A-Z0-9]{6,20})', 'Bank Reference'),
            
            # ID patterns
            (r'\b(?:id|no|number)(?!.*(?:card|account))[\s:]*([A-Z0-9]{6,20})', 'ID'),
            
            # Transfer patterns
            (r'\b(?:transfer|xfer|trf)(?!.*(?:amount|date))[\s:]*([A-Z0-9]{6,20})', 'Transfer'),
        ]
        
        # Common false positives to exclude
        self.false_positives = {
            'date', 'time', 'amount', 'total', 'balance', 'account', 'name', 'address',
            'phone', 'email', 'website', 'date:', 'time:', 'amount:', 'total:', 'balance:',
            'successful', 'completed', 'pending', 'failed', 'error', 'status',
            'reference', 'number', 'transaction', 'payment', 'transfer', 'invoice',
            'receipt', 'bank', 'customer', 'cust', 'ref', 'id', 'no'
        }
    
    def extract_transaction_ids(self, text: str, bank_name: str = None) -> List[Dict[str, any]]:
        """Extract transaction IDs using enhanced pattern matching."""
        results = []
        
        # Normalize text
        text_upper = text.upper()
        text_lower = text.lower()
        
        # Try bank-specific patterns first
        if bank_name and bank_name in self.bank_patterns:
            bank_results = self._extract_with_bank_patterns(text, bank_name)
            results.extend(bank_results)
        
        # Try all bank patterns if no specific bank or if bank-specific failed
        if not results:
            for bank, config in self.bank_patterns.items():
                if bank_name and bank != bank_name:
                    continue  # Skip if we have a specific bank name
                
                bank_results = self._extract_with_bank_patterns(text, bank)
                results.extend(bank_results)
        
        # Try general patterns
        general_results = self._extract_with_general_patterns(text)
        results.extend(general_results)
        
        # Remove duplicates and sort by confidence
        unique_results = self._remove_duplicates(results)
        unique_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_results
    
    def _extract_with_bank_patterns(self, text: str, bank_name: str) -> List[Dict[str, any]]:
        """Extract using bank-specific patterns."""
        results = []
        config = self.bank_patterns[bank_name]
        
        for pattern in config['patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                match_text = match.group(1) if match.groups() else match.group(0)
                
                # Basic validation
                if not self._basic_validation(match_text, config['min_length'], config['max_length']):
                    continue
                
                # Bank-specific validation
                if config['validation']:
                    confidence = config['validation'](match_text)
                    if confidence < 0.5:
                        continue
                else:
                    confidence = self._calculate_general_confidence(match_text)
                
                results.append({
                    'text': match_text,
                    'pattern': pattern,
                    'source': bank_name,
                    'confidence': confidence,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return results
    
    def _extract_with_general_patterns(self, text: str) -> List[Dict[str, any]]:
        """Extract using general patterns."""
        results = []
        
        for pattern, source in self.general_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                match_text = match.group(1) if match.groups() else match.group(0)
                
                # Skip false positives
                if match_text.lower() in self.false_positives:
                    continue
                
                # Basic validation
                if not self._basic_validation(match_text, 6, 25):
                    continue
                
                # General validation
                confidence = self._calculate_general_confidence(match_text)
                if confidence < 0.3:
                    continue
                
                results.append({
                    'text': match_text,
                    'pattern': pattern,
                    'source': source,
                    'confidence': confidence,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return results
    
    def _basic_validation(self, text: str, min_length: int, max_length: int) -> bool:
        """Basic validation for transaction ID candidates."""
        # Length check
        if len(text) < min_length or len(text) > max_length:
            return False
        
        # Must contain at least 2 digits
        if sum(c.isdigit() for c in text) < 2:
            return False
        
        # Must be mostly alphanumeric
        alphanumeric_count = sum(c.isalnum() for c in text)
        if alphanumeric_count < len(text) * 0.8:
            return False
        
        # Avoid common false positives
        if text.lower() in self.false_positives:
            return False
        
        # Must have reasonable character distribution
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        # Avoid strings that are mostly letters or mostly digits
        if letters > len(text) * 0.9 or digits > len(text) * 0.9:
            return False
        
        return True
    
    def _calculate_general_confidence(self, text: str) -> float:
        """Calculate confidence score for general patterns."""
        confidence = 0.5  # Base confidence
        
        # Length bonus
        if len(text) >= 10:
            confidence += 0.2
        elif len(text) >= 8:
            confidence += 0.1
        
        # Mixed case bonus
        if any(c.isupper() for c in text) and any(c.islower() for c in text):
            confidence += 0.1
        
        # Digit count bonus
        digit_count = sum(c.isdigit() for c in text)
        if digit_count >= 4:
            confidence += 0.2
        elif digit_count >= 2:
            confidence += 0.1
        
        # Special character bonus (limited)
        special_count = sum(not c.isalnum() for c in text)
        if 1 <= special_count <= 3:
            confidence += 0.1
        
        # Pattern bonus (alternating letters and digits)
        alternations = 0
        for i in range(1, len(text)):
            if (text[i-1].isalpha() and text[i].isdigit()) or (text[i-1].isdigit() and text[i].isalpha()):
                alternations += 1
        if alternations >= 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _remove_duplicates(self, results: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Remove duplicate results."""
        seen = set()
        unique_results = []
        
        for result in results:
            # Use normalized text as key
            normalized = re.sub(r'[^A-Z0-9]', '', result['text'].upper())
            
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_results.append(result)
        
        return unique_results
    
    # Bank-specific validation functions
    def validate_maybank_id(self, text: str) -> float:
        """Validate Maybank transaction ID."""
        confidence = 0.6  # Base confidence for Maybank pattern
        
        # Check for Maybank-specific prefixes
        if text.startswith('MYCN') or text.startswith('MB'):
            confidence += 0.3
        
        # Check for specific patterns
        if re.match(r'MYCN[0-9]{8,12}', text):
            confidence += 0.2
        
        # Length validation
        if 10 <= len(text) <= 16:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 2 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_cimb_id(self, text: str) -> float:
        """Validate CIMB transaction ID."""
        confidence = 0.6  # Base confidence for CIMB pattern
        
        # Check for CIMB-specific format
        if re.match(r'B10-[0-9]{4}-[0-9]{6,8}', text):
            confidence += 0.4
        
        # Check for CIMB prefix
        if text.startswith('CIMB') or text.startswith('CB'):
            confidence += 0.2
        
        # Length validation
        if 8 <= len(text) <= 14:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 2 and digits >= 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_public_bank_id(self, text: str) -> float:
        """Validate Public Bank transaction ID."""
        confidence = 0.6  # Base confidence for Public Bank pattern
        
        # Check for PBB suffix/prefix
        if 'PBB' in text:
            confidence += 0.3
        
        # Check for numeric patterns
        if re.match(r'[0-9]{8,12}', text):
            confidence += 0.2
        
        # Length validation
        if 8 <= len(text) <= 14:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 1 and digits >= 6:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_rhb_id(self, text: str) -> float:
        """Validate RHB transaction ID."""
        confidence = 0.6  # Base confidence for RHB pattern
        
        # Check for RHB in text
        if 'RHB' in text:
            confidence += 0.3
        
        # Check for numeric patterns
        if re.match(r'[0-9]{6,10}', text):
            confidence += 0.2
        
        # Length validation
        if 6 <= len(text) <= 12:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 1 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_hsbc_id(self, text: str) -> float:
        """Validate HSBC transaction ID."""
        confidence = 0.6  # Base confidence for HSBC pattern
        
        # Check for HSBC in text
        if 'HSBC' in text:
            confidence += 0.3
        
        # Check for longer alphanumeric patterns
        if len(text) >= 10 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            confidence += 0.2
        
        # Length validation
        if 8 <= len(text) <= 16:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 2 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_uob_id(self, text: str) -> float:
        """Validate UOB transaction ID."""
        confidence = 0.6  # Base confidence for UOB pattern
        
        # Check for UOB in text
        if 'UOB' in text:
            confidence += 0.3
        
        # Check for numeric patterns
        if re.match(r'[0-9]{6,10}', text):
            confidence += 0.2
        
        # Length validation
        if 6 <= len(text) <= 12:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 1 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_scb_id(self, text: str) -> float:
        """Validate Standard Chartered transaction ID."""
        confidence = 0.6  # Base confidence for SCB pattern
        
        # Check for SCB in text
        if 'SCB' in text:
            confidence += 0.3
        
        # Check for longer alphanumeric patterns
        if len(text) >= 10 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            confidence += 0.2
        
        # Length validation
        if 8 <= len(text) <= 16:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 2 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_ambank_id(self, text: str) -> float:
        """Validate Ambank transaction ID."""
        confidence = 0.6  # Base confidence for Ambank pattern
        
        # Check for AM prefix
        if text.startswith('AM'):
            confidence += 0.3
        
        # Check for numeric patterns
        if re.match(r'[0-9]{6,10}', text):
            confidence += 0.2
        
        # Length validation
        if 6 <= len(text) <= 12:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 1 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_affin_id(self, text: str) -> float:
        """Validate Affin Bank transaction ID."""
        confidence = 0.6  # Base confidence for Affin pattern
        
        # Check for AF prefix
        if text.startswith('AF'):
            confidence += 0.3
        
        # Check for numeric patterns
        if re.match(r'[0-9]{6,10}', text):
            confidence += 0.2
        
        # Length validation
        if 6 <= len(text) <= 12:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 1 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_hlb_id(self, text: str) -> float:
        """Validate Hong Leong Bank transaction ID."""
        confidence = 0.6  # Base confidence for HLB pattern
        
        # Check for HL prefix
        if text.startswith('HL'):
            confidence += 0.3
        
        # Check for longer alphanumeric patterns
        if len(text) >= 10 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            confidence += 0.2
        
        # Length validation
        if 8 <= len(text) <= 14:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 2 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_citi_id(self, text: str) -> float:
        """Validate Citibank transaction ID."""
        confidence = 0.6  # Base confidence for Citi pattern
        
        # Check for CT prefix
        if text.startswith('CT'):
            confidence += 0.3
        
        # Check for longer alphanumeric patterns
        if len(text) >= 10 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            confidence += 0.2
        
        # Length validation
        if 8 <= len(text) <= 16:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 2 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_duitnow_id(self, text: str) -> float:
        """Validate DuitNow transaction ID."""
        confidence = 0.6  # Base confidence for DuitNow pattern
        
        # Check for DN prefix
        if text.startswith('DN'):
            confidence += 0.3
        
        # Check for longer alphanumeric patterns (DuitNow IDs are typically longer)
        if len(text) >= 12 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            confidence += 0.2
        
        # Length validation (DuitNow IDs are typically longer)
        if 10 <= len(text) <= 20:
            confidence += 0.1
        
        # Mixed alphanumeric validation
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 3 and digits >= 4:
            confidence += 0.1
        
        return min(confidence, 1.0)

# Global instance for easy import
enhanced_pattern_matcher = EnhancedPatternMatcher()

def extract_with_enhanced_patterns(text: str, bank_name: str = None) -> List[Dict[str, any]]:
    """Convenience function to extract transaction IDs using enhanced patterns."""
    return enhanced_pattern_matcher.extract_transaction_ids(text, bank_name)