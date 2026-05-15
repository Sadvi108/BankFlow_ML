import json
import re
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class UltimatePatternMatcher:
    """Ultimate pattern matcher with comprehensive bank-specific patterns."""
    
    def __init__(self):
        # Comprehensive patterns for all Malaysian banks
        self.bank_patterns = {
            'Maybank': {
                'patterns': [
                    r'\b(?:Maybank|M2U|Maybank2u|MYB)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:Maybank|M2U|Maybank2u|MYB)',
                    r'\bMYCN[A-Z0-9]{6,15}\b',  # Maybank specific
                    r'\bMB[A-Z0-9]{6,12}\b',    # Maybank short form
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and (x.isalnum() or '-' in x)
            },
            'CIMB': {
                'patterns': [
                    r'\b(?:CIMB|CIMBClicks|CIMB\s*Clicks)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9-]{8,20})',
                    r'\b([A-Z0-9-]{8,20})\s*(?:CIMB|CIMBClicks|CIMB\s*Clicks)',
                    r'\bB10-\d{4}-[A-Z0-9]{6,10}\b',  # CIMB specific format
                    r'\bCIMB[A-Z0-9]{6,12}\b',         # CIMB prefix
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9-]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9-]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and (x.isalnum() or '-' in x)
            },
            'Public Bank': {
                'patterns': [
                    r'\b(?:Public\s*Bank|PBe?Bank|PB|PBB)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:Public\s*Bank|PBe?Bank|PB|PBB)',
                    r'\bPBB[A-Z0-9]{6,12}\b',           # Public Bank specific
                    r'\bPB[A-Z0-9]{6,12}\b',            # Public Bank short form
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and x.isalnum()
            },
            'RHB': {
                'patterns': [
                    r'\b(?:RHB|RHB\s*Now|RHBNow)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:RHB|RHB\s*Now|RHBNow)',
                    r'\bRHB[A-Z0-9]{6,12}\b',            # RHB specific
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and x.isalnum()
            },
            'HSBC': {
                'patterns': [
                    r'\b(?:HSBC)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:HSBC)',
                    r'\bHSBC[A-Z0-9]{6,12}\b',          # HSBC specific
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and x.isalnum()
            },
            'UOB': {
                'patterns': [
                    r'\b(?:UOB)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:UOB)',
                    r'\bUOB[A-Z0-9]{6,12}\b',           # UOB specific
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and x.isalnum()
            },
            'Standard Chartered': {
                'patterns': [
                    r'\b(?:Standard\s*Chartered|SCB|StandardChartered)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:Standard\s*Chartered|SCB|StandardChartered)',
                    r'\bSCB[A-Z0-9]{6,12}\b',            # SCB specific
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and x.isalnum()
            },
            'DuitNow': {
                'patterns': [
                    r'\b(?:DuitNow|DN)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:DuitNow|DN)',
                    r'\bDN[A-Z0-9]{6,12}\b',            # DuitNow specific
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and x.isalnum()
            },
            'Ambank': {
                'patterns': [
                    r'\b(?:Ambank|AMBank|AMB)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:Ambank|AMBank|AMB)',
                    r'\bAMB[A-Z0-9]{6,12}\b',           # Ambank specific
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and x.isalnum()
            },
            'Hong Leong Bank': {
                'patterns': [
                    r'\b(?:Hong\s*Leong|HLB|HongLeong)\s*(?:Ref|Reference|ID|No|Number)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b([A-Z0-9]{8,20})\s*(?:Hong\s*Leong|HLB|HongLeong)',
                    r'\bHLB[A-Z0-9]{6,12}\b',          # Hong Leong specific
                    r'\b(?:Transaction|Txn|Transfer)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                    r'\b(?:Payment|Pymt)\s*(?:Ref|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
                ],
                'validation': lambda x: len(x) >= 8 and x.isalnum()
            }
        }
        
        # Generic patterns for any bank
        self.generic_patterns = [
            r'\b(?:Ref|Reference|ID|No|Number)\s*[:#-]?\s*([A-Z0-9]{8,20})',
            r'\b([A-Z0-9]{8,20})\s*(?:Ref|Reference|ID|No|Number)',
            r'\b(?:Transaction|Txn|Trans)\s*(?:ID|No|Number)?\s*[:#-]?\s*([A-Z0-9]{8,20})',
            r'\b(?:Transfer|XFER|TRF)\s*(?:Ref|ID|No)?\s*[:#-]?\s*([A-Z0-9]{8,20})',
            r'\b(?:Payment|Pymt|Pay)\s*(?:Ref|ID|No)?\s*[:#-]?\s*([A-Z0-9]{8,20})',
            r'\b(?:Invoice|Inv|Bill)\s*(?:Ref|ID|No)?\s*[:#-]?\s*([A-Z0-9]{8,20})',
            r'\b(?:Receipt|Rcpt|RCP)\s*(?:No|Number|ID)?\s*[:#-]?\s*([A-Z0-9]{8,20})',
            r'\b(?:Customer|Cust|Client)\s*(?:Ref|ID|No)?\s*[:#-]?\s*([A-Z0-9]{8,20})',
            r'\b(?:Bank)\s*(?:Ref|Reference|ID|No)?\s*[:#-]?\s*([A-Z0-9]{8,20})',
            r'\b[A-Z]{2,4}\d{8,20}\b',  # Generic alphanumeric pattern
            r'\b\d{8,20}\b',  # Pure numeric pattern (as fallback)
        ]
        
        # Amount patterns
        self.amount_patterns = [
            r'RM\s*([0-9,]+\.?\d{0,2})',  # RM format
            r'\b([0-9,]+\.?\d{0,2})\s*RM\b',  # Amount RM format
            r'\bRM\s*([0-9,]+\.?\d{0,2})\b',  # RM with amount
            r'\b([0-9,]+\.\d{2})\b',  # Decimal format
            r'\b([0-9,]+)\b',  # Comma separated format
            r'(?:Amount|Amt|Paid|Total|Sum)\s*[:#-]?\s*([0-9,]+\.?\d{0,2})',
        ]

    def extract_all_fields(self, text: str) -> Dict[str, Any]:
        """Extract all possible fields from receipt text."""
        text_upper = text.upper()
        
        # Determine bank first
        bank_name = self.detect_bank(text_upper)
        
        # Extract transaction IDs
        transaction_ids = self.extract_transaction_ids(text_upper, bank_name)
        
        # Extract amount
        amount = self.extract_amount(text)
        
        # Extract date
        date = self.extract_date(text)
        
        # Extract reference numbers
        reference_numbers = self.extract_reference_numbers(text_upper, bank_name)
        
        # Extract invoice numbers
        invoice_numbers = self.extract_invoice_numbers(text_upper)
        
        # Extract DuitNow references
        duitnow_refs = self.extract_duitnow_references(text_upper)
        
        return {
            'bank_name': bank_name,
            'transaction_ids': transaction_ids,
            'reference_numbers': reference_numbers,
            'invoice_numbers': invoice_numbers,
            'duitnow_references': duitnow_refs,
            'amount': amount,
            'date': date,
            'all_ids': list(set(transaction_ids + reference_numbers + invoice_numbers + duitnow_refs)),
            'confidence': self.calculate_confidence(transaction_ids, bank_name)
        }
    
    def detect_bank(self, text: str) -> str:
        """Detect bank from text."""
        bank_keywords = {
            'MAYBANK': ['MAYBANK', 'M2U', 'MAYBANK2U', 'MYB', 'MYCN'],
            'CIMB': ['CIMB', 'CIMBCLICKS', 'CIMB CLICKS', 'B10'],
            'PUBLIC BANK': ['PUBLIC BANK', 'PBE BANK', 'PBB', 'PB'],
            'RHB': ['RHB', 'RHB NOW', 'RHBNOW'],
            'HSBC': ['HSBC'],
            'UOB': ['UOB'],
            'STANDARD CHARTERED': ['STANDARD CHARTERED', 'SCB'],
            'DUITNOW': ['DUITNOW', 'DN'],
            'AMBANK': ['AMBANK', 'AM BANK', 'AMB'],
            'HONG LEONG BANK': ['HONG LEONG', 'HLB', 'HONGLEONG']
        }
        
        for bank, keywords in bank_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return bank
        
        return 'UNKNOWN'
    
    def extract_transaction_ids(self, text: str, bank_name: str = None) -> List[str]:
        """Extract transaction IDs with bank-specific patterns."""
        all_matches = []
        
        # Try bank-specific patterns first
        if bank_name and bank_name in self.bank_patterns:
            bank_data = self.bank_patterns[bank_name]
            for pattern in bank_data['patterns']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    match = match.strip()
                    if bank_data['validation'](match) and match not in all_matches:
                        all_matches.append(match)
        
        # Try all bank patterns
        for bank, bank_data in self.bank_patterns.items():
            if bank == bank_name:  # Skip already processed
                continue
            for pattern in bank_data['patterns']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    match = match.strip()
                    if bank_data['validation'](match) and match not in all_matches:
                        all_matches.append(match)
        
        # Try generic patterns as fallback
        for pattern in self.generic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.strip()
                if self.validate_generic_id(match) and match not in all_matches:
                    all_matches.append(match)
        
        return all_matches
    
    def extract_reference_numbers(self, text: str, bank_name: str = None) -> List[str]:
        """Extract reference numbers."""
        patterns = [
            r'\b(?:Reference|Ref)\s*(?:No|Number|ID)?\.?\s*:?\s*([A-Z0-9]{6,20})',
            r'\bCustomer\s*(?:Ref|Reference|ID|No)?\.?\s*:?\s*([A-Z0-9]{6,20})',
            r'\bBank\s*(?:Ref|Reference|ID|No)?\.?\s*:?\s*([A-Z0-9]{6,20})',
        ]
        
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            for match in found:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.strip()
                if self.validate_generic_id(match) and match not in matches:
                    matches.append(match)
        
        return matches
    
    def extract_invoice_numbers(self, text: str) -> List[str]:
        """Extract invoice numbers."""
        patterns = [
            r'\b(?:Invoice|Inv)\s*(?:No|Number|ID)?\.?\s*:?\s*([A-Z0-9]{6,20})',
            r'\b(?:Bill|Receipt)\s*(?:No|Number|ID)?\.?\s*:?\s*([A-Z0-9]{6,20})',
        ]
        
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            for match in found:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.strip()
                if self.validate_generic_id(match) and match not in matches:
                    matches.append(match)
        
        return matches
    
    def extract_duitnow_references(self, text: str) -> List[str]:
        """Extract DuitNow references."""
        patterns = [
            r'\b(?:DuitNow|DN)\s*(?:Ref|Reference|ID|No)?\.?\s*:?\s*([A-Z0-9]{8,20})',
            r'\b([A-Z0-9]{8,20})\s*(?:DuitNow|DN)',
            r'\bDN[A-Z0-9]{6,15}\b',
        ]
        
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            for match in found:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.strip()
                if self.validate_generic_id(match) and match not in matches:
                    matches.append(match)
        
        return matches
    
    def extract_amount(self, text: str) -> Optional[str]:
        """Extract amount from text."""
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.strip().replace(',', '')
                try:
                    amount = float(match)
                    if 0.01 <= amount <= 100000:  # Reasonable range
                        return f"RM {match}"
                except ValueError:
                    continue
        
        return None
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract date from text."""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None
    
    def validate_generic_id(self, text: str) -> bool:
        """Validate if text looks like a transaction/reference ID."""
        if len(text) < 6 or len(text) > 25:
            return False
        
        # Must contain at least some digits
        if not any(c.isdigit() for c in text):
            return False
        
        # Must be mostly alphanumeric
        alphanumeric_count = sum(c.isalnum() for c in text)
        if alphanumeric_count < len(text) * 0.8:
            return False
        
        # Avoid common false positives
        false_positives = ['DATE', 'PAGE', 'TOTAL', 'AMOUNT', 'BALANCE', 'STATUS']
        if text.upper() in false_positives:
            return False
        
        return True
    
    def calculate_confidence(self, transaction_ids: List[str], bank_name: str) -> float:
        """Calculate confidence score for extraction."""
        if not transaction_ids:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence if we detected a specific bank
        if bank_name != 'UNKNOWN':
            confidence += 0.2
        
        # Higher confidence for longer IDs
        for tid in transaction_ids:
            if len(tid) >= 10:
                confidence += 0.1
            if len(tid) >= 12:
                confidence += 0.1
        
        # Higher confidence for multiple matches
        if len(transaction_ids) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)

# Global instance
ultimate_matcher = UltimatePatternMatcher()

def extract_all_fields(text: str) -> Dict[str, Any]:
    """Main function to extract all fields from receipt text."""
    return ultimate_matcher.extract_all_fields(text)