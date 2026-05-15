"""
Comprehensive Pattern-Based Training System

This script analyzes all receipts, learns patterns, and creates an improved
extraction system WITHOUT requiring PyTorch or heavy ML dependencies.
"""

import sys
from pathlib import Path
import json
import logging
from collections import defaultdict, Counter
import re
from tqdm import tqdm

# Add app to path
sys.path.insert(0, str(Path.cwd()))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3, UltimatePatternMatcherV3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatternLearner:
    """Learns patterns from successfully processed receipts."""
    
    def __init__(self):
        self.ocr_pipeline = EnhancedOCRPipeline()
        self.learned_patterns = defaultdict(list)
        self.bank_examples = defaultdict(list)
        self.successful_extractions = []
        self.failed_extractions = []
    
    def analyze_all_receipts(self, receipts_dir: Path):
        """Analyze all receipts and learn patterns."""
        logger.info("="*60)
        logger.info("ANALYZING ALL RECEIPTS TO LEARN PATTERNS")
        logger.info("="*60)
        
        # Get all receipt files
        receipt_files = []
        for ext in ['*.pdf', '*.png', '*.jpg', '*.jpeg']:
            receipt_files.extend(receipts_dir.glob(ext))
        
        logger.info(f"Found {len(receipt_files)} receipts to analyze")
        
        # Process each receipt
        for receipt_path in tqdm(receipt_files, desc="Analyzing receipts"):
            try:
                result = self.ocr_pipeline.process_file(str(receipt_path))
                text = result.get('text', '')
                
                if not text or len(text) < 20:
                    continue
                
                # Extract fields
                extraction = extract_all_fields_v3(text)
                
                # Store results
                analysis = {
                    'file': receipt_path.name,
                    'text': text,
                    'bank': extraction.get('bank_name'),
                    'ref_ids': extraction.get('all_ids', []),
                    'amount': extraction.get('amount'),
                    'date': extraction.get('date'),
                    'confidence': extraction.get('confidence', 0)
                }
                
                if extraction.get('bank_name') != 'Unknown' and extraction.get('all_ids'):
                    self.successful_extractions.append(analysis)
                    self.bank_examples[extraction['bank_name']].append(analysis)
                    
                    # Learn patterns from successful extractions
                    self._learn_from_success(text, extraction)
                else:
                    self.failed_extractions.append(analysis)
                    
            except Exception as e:
                logger.debug(f"Error processing {receipt_path.name}: {e}")
                continue
        
        logger.info(f"\nAnalysis complete:")
        logger.info(f"  Successful: {len(self.successful_extractions)}")
        logger.info(f"  Failed: {len(self.failed_extractions)}")
        logger.info(f"  Success rate: {len(self.successful_extractions)/(len(self.successful_extractions)+len(self.failed_extractions))*100:.1f}%")
    
    def _learn_from_success(self, text: str, extraction: Dict):
        """Learn patterns from successful extraction."""
        bank = extraction['bank_name']
        ref_ids = extraction['all_ids']
        
        for ref_id in ref_ids:
            # Find context around the reference ID
            context = self._extract_context(text, ref_id)
            if context:
                self.learned_patterns[bank].append({
                    'ref_id': ref_id,
                    'context': context,
                    'pattern': self._generalize_pattern(context, ref_id)
                })
    
    def _extract_context(self, text: str, ref_id: str, window=50):
        """Extract context around a reference ID."""
        # Find the reference ID in text
        pos = text.upper().find(ref_id.upper())
        if pos == -1:
            return None
        
        start = max(0, pos - window)
        end = min(len(text), pos + len(ref_id) + window)
        
        return text[start:end]
    
    def _generalize_pattern(self, context: str, ref_id: str):
        """Create a generalized regex pattern from context."""
        # Replace the specific ID with a pattern
        pattern = re.escape(context)
        pattern = pattern.replace(re.escape(ref_id), r'([A-Z0-9]{' + str(len(ref_id)-2) + ',' + str(len(ref_id)+2) + '})')
        return pattern
    
    def generate_improved_patterns(self):
        """Generate improved patterns based on learned examples."""
        logger.info("\n" + "="*60)
        logger.info("GENERATING IMPROVED PATTERNS")
        logger.info("="*60)
        
        improved_patterns = {}
        
        for bank, examples in self.bank_examples.items():
            logger.info(f"\n{bank}: {len(examples)} successful extractions")
            
            # Analyze reference ID formats
            ref_id_formats = Counter()
            for ex in examples:
                for ref_id in ex['ref_ids']:
                    # Classify format
                    format_type = self._classify_format(ref_id)
                    ref_id_formats[format_type] += 1
            
            logger.info(f"  Common formats: {dict(ref_id_formats.most_common(3))}")
            
            # Generate patterns for this bank
            bank_patterns = []
            for format_type, count in ref_id_formats.most_common(5):
                pattern = self._format_to_pattern(format_type)
                bank_patterns.append(pattern)
            
            improved_patterns[bank] = bank_patterns
        
        return improved_patterns
    
    def _classify_format(self, ref_id: str):
        """Classify the format of a reference ID."""
        # Remove spaces for analysis
        clean_id = ref_id.replace(' ', '')
        
        if clean_id.isdigit():
            return f"numeric_{len(clean_id)}"
        elif clean_id.isalpha():
            return f"alpha_{len(clean_id)}"
        elif clean_id.isalnum():
            # Count letters and digits
            letters = sum(c.isalpha() for c in clean_id)
            digits = sum(c.isdigit() for c in clean_id)
            return f"alnum_L{letters}_D{digits}"
        else:
            return "mixed_special"
    
    def _format_to_pattern(self, format_type: str):
        """Convert format type to regex pattern."""
        if format_type.startswith('numeric_'):
            length = int(format_type.split('_')[1])
            return rf'\\b\\d{{{length-2},{length+2}}}\\b'
        elif format_type.startswith('alpha_'):
            length = int(format_type.split('_')[1])
            return rf'\\b[A-Z]{{{length-2},{length+2}}}\\b'
        elif format_type.startswith('alnum_'):
            return r'\\b[A-Z0-9]{8,20}\\b'
        else:
            return r'\\b[A-Z0-9\\-/]{8,25}\\b'
    
    def analyze_failures(self):
        """Analyze why extractions failed."""
        logger.info("\n" + "="*60)
        logger.info("ANALYZING FAILURES")
        logger.info("="*60)
        
        if not self.failed_extractions:
            logger.info("No failures to analyze!")
            return
        
        # Categorize failures
        no_bank = sum(1 for f in self.failed_extractions if f['bank'] == 'Unknown')
        no_ref_id = sum(1 for f in self.failed_extractions if not f['ref_ids'])
        
        logger.info(f"Total failures: {len(self.failed_extractions)}")
        logger.info(f"  Bank not identified: {no_bank}")
        logger.info(f"  Reference ID not found: {no_ref_id}")
        
        # Show sample failures
        logger.info("\nSample failed extractions:")
        for i, failure in enumerate(self.failed_extractions[:5]):
            logger.info(f"\n{i+1}. {failure['file']}")
            logger.info(f"   Bank: {failure['bank']}")
            logger.info(f"   Text preview: {failure['text'][:150]}...")
            
            # Try to find potential IDs
            potential_ids = re.findall(r'\\b[A-Z0-9]{8,20}\\b', failure['text'].upper())
            if potential_ids:
                logger.info(f"   Potential IDs found: {potential_ids[:5]}")
    
    def save_training_data(self, output_dir: Path):
        """Save learned patterns and training data."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save successful extractions
        with open(output_dir / 'successful_extractions.json', 'w', encoding='utf-8') as f:
            json.dump(self.successful_extractions, f, indent=2, ensure_ascii=False)
        
        # Save failed extractions
        with open(output_dir / 'failed_extractions.json', 'w', encoding='utf-8') as f:
            json.dump(self.failed_extractions, f, indent=2, ensure_ascii=False)
        
        # Save learned patterns
        with open(output_dir / 'learned_patterns.json', 'w', encoding='utf-8') as f:
            json.dump(dict(self.learned_patterns), f, indent=2, ensure_ascii=False, default=str)
        
        # Generate improved patterns
        improved_patterns = self.generate_improved_patterns()
        with open(output_dir / 'improved_patterns.json', 'w', encoding='utf-8') as f:
            json.dump(improved_patterns, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nTraining data saved to {output_dir}")
        
        # Generate summary report
        self._generate_report(output_dir)
    
    def _generate_report(self, output_dir: Path):
        """Generate training summary report."""
        report = []
        report.append("="*60)
        report.append("PATTERN LEARNING REPORT")
        report.append("="*60)
        report.append(f"\nTotal Receipts Analyzed: {len(self.successful_extractions) + len(self.failed_extractions)}")
        report.append(f"Successful Extractions: {len(self.successful_extractions)}")
        report.append(f"Failed Extractions: {len(self.failed_extractions)}")
        report.append(f"Success Rate: {len(self.successful_extractions)/(len(self.successful_extractions)+len(self.failed_extractions))*100:.1f}%")
        
        report.append("\n\nPER-BANK PERFORMANCE:")
        report.append("-"*60)
        for bank, examples in sorted(self.bank_examples.items(), key=lambda x: len(x[1]), reverse=True):
            report.append(f"{bank:20s}: {len(examples):3d} successful extractions")
        
        report.append("\n\nLEARNED PATTERNS:")
        report.append("-"*60)
        for bank, patterns in self.learned_patterns.items():
            report.append(f"\n{bank}: {len(patterns)} patterns learned")
        
        report_text = "\n".join(report)
        
        with open(output_dir / 'training_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("COMPREHENSIVE PATTERN-BASED TRAINING")
    logger.info("="*60)
    
    # Initialize learner
    learner = PatternLearner()
    
    # Analyze all receipts
    receipts_dir = Path('Receipts')
    learner.analyze_all_receipts(receipts_dir)
    
    # Analyze failures
    learner.analyze_failures()
    
    # Save training data
    output_dir = Path('pattern_training_results')
    learner.save_training_data(output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review pattern_training_results/training_report.txt")
    logger.info("2. Check pattern_training_results/improved_patterns.json")
    logger.info("3. Update app/ultimate_patterns_v3.py with learned patterns")
    logger.info("4. Test improved system with: python comprehensive_evaluation.py")
    logger.info("="*60)


if __name__ == '__main__':
    main()
