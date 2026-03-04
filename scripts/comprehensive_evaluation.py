"""
Comprehensive Evaluation Framework for Bank Receipt ML Models

This script evaluates trained models on all receipts and generates detailed reports.
"""

import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import time

# Add app to path
sys.path.insert(0, str(Path.cwd()))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation of receipt processing system."""
    
    def __init__(self, receipts_dir: Path, output_dir: Path):
        self.receipts_dir = receipts_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ocr_pipeline = EnhancedOCRPipeline()
        
        # Results storage
        self.results = []
        self.metrics = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'by_bank': defaultdict(lambda: {'total': 0, 'successful': 0}),
            'errors': defaultdict(int)
        }
    
    def evaluate_all_receipts(self) -> Dict:
        """Evaluate all receipts in the directory."""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE RECEIPT EVALUATION")
        logger.info("=" * 60)
        
        # Get all receipt files
        receipt_files = []
        for ext in ['*.pdf', '*.png', '*.jpg', '*.jpeg']:
            receipt_files.extend(self.receipts_dir.glob(ext))
        
        logger.info(f"Found {len(receipt_files)} receipts to evaluate")
        
        # Process each receipt
        start_time = time.time()
        
        for receipt_path in tqdm(receipt_files, desc="Evaluating receipts"):
            result = self.evaluate_single_receipt(receipt_path)
            self.results.append(result)
            self.update_metrics(result)
        
        elapsed_time = time.time() - start_time
        
        # Calculate final metrics
        self.metrics['success_rate'] = (
            self.metrics['successful'] / self.metrics['total'] * 100
            if self.metrics['total'] > 0 else 0
        )
        self.metrics['avg_processing_time'] = elapsed_time / len(receipt_files)
        
        # Generate reports
        self.generate_reports()
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Receipts: {self.metrics['total']}")
        logger.info(f"Successful: {self.metrics['successful']} ({self.metrics['success_rate']:.2f}%)")
        logger.info(f"Failed: {self.metrics['failed']}")
        logger.info(f"Avg Processing Time: {self.metrics['avg_processing_time']:.2f}s")
        
        return self.metrics
    
    def evaluate_single_receipt(self, receipt_path: Path) -> Dict:
        """Evaluate a single receipt."""
        result = {
            'file': receipt_path.name,
            'path': str(receipt_path),
            'success': False,
            'bank': None,
            'reference_id': None,
            'confidence': 0,
            'processing_time': 0,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Process receipt
            ocr_result = self.ocr_pipeline.process_file(str(receipt_path))
            text = ocr_result.get('text', '')
            
            if not text:
                result['error'] = 'No text extracted'
                return result
            
            # Extract fields
            extraction = extract_all_fields_v3(text)
            
            # Check if extraction was successful
            bank = extraction.get('bank_name', 'Unknown')
            ref_ids = extraction.get('all_ids', [])
            
            result['bank'] = bank
            result['reference_id'] = ref_ids[0] if ref_ids else None
            result['all_reference_ids'] = ref_ids
            result['confidence'] = extraction.get('confidence', 0)
            result['date'] = extraction.get('date')
            result['amount'] = extraction.get('amount')
            result['processing_time'] = time.time() - start_time
            
            # Determine success
            if bank != 'Unknown' and ref_ids:
                result['success'] = True
            else:
                if bank == 'Unknown':
                    result['error'] = 'Bank not identified'
                else:
                    result['error'] = 'Reference ID not found'
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing {receipt_path.name}: {e}")
        
        return result
    
    def update_metrics(self, result: Dict):
        """Update metrics based on result."""
        self.metrics['total'] += 1
        
        if result['success']:
            self.metrics['successful'] += 1
            
            # Update bank-specific metrics
            bank = result['bank']
            if bank:
                self.metrics['by_bank'][bank]['total'] += 1
                self.metrics['by_bank'][bank]['successful'] += 1
        else:
            self.metrics['failed'] += 1
            
            # Track error types
            error = result.get('error', 'Unknown error')
            self.metrics['errors'][error] += 1
            
            # Update bank-specific metrics
            bank = result.get('bank')
            if bank and bank != 'Unknown':
                self.metrics['by_bank'][bank]['total'] += 1
    
    def generate_reports(self):
        """Generate evaluation reports."""
        # Save detailed results
        results_path = self.output_dir / 'detailed_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detailed results to {results_path}")
        
        # Save metrics summary
        metrics_path = self.output_dir / 'metrics.json'
        
        # Convert defaultdict to regular dict for JSON serialization
        metrics_json = {
            'total': self.metrics['total'],
            'successful': self.metrics['successful'],
            'failed': self.metrics['failed'],
            'success_rate': self.metrics['success_rate'],
            'avg_processing_time': self.metrics['avg_processing_time'],
            'by_bank': dict(self.metrics['by_bank']),
            'errors': dict(self.metrics['errors'])
        }
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Generate per-bank report
        self.generate_per_bank_report()
        
        # Generate error analysis
        self.generate_error_analysis()
        
        # Generate HTML report
        self.generate_html_report()
    
    def generate_per_bank_report(self):
        """Generate per-bank performance report."""
        per_bank_metrics = {}
        
        for bank, stats in self.metrics['by_bank'].items():
            total = stats['total']
            successful = stats['successful']
            accuracy = (successful / total * 100) if total > 0 else 0
            
            per_bank_metrics[bank] = {
                'total_receipts': total,
                'successful_extractions': successful,
                'accuracy': accuracy
            }
        
        # Sort by total receipts
        sorted_banks = sorted(
            per_bank_metrics.items(),
            key=lambda x: x[1]['total_receipts'],
            reverse=True
        )
        
        report_path = self.output_dir / 'per_bank_metrics.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(dict(sorted_banks), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved per-bank metrics to {report_path}")
        
        # Print summary
        logger.info("\nPer-Bank Performance:")
        logger.info("-" * 60)
        for bank, metrics in sorted_banks[:10]:  # Top 10
            logger.info(f"{bank:20s}: {metrics['successful_extractions']:3d}/{metrics['total_receipts']:3d} "
                       f"({metrics['accuracy']:.1f}%)")
    
    def generate_error_analysis(self):
        """Generate error analysis report."""
        error_analysis = {
            'total_errors': self.metrics['failed'],
            'error_breakdown': dict(self.metrics['errors']),
            'failed_receipts': [
                {
                    'file': r['file'],
                    'bank': r.get('bank', 'Unknown'),
                    'error': r.get('error', 'Unknown')
                }
                for r in self.results if not r['success']
            ]
        }
        
        report_path = self.output_dir / 'error_analysis.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved error analysis to {report_path}")
        
        # Print error summary
        logger.info("\nError Breakdown:")
        logger.info("-" * 60)
        for error, count in sorted(self.metrics['errors'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{error:40s}: {count:3d}")
    
    def generate_html_report(self):
        """Generate interactive HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bank Receipt Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Bank Receipt Evaluation Report</h1>
    
    <div class="metric">
        <h2>Overall Performance</h2>
        <p><strong>Total Receipts:</strong> {self.metrics['total']}</p>
        <p><strong>Successful Extractions:</strong> <span class="success">{self.metrics['successful']}</span></p>
        <p><strong>Failed Extractions:</strong> <span class="failure">{self.metrics['failed']}</span></p>
        <p><strong>Success Rate:</strong> {self.metrics['success_rate']:.2f}%</p>
        <p><strong>Avg Processing Time:</strong> {self.metrics['avg_processing_time']:.2f}s</p>
    </div>
    
    <h2>Per-Bank Performance</h2>
    <table>
        <tr>
            <th>Bank</th>
            <th>Total</th>
            <th>Successful</th>
            <th>Accuracy</th>
        </tr>
"""
        
        # Add per-bank rows
        for bank, stats in sorted(self.metrics['by_bank'].items(), 
                                  key=lambda x: x[1]['total'], reverse=True):
            total = stats['total']
            successful = stats['successful']
            accuracy = (successful / total * 100) if total > 0 else 0
            
            html_content += f"""
        <tr>
            <td>{bank}</td>
            <td>{total}</td>
            <td>{successful}</td>
            <td>{accuracy:.1f}%</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Failed Extractions</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Bank</th>
            <th>Error</th>
        </tr>
"""
        
        # Add failed receipts
        for result in self.results:
            if not result['success']:
                html_content += f"""
        <tr>
            <td>{result['file']}</td>
            <td>{result.get('bank', 'Unknown')}</td>
            <td>{result.get('error', 'Unknown')}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        report_path = self.output_dir / 'report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML report to {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Receipt Evaluation')
    parser.add_argument('--receipts-dir', type=str, default='Receipts',
                       help='Directory containing receipts to evaluate')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(
        receipts_dir=Path(args.receipts_dir),
        output_dir=Path(args.output_dir)
    )
    
    metrics = evaluator.evaluate_all_receipts()
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Success Rate: {metrics['success_rate']:.2f}%")
    print(f"Total Receipts: {metrics['total']}")
    print(f"Successful: {metrics['successful']}")
    print(f"Failed: {metrics['failed']}")
    print(f"\nReports saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
