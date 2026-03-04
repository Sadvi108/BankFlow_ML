#!/usr/bin/env python3
"""
Comprehensive testing script for bank receipt extraction system.
Tests all Malaysian banks with various receipt formats to achieve 98%+ accuracy.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import requests
import concurrent.futures
from datetime import datetime

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ultimate_patterns_v2 import UltimatePatternMatcherV2
from app.ocr_enhanced import EnhancedOCRProcessor
from app.training_pipeline import TrainingPipeline


class ComprehensiveTester:
    """Comprehensive testing system for all Malaysian banks."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.pattern_matcher = UltimatePatternMatcherV2()
        self.ocr_processor = EnhancedOCRProcessor()
        self.training_pipeline = TrainingPipeline()
        
        # Comprehensive test cases for all banks
        self.test_cases = [
            # Maybank test cases
            {
                "name": "Maybank_1",
                "text": "Maybank2u Reference: MB20251031123456 Amount: RM150.00 Date: 31/10/2025",
                "expected": {
                    "bank": "Maybank",
                    "transaction_id": "MB20251031123456",
                    "amount": "150.00",
                    "date": "31/10/2025"
                }
            },
            {
                "name": "Maybank_2", 
                "text": "MBB Transaction Ref: 20251031MYB98765432 Trx Amount: RM250.50",
                "expected": {
                    "bank": "Maybank",
                    "transaction_id": "20251031MYB98765432",
                    "amount": "250.50"
                }
            },
            {
                "name": "Maybank_3",
                "text": "Maybank Berhad\nReference No: MYB20251101ABC123\nAmount: RM500.00",
                "expected": {
                    "bank": "Maybank",
                    "transaction_id": "MYB20251101ABC123",
                    "amount": "500.00"
                }
            },
            
            # CIMB test cases
            {
                "name": "CIMB_1",
                "text": "CIMB Bank Reference: CIMB20251031XYZ789 Amount: RM750.25",
                "expected": {
                    "bank": "CIMB",
                    "transaction_id": "CIMB20251031XYZ789",
                    "amount": "750.25"
                }
            },
            {
                "name": "CIMB_2",
                "text": "CIMB Transaction ID: 20251031CIMB12345678 Total: RM1000.00",
                "expected": {
                    "bank": "CIMB",
                    "transaction_id": "20251031CIMB12345678",
                    "amount": "1000.00"
                }
            },
            
            # Public Bank test cases
            {
                "name": "PublicBank_1",
                "text": "Public Bank Berhad\nRef: PBB20251031DEF456\nAmount: RM300.00",
                "expected": {
                    "bank": "Public Bank",
                    "transaction_id": "PBB20251031DEF456",
                    "amount": "300.00"
                }
            },
            {
                "name": "PublicBank_2",
                "text": "PBB Reference Number: 20251031PBB23456789 Transaction: RM450.75",
                "expected": {
                    "bank": "Public Bank",
                    "transaction_id": "20251031PBB23456789",
                    "amount": "450.75"
                }
            },
            
            # RHB test cases
            {
                "name": "RHB_1",
                "text": "RHB Bank Reference: RHB20251031GHI012 Amount: RM600.50",
                "expected": {
                    "bank": "RHB",
                    "transaction_id": "RHB20251031GHI012",
                    "amount": "600.50"
                }
            },
            {
                "name": "RHB_2",
                "text": "RHB Ref: 20251031RHB34567890 Trx Amount: RM850.00",
                "expected": {
                    "bank": "RHB",
                    "transaction_id": "20251031RHB34567890",
                    "amount": "850.00"
                }
            },
            
            # HSBC test cases
            {
                "name": "HSBC_1",
                "text": "HSBC Bank Malaysia\nReference: HSBC20251031JKL345\nAmount: RM1200.00",
                "expected": {
                    "bank": "HSBC",
                    "transaction_id": "HSBC20251031JKL345",
                    "amount": "1200.00"
                }
            },
            {
                "name": "HSBC_2",
                "text": "HSBC Reference: 20251031HSBC45678901 Payment: RM1500.25",
                "expected": {
                    "bank": "HSBC",
                    "transaction_id": "20251031HSBC45678901",
                    "amount": "1500.25"
                }
            },
            
            # UOB test cases
            {
                "name": "UOB_1",
                "text": "UOB Bank Reference: UOB20251031MNO678 Amount: RM950.75",
                "expected": {
                    "bank": "UOB",
                    "transaction_id": "UOB20251031MNO678",
                    "amount": "950.75"
                }
            },
            {
                "name": "UOB_2",
                "text": "United Overseas Bank\nRef: 20251031UOB56789012\nTotal: RM1100.00",
                "expected": {
                    "bank": "UOB",
                    "transaction_id": "20251031UOB56789012",
                    "amount": "1100.00"
                }
            },
            
            # Standard Chartered test cases
            {
                "name": "SC_1",
                "text": "Standard Chartered\nReference: SC20251031PQR901\nAmount: RM750.50",
                "expected": {
                    "bank": "Standard Chartered",
                    "transaction_id": "SC20251031PQR901",
                    "amount": "750.50"
                }
            },
            {
                "name": "SC_2",
                "text": "SCB Ref No: 20251031SCB67890123 Transaction: RM900.25",
                "expected": {
                    "bank": "Standard Chartered",
                    "transaction_id": "20251031SCB67890123",
                    "amount": "900.25"
                }
            },
            
            # DuitNow test cases
            {
                "name": "DuitNow_1",
                "text": "DuitNow Transfer\nReference: DN20251031STU234\nAmount: RM350.00",
                "expected": {
                    "bank": "DuitNow",
                    "transaction_id": "DN20251031STU234",
                    "amount": "350.00"
                }
            },
            {
                "name": "DuitNow_2",
                "text": "DuitNow Ref: M10169596 Amount: RM31.00",
                "expected": {
                    "bank": "DuitNow",
                    "transaction_id": "M10169596",
                    "amount": "31.00"
                }
            },
            {
                "name": "DuitNow_3",
                "text": "DuitNow Reference Number: 20251031DN34567890 Payment: RM200.50",
                "expected": {
                    "bank": "DuitNow",
                    "transaction_id": "20251031DN34567890",
                    "amount": "200.50"
                }
            },
            
            # Ambank test cases
            {
                "name": "Ambank_1",
                "text": "Ambank Berhad\nReference: AMB20251031VWX567\nAmount: RM650.25",
                "expected": {
                    "bank": "Ambank",
                    "transaction_id": "AMB20251031VWX567",
                    "amount": "650.25"
                }
            },
            {
                "name": "Ambank_2",
                "text": "AMB Ref: 20251031AMB78901234 Trx: RM800.00",
                "expected": {
                    "bank": "Ambank",
                    "transaction_id": "20251031AMB78901234",
                    "amount": "800.00"
                }
            },
            
            # Hong Leong Bank test cases
            {
                "name": "HLB_1",
                "text": "Hong Leong Bank\nReference: HLB20251031YZA890\nAmount: RM550.50",
                "expected": {
                    "bank": "Hong Leong Bank",
                    "transaction_id": "HLB20251031YZA890",
                    "amount": "550.50"
                }
            },
            {
                "name": "HLB_2",
                "text": "HLB Reference: 20251031HLB89012345 Transaction: RM700.75",
                "expected": {
                    "bank": "Hong Leong Bank",
                    "transaction_id": "20251031HLB89012345",
                    "amount": "700.75"
                }
            }
        ]
    
    def test_pattern_matcher_directly(self) -> Dict[str, Any]:
        """Test pattern matcher directly for accuracy validation."""
        print("Testing pattern matcher directly...")
        
        results = {
            "test_cases": [],
            "summary": {
                "total_tests": len(self.test_cases),
                "passed": 0,
                "failed": 0,
                "accuracy": 0.0
            }
        }
        
        for test_case in self.test_cases:
            try:
                # Extract using pattern matcher
                extraction_result = self.pattern_matcher.extract_all_fields(test_case["text"])
                
                # Validate results - check both transaction_ids and reference_numbers for transaction ID
                expected_bank = test_case["expected"]["bank"].lower()
                detected_bank = extraction_result.get("bank_name", "").lower()
                bank_detected = expected_bank in [b.lower() for b in extraction_result.get("banks", [])] or expected_bank == detected_bank
                all_transaction_ids = extraction_result.get("transaction_ids", []) + extraction_result.get("reference_numbers", [])
                transaction_found = test_case["expected"]["transaction_id"] in all_transaction_ids
                amount_found = test_case["expected"]["amount"] in extraction_result.get("amounts", []) or test_case["expected"]["amount"] in extraction_result.get("amount", "")
                
                # Check if all expected fields are found
                passed = bank_detected and transaction_found and amount_found
                
                test_result = {
                    "test_name": test_case["name"],
                    "text": test_case["text"],
                    "expected": test_case["expected"],
                    "extracted": extraction_result,
                    "validation": {
                        "bank_detected": bank_detected,
                        "transaction_found": transaction_found,
                        "amount_found": amount_found,
                        "overall_passed": passed
                    },
                    "confidence": extraction_result.get("confidence", 0)
                }
                
                results["test_cases"].append(test_result)
                
                if passed:
                    results["summary"]["passed"] += 1
                else:
                    results["summary"]["failed"] += 1
                
            except Exception as e:
                print(f"Error testing {test_case['name']}: {e}")
                results["test_cases"].append({
                    "test_name": test_case["name"],
                    "error": str(e),
                    "passed": False
                })
                results["summary"]["failed"] += 1
        
        # Calculate accuracy
        results["summary"]["accuracy"] = results["summary"]["passed"] / results["summary"]["total_tests"]
        
        return results
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test API endpoints for comprehensive validation."""
        print("Testing API endpoints...")
        
        results = {
            "endpoints_tested": [],
            "summary": {
                "total_endpoints": 0,
                "working": 0,
                "failed": 0
            }
        }
        
        # Test health endpoint
        try:
            health_response = requests.get(f"{self.base_url}/health", timeout=10)
            health_result = {
                "endpoint": "/health",
                "status_code": health_response.status_code,
                "working": health_response.status_code == 200,
                "response": health_response.json() if health_response.status_code == 200 else None
            }
            results["endpoints_tested"].append(health_result)
            if health_result["working"]:
                results["summary"]["working"] += 1
            else:
                results["summary"]["failed"] += 1
            results["summary"]["total_endpoints"] += 1
        except Exception as e:
            results["endpoints_tested"].append({
                "endpoint": "/health",
                "error": str(e),
                "working": False
            })
            results["summary"]["failed"] += 1
            results["summary"]["total_endpoints"] += 1
        
        # Test comprehensive testing endpoint
        try:
            test_response = requests.get(f"{self.base_url}/test_comprehensive", timeout=30)
            test_result = {
                "endpoint": "/test_comprehensive",
                "status_code": test_response.status_code,
                "working": test_response.status_code == 200,
                "response": test_response.json() if test_response.status_code == 200 else None
            }
            results["endpoints_tested"].append(test_result)
            if test_result["working"]:
                results["summary"]["working"] += 1
            else:
                results["summary"]["failed"] += 1
            results["summary"]["total_endpoints"] += 1
        except Exception as e:
            results["endpoints_tested"].append({
                "endpoint": "/test_comprehensive",
                "error": str(e),
                "working": False
            })
            results["summary"]["failed"] += 1
            results["summary"]["total_endpoints"] += 1
        
        # Test training endpoint
        try:
            train_response = requests.post(f"{self.base_url}/train_enhanced", timeout=60)
            train_result = {
                "endpoint": "/train_enhanced",
                "status_code": train_response.status_code,
                "working": train_response.status_code in [200, 202],
                "response": train_response.json() if train_response.status_code in [200, 202] else None
            }
            results["endpoints_tested"].append(train_result)
            if train_result["working"]:
                results["summary"]["working"] += 1
            else:
                results["summary"]["failed"] += 1
            results["summary"]["total_endpoints"] += 1
        except Exception as e:
            results["endpoints_tested"].append({
                "endpoint": "/train_enhanced",
                "error": str(e),
                "working": False
            })
            results["summary"]["failed"] += 1
            results["summary"]["total_endpoints"] += 1
        
        return results
    
    def test_ocr_quality(self, test_images_dir: str = None) -> Dict[str, Any]:
        """Test OCR quality on sample images."""
        print("Testing OCR quality...")
        
        results = {
            "ocr_tests": [],
            "summary": {
                "total_images": 0,
                "successful_extractions": 0,
                "failed_extractions": 0,
                "average_confidence": 0.0
            }
        }
        
        # If no test images directory provided, create sample test data
        if not test_images_dir:
            # Test with text-based simulation
            sample_texts = [
                "Maybank2u Ref: MB20251031123456 Amount: RM150.00",
                "CIMB Bank Reference: CIMB20251031XYZ789 Amount: RM750.25",
                "DuitNow Ref: M10169596 Amount: RM31.00",
                "RHB Bank Reference: RHB20251031GHI012 Amount: RM600.50"
            ]
            
            for i, text in enumerate(sample_texts):
                try:
                    extraction_result = self.pattern_matcher.extract_all_fields(text)
                    
                    ocr_test = {
                        "test_id": f"text_simulation_{i}",
                        "text": text,
                        "extraction": extraction_result,
                        "confidence": extraction_result.get("confidence", 0),
                        "successful": extraction_result.get("confidence", 0) > 0.8
                    }
                    
                    results["ocr_tests"].append(ocr_test)
                    results["summary"]["total_images"] += 1
                    
                    if ocr_test["successful"]:
                        results["summary"]["successful_extractions"] += 1
                    else:
                        results["summary"]["failed_extractions"] += 1
                        
                except Exception as e:
                    results["ocr_tests"].append({
                        "test_id": f"text_simulation_{i}",
                        "error": str(e),
                        "successful": False
                    })
                    results["summary"]["total_images"] += 1
                    results["summary"]["failed_extractions"] += 1
        
        # Calculate average confidence
        confidences = [test.get("confidence", 0) for test in results["ocr_tests"]]
        if confidences:
            results["summary"]["average_confidence"] = sum(confidences) / len(confidences)
        
        return results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests and generate report."""
        print("Running comprehensive tests for all Malaysian banks...")
        
        start_time = time.time()
        
        # Run all test suites
        pattern_results = self.test_pattern_matcher_directly()
        api_results = self.test_api_endpoints()
        ocr_results = self.test_ocr_quality()
        
        # Generate comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_duration": time.time() - start_time,
            "pattern_matcher_results": pattern_results,
            "api_results": api_results,
            "ocr_results": ocr_results,
            "overall_assessment": {},
            "recommendations": []
        }
        
        # Overall assessment
        pattern_accuracy = pattern_results["summary"]["accuracy"]
        api_availability = api_results["summary"]["working"] / api_results["summary"]["total_endpoints"] if api_results["summary"]["total_endpoints"] > 0 else 0
        ocr_success_rate = ocr_results["summary"]["successful_extractions"] / ocr_results["summary"]["total_images"] if ocr_results["summary"]["total_images"] > 0 else 0
        
        report["overall_assessment"] = {
            "pattern_matcher_accuracy": pattern_accuracy,
            "api_availability": api_availability,
            "ocr_success_rate": ocr_success_rate,
            "target_accuracy": 0.98,
            "target_met": pattern_accuracy >= 0.98,
            "overall_score": (pattern_accuracy + api_availability + ocr_success_rate) / 3
        }
        
        # Generate recommendations
        if pattern_accuracy < 0.98:
            report["recommendations"].append(f"Pattern matcher accuracy ({pattern_accuracy:.2%}) is below target (98%). Review and enhance patterns.")
        
        if api_availability < 1.0:
            report["recommendations"].append(f"Some API endpoints are not working properly. Check server status and configuration.")
        
        if ocr_success_rate < 0.95:
            report["recommendations"].append(f"OCR success rate ({ocr_success_rate:.2%}) could be improved. Consider enhancing image preprocessing.")
        
        if pattern_accuracy >= 0.98:
            report["recommendations"].append("Excellent! Pattern matcher has achieved target accuracy of 98%+.")
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to file."""
        if not filename:
            filename = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Test report saved to: {filepath}")
        return filepath


def main():
    """Main testing function."""
    print("=" * 80)
    print("COMPREHENSIVE BANK RECEIPT EXTRACTION TESTING")
    print("=" * 80)
    
    # Initialize tester
    tester = ComprehensiveTester()
    
    # Run comprehensive tests
    print("Starting comprehensive tests...")
    report = tester.run_comprehensive_tests()
    
    # Display results
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    assessment = report["overall_assessment"]
    print(f"Pattern Matcher Accuracy: {assessment['pattern_matcher_accuracy']:.2%}")
    print(f"API Availability: {assessment['api_availability']:.2%}")
    print(f"OCR Success Rate: {assessment['ocr_success_rate']:.2%}")
    print(f"Overall Score: {assessment['overall_score']:.2%}")
    print(f"Target Accuracy (98%): {'✓ ACHIEVED' if assessment['target_met'] else '✗ NOT ACHIEVED'}")
    
    print("\nRecommendations:")
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"{i}. {recommendation}")
    
    # Save report
    report_file = tester.save_report(report)
    
    print(f"\nDetailed test report saved to: {report_file}")
    print(f"Test duration: {report['test_duration']:.2f} seconds")
    
    # Return success status
    return 0 if assessment['target_met'] else 1


if __name__ == "__main__":
    sys.exit(main())