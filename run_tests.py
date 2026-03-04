#!/usr/bin/env python3
"""
Unified Test Runner for Bank Receipt OCR
Runs the primary validation suites to ensure system integrity.
"""

import sys
import os
import subprocess
import time

def run_test(script_path, description):
    """Run a test script and report result."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"SCRIPT:  {script_path}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Use python executable to run the script
    cmd = [sys.executable, script_path]
    
    try:
        # Use subprocess.run to execute the script
        result = subprocess.run(cmd, capture_output=False) # let output flow to stdout
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ PASS ({duration:.2f}s)")
            return True
        else:
            print(f"\n❌ FAIL ({duration:.2f}s) - Return Code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Failed to execute script: {e}")
        return False

def main():
    print("\n🚀 STARTING BANK RECEIPT OCR VALIDATION SUITE\n")
    
    # Define the critical tests to run
    tests = [
        ("tests/test_100_percent_accuracy.py", "100% Accuracy Pattern Validation (Text-based)"),
        # Add other critical tests here if needed, e.g. integration tests if server is running
    ]
    
    passed = 0
    total = len(tests)
    
    for script, desc in tests:
        if run_test(script, desc):
            passed += 1
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {passed}/{total} Test Suites Passed")
    print(f"{'='*80}")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS GO! Ready for deployment.")
        sys.exit(0)
    else:
        print("\n⚠️  SYSTEM UNSTABLE. Fix failures before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()
