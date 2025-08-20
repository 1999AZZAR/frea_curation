#!/usr/bin/env python3
"""
Simple verification script for integration tests.
Checks that test files can be imported and basic structure is correct.
"""

import sys
import os
import importlib.util

def check_test_file(file_path):
    """Check if a test file can be imported successfully."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Count test classes and methods
        test_classes = []
        total_methods = 0
        
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and name.startswith('Test'):
                test_classes.append(name)
                methods = [m for m in dir(obj) if m.startswith('test_')]
                total_methods += len(methods)
                print(f"  {name}: {len(methods)} test methods")
        
        print(f"✓ {os.path.basename(file_path)}: {len(test_classes)} test classes, {total_methods} test methods")
        return True
        
    except Exception as e:
        print(f"✗ {os.path.basename(file_path)}: Import failed - {str(e)}")
        return False

def main():
    """Main verification function."""
    print("Verifying Integration Test Files")
    print("=" * 50)
    
    test_files = [
        "tests/test_integration_workflows.py",
        "tests/test_performance_benchmarks.py", 
        "tests/test_config.py",
        "tests/run_integration_tests.py"
    ]
    
    success_count = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            if check_test_file(test_file):
                success_count += 1
        else:
            print(f"✗ {test_file}: File not found")
    
    print(f"\nVerification Summary:")
    print(f"Successfully verified: {success_count}/{len(test_files)} files")
    
    if success_count == len(test_files):
        print("✓ All integration test files are properly structured")
        return True
    else:
        print("✗ Some integration test files have issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)