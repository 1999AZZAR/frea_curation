#!/usr/bin/env python3
"""
Integration test runner for AI Content Curator.

Runs comprehensive integration tests including:
- End-to-end workflow tests
- External service integration tests  
- Performance benchmarks
- Concurrent user scenario tests

Usage:
    python tests/run_integration_tests.py [--performance] [--verbose]
"""

import unittest
import sys
import os
import time
import argparse
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_integration_workflows import (
    TestEndToEndWorkflows,
    TestExternalServiceIntegration, 
    TestConcurrentUserScenarios,
    TestResourceManagement
)
from tests.test_performance_benchmarks import (
    TestPerformanceBenchmarks,
    TestScalabilityBenchmarks
)


class IntegrationTestRunner:
    """Custom test runner for integration tests with reporting."""
    
    def __init__(self, include_performance=False, verbose=False):
        self.include_performance = include_performance
        self.verbose = verbose
        self.results = {}
        
    def run_test_suite(self, test_class, suite_name):
        """Run a specific test suite and collect results."""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Run tests with custom result collector
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2 if self.verbose else 1,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Collect results
        self.results[suite_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
            'duration': end_time - start_time,
            'output': stream.getvalue()
        }
        
        # Print summary
        print(f"\nSuite: {suite_name}")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success rate: {self.results[suite_name]['success_rate']:.1f}%")
        print(f"Duration: {self.results[suite_name]['duration']:.2f}s")
        
        if self.verbose or result.failures or result.errors:
            print("\nDetailed output:")
            print(stream.getvalue())
        
        return result.wasSuccessful()
    
    def run_all_tests(self):
        """Run all integration test suites."""
        print("AI Content Curator - Integration Test Suite")
        print("=" * 60)
        
        # Check dependencies
        self.check_dependencies()
        
        # Define test suites
        core_suites = [
            (TestEndToEndWorkflows, "End-to-End Workflow Tests"),
            (TestExternalServiceIntegration, "External Service Integration Tests"),
            (TestConcurrentUserScenarios, "Concurrent User Scenario Tests"),
            (TestResourceManagement, "Resource Management Tests")
        ]
        
        performance_suites = [
            (TestPerformanceBenchmarks, "Performance Benchmark Tests"),
            (TestScalabilityBenchmarks, "Scalability Benchmark Tests")
        ]
        
        # Run core integration tests
        all_successful = True
        for test_class, suite_name in core_suites:
            success = self.run_test_suite(test_class, suite_name)
            all_successful = all_successful and success
        
        # Run performance tests if requested
        if self.include_performance:
            print(f"\n{'='*60}")
            print("Running Performance Tests (this may take longer...)")
            print(f"{'='*60}")
            
            for test_class, suite_name in performance_suites:
                success = self.run_test_suite(test_class, suite_name)
                all_successful = all_successful and success
        
        # Generate final report
        self.generate_report()
        
        return all_successful
    
    def check_dependencies(self):
        """Check that required dependencies are available."""
        print("Checking dependencies...")
        
        required_packages = [
            ('responses', 'HTTP response mocking'),
            ('psutil', 'System resource monitoring'),
            ('concurrent.futures', 'Concurrent execution'),
        ]
        
        missing_packages = []
        
        for package, description in required_packages:
            try:
                if package == 'concurrent.futures':
                    import concurrent.futures
                else:
                    __import__(package)
                print(f"✓ {package} - {description}")
            except ImportError:
                print(f"✗ {package} - {description} (MISSING)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\nWarning: Missing packages: {', '.join(missing_packages)}")
            print("Some tests may be skipped. Install with:")
            for package in missing_packages:
                if package != 'concurrent.futures':  # Built-in in Python 3.2+
                    print(f"  pip install {package}")
            print()
        else:
            print("All dependencies available.\n")
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print(f"\n{'='*60}")
        print("INTEGRATION TEST REPORT")
        print(f"{'='*60}")
        
        total_tests = sum(r['tests_run'] for r in self.results.values())
        total_failures = sum(r['failures'] for r in self.results.values())
        total_errors = sum(r['errors'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())
        total_duration = sum(r['duration'] for r in self.results.values())
        
        overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Overall Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_tests - total_failures - total_errors}")
        print(f"  Failed: {total_failures}")
        print(f"  Errors: {total_errors}")
        print(f"  Skipped: {total_skipped}")
        print(f"  Success Rate: {overall_success_rate:.1f}%")
        print(f"  Total Duration: {total_duration:.2f}s")
        
        print(f"\nSuite Breakdown:")
        print(f"{'Suite':<40} {'Tests':<8} {'Pass%':<8} {'Time':<8}")
        print("-" * 64)
        
        for suite_name, results in self.results.items():
            print(f"{suite_name:<40} {results['tests_run']:<8} {results['success_rate']:<7.1f}% {results['duration']:<7.2f}s")
        
        # Performance summary if available
        performance_suites = [name for name in self.results.keys() if 'Performance' in name or 'Scalability' in name]
        if performance_suites:
            print(f"\nPerformance Test Summary:")
            for suite_name in performance_suites:
                results = self.results[suite_name]
                if results['success_rate'] >= 90:
                    status = "✓ PASS"
                elif results['success_rate'] >= 70:
                    status = "⚠ WARN"
                else:
                    status = "✗ FAIL"
                print(f"  {suite_name}: {status} ({results['success_rate']:.1f}%)")
        
        # Recommendations
        print(f"\nRecommendations:")
        if overall_success_rate >= 95:
            print("  ✓ Excellent test coverage and performance")
        elif overall_success_rate >= 85:
            print("  ⚠ Good test coverage, review any failures")
        else:
            print("  ✗ Test failures need attention before deployment")
        
        if total_failures > 0 or total_errors > 0:
            print("  • Review failed tests and fix underlying issues")
        
        if not self.include_performance:
            print("  • Run with --performance flag for complete performance validation")
        
        print(f"\n{'='*60}")


def main():
    """Main entry point for integration test runner."""
    parser = argparse.ArgumentParser(description='Run AI Content Curator integration tests')
    parser.add_argument('--performance', action='store_true', 
                       help='Include performance benchmark tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed test results')
    parser.add_argument('--suite', choices=['workflows', 'external', 'concurrent', 'resource', 'performance', 'scalability'],
                       help='Run specific test suite only')
    
    args = parser.parse_args()
    
    # Create and run test runner
    runner = IntegrationTestRunner(
        include_performance=args.performance,
        verbose=args.verbose
    )
    
    if args.suite:
        # Run specific suite only
        suite_map = {
            'workflows': (TestEndToEndWorkflows, "End-to-End Workflow Tests"),
            'external': (TestExternalServiceIntegration, "External Service Integration Tests"),
            'concurrent': (TestConcurrentUserScenarios, "Concurrent User Scenario Tests"),
            'resource': (TestResourceManagement, "Resource Management Tests"),
            'performance': (TestPerformanceBenchmarks, "Performance Benchmark Tests"),
            'scalability': (TestScalabilityBenchmarks, "Scalability Benchmark Tests")
        }
        
        if args.suite in suite_map:
            test_class, suite_name = suite_map[args.suite]
            runner.check_dependencies()
            success = runner.run_test_suite(test_class, suite_name)
            runner.generate_report()
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown suite: {args.suite}")
            sys.exit(1)
    else:
        # Run all tests
        success = runner.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()