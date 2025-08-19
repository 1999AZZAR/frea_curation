# Integration Tests for AI Content Curator

This directory contains comprehensive integration tests for the AI Content Curator application, covering end-to-end workflows, external service integration, performance benchmarks, and concurrent user scenarios.

## Test Structure

### Core Integration Test Files

1. **`test_integration_workflows.py`** - Complete end-to-end workflow tests
   - `TestEndToEndWorkflows` - Full user journey tests (homepage → analysis → results)
   - `TestExternalServiceIntegration` - NewsAPI and article parsing integration with mocked responses
   - `TestConcurrentUserScenarios` - Multi-user concurrent access tests
   - `TestResourceManagement` - Resource cleanup and system limits

2. **`test_performance_benchmarks.py`** - Performance and scalability tests
   - `TestPerformanceBenchmarks` - Response time and memory usage benchmarks
   - `TestScalabilityBenchmarks` - Load testing and throughput measurements

3. **`test_config.py`** - Test configuration and utilities
   - Common fixtures, helpers, and test data
   - Performance thresholds and configuration constants

4. **`run_integration_tests.py`** - Test runner with comprehensive reporting
   - Automated test execution with detailed reports
   - Performance metrics collection and analysis

## Requirements

### Required Dependencies

Install the following packages for full integration test functionality:

```bash
pip install responses psutil
```

- **responses** - HTTP response mocking for external service tests
- **psutil** - System resource monitoring for performance tests

### Optional Dependencies

The tests will gracefully degrade if optional dependencies are missing:

- **concurrent.futures** - Built-in Python 3.2+ (for concurrent testing)
- **statistics** - Built-in Python 3.4+ (for performance analysis)

## Running the Tests

### Quick Start

Run all core integration tests:
```bash
python tests/run_integration_tests.py
```

Run with performance benchmarks (takes longer):
```bash
python tests/run_integration_tests.py --performance
```

Run with verbose output:
```bash
python tests/run_integration_tests.py --verbose
```

### Individual Test Suites

Run specific test suites:

```bash
# End-to-end workflow tests
python tests/run_integration_tests.py --suite workflows

# External service integration tests  
python tests/run_integration_tests.py --suite external

# Concurrent user scenario tests
python tests/run_integration_tests.py --suite concurrent

# Resource management tests
python tests/run_integration_tests.py --suite resource

# Performance benchmark tests
python tests/run_integration_tests.py --suite performance

# Scalability benchmark tests
python tests/run_integration_tests.py --suite scalability
```

### Using Standard unittest

You can also run tests using Python's built-in unittest module:

```bash
# Run all integration workflow tests
python -m unittest tests.test_integration_workflows -v

# Run specific test class
python -m unittest tests.test_integration_workflows.TestEndToEndWorkflows -v

# Run specific test method
python -m unittest tests.test_integration_workflows.TestEndToEndWorkflows.test_homepage_loads_successfully -v

# Run performance benchmarks
python -m unittest tests.test_performance_benchmarks -v
```

## Test Coverage

### End-to-End Workflows

- **Manual Analysis Workflow**: URL input → validation → parsing → scoring → results display
- **Topic Curation Workflow**: Topic input → NewsAPI fetch → batch analysis → ranked results
- **Comparison Workflow**: URL/text input → similarity calculation → comparison display
- **Error Handling Workflow**: Invalid inputs → appropriate error responses

### External Service Integration

- **NewsAPI Integration**: Successful responses, rate limiting, API errors, network failures
- **Article Parsing Integration**: HTML parsing, content extraction, encoding issues
- **HTTP Response Mocking**: Realistic external service behavior simulation

### Performance Requirements

- **Response Time Benchmarks**:
  - Homepage load: < 1 second
  - Single analysis: < 3 seconds  
  - Batch curation (5 articles): < 10 seconds
- **Memory Usage**: < 100MB increase per request
- **Concurrent Load**: 5+ simultaneous users
- **Throughput**: > 1 request/second sustained

### Concurrent User Scenarios

- **Simultaneous Analysis Requests**: Multiple users analyzing different articles
- **Mixed Request Types**: Homepage, analysis, curation requests concurrently
- **Resource Isolation**: Requests don't interfere with each other
- **Error Handling Under Load**: Graceful degradation during high traffic

## Test Configuration

### Performance Thresholds

Modify thresholds in `tests/test_config.py`:

```python
class TestConfig:
    SINGLE_ANALYSIS_THRESHOLD = 3.0    # seconds
    HOMEPAGE_LOAD_THRESHOLD = 1.0      # seconds
    BATCH_ANALYSIS_THRESHOLD = 10.0    # seconds
    MEMORY_INCREASE_THRESHOLD = 100    # MB
```

### Test Data

Common test fixtures are available in `TestFixtures` class:

- `create_sample_article()` - Generate realistic article data
- `create_sample_scorecard()` - Generate scorecard with valid metrics
- `create_newsapi_response()` - Generate mock NewsAPI responses

### Environment Variables

Tests respect environment variables for configuration:

- `NEWS_API_KEY` - NewsAPI key (uses 'test-api-key' if not set)
- `FLASK_ENV` - Flask environment (set to 'testing' during tests)
- `USE_EMBEDDINGS_RELEVANCE` - Enable/disable embeddings in tests

## Interpreting Results

### Test Report Format

The test runner provides comprehensive reports:

```
INTEGRATION TEST REPORT
============================================================
Overall Summary:
  Total Tests: 45
  Passed: 43
  Failed: 1
  Errors: 1
  Skipped: 0
  Success Rate: 95.6%
  Total Duration: 12.34s

Suite Breakdown:
Suite                                    Tests    Pass%    Time
----------------------------------------------------------------
End-to-End Workflow Tests                12       100.0%   3.45s
External Service Integration Tests       8        87.5%    2.10s
Concurrent User Scenario Tests           10       100.0%   4.20s
Resource Management Tests                6        100.0%   1.15s
Performance Benchmark Tests              9        100.0%   1.44s
```

### Performance Metrics

Performance tests provide detailed metrics:

- **Response Times**: Average, maximum, and percentile measurements
- **Memory Usage**: Memory increase per request and leak detection
- **Throughput**: Requests per second under various load conditions
- **Scalability**: Performance degradation under increasing load

### Success Criteria

- **95%+ Success Rate**: Excellent test coverage and performance
- **85-94% Success Rate**: Good coverage, review any failures
- **<85% Success Rate**: Issues need attention before deployment

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
2. **Missing Dependencies**: Install `responses` and `psutil` packages
3. **Performance Test Failures**: Check system resources and adjust thresholds
4. **External Service Mocking**: Verify `responses` library is properly configured

### Debug Mode

Run tests with verbose output for debugging:

```bash
python tests/run_integration_tests.py --verbose
```

### Memory Issues

If memory tests fail:
1. Close other applications to free memory
2. Adjust `MEMORY_INCREASE_THRESHOLD` in test config
3. Run tests individually to isolate memory usage

### Performance Issues

If performance tests fail:
1. Check system load and available resources
2. Adjust performance thresholds in `TestConfig`
3. Run performance tests separately: `--suite performance`

## Contributing

When adding new integration tests:

1. Follow existing test patterns and naming conventions
2. Use `TestFixtures` for common test data
3. Add appropriate mocking for external services
4. Include performance considerations for new features
5. Update this documentation for new test suites

### Test Naming Convention

- Test classes: `TestFeatureName`
- Test methods: `test_specific_scenario_description`
- Mock objects: `mock_service_name`
- Fixtures: `sample_data_type`

### Best Practices

- Mock external services to ensure test reliability
- Use realistic test data that reflects production scenarios
- Include both positive and negative test cases
- Test error conditions and edge cases
- Measure and assert performance characteristics
- Clean up resources after tests complete