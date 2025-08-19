"""
Performance benchmark tests for the AI Content Curator.

Tests response times, throughput, memory usage, and scalability
to ensure the application meets performance requirements.
"""

import unittest
import time
import threading
import concurrent.futures
from unittest.mock import patch, Mock
import statistics
import psutil
import os
from datetime import datetime, timedelta

from app import create_app
try:
    from curator.core.models import Article, ScoreCard, ScoringConfig
except ImportError:
    # Fallback for existing project structure
    from models import Article, ScoreCard, ScoringConfig


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests with specific metrics."""
    
    def setUp(self):
        """Set up test client and performance monitoring."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Performance thresholds (based on design requirements)
        self.SINGLE_ANALYSIS_THRESHOLD = 3.0  # seconds
        self.HOMEPAGE_LOAD_THRESHOLD = 1.0     # seconds
        self.BATCH_ANALYSIS_THRESHOLD = 10.0   # seconds for 5 articles
        self.MEMORY_INCREASE_THRESHOLD = 100   # MB per request
        
        # Sample data for consistent testing
        self.sample_article = Article(
            url="https://example.com/benchmark-article",
            title="Performance Benchmark Article",
            author="Benchmark Author",
            publish_date=datetime.now() - timedelta(hours=1),
            content="This is benchmark content for performance testing. " * 50,  # ~500 words
            summary="Benchmark article summary for testing performance metrics.",
            entities=[]
        )
        
        self.sample_scorecard = ScoreCard(
            overall_score=82.5,
            readability_score=85.0,
            ner_density_score=78.0,
            sentiment_score=72.0,
            tfidf_relevance_score=88.0,
            recency_score=95.0,
            article=self.sample_article
        )

    def measure_response_time(self, func, *args, **kwargs):
        """Measure response time of a function call."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution."""
        process = psutil.Process(os.getpid())
        
        # Force garbage collection before measurement
        import gc
        gc.collect()
        
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        memory_after = process.memory_info().rss / 1024 / 1024   # MB
        
        return result, memory_after - memory_before

    def test_homepage_load_performance(self):
        """Test homepage load time meets performance requirements."""
        response_times = []
        
        # Measure multiple requests for statistical accuracy
        for _ in range(10):
            response, response_time = self.measure_response_time(
                self.client.get, '/'
            )
            self.assertEqual(response.status_code, 200)
            response_times.append(response_time)
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        print(f"Homepage load times - Avg: {avg_response_time:.3f}s, Max: {max_response_time:.3f}s")
        
        # Performance assertions
        self.assertLess(avg_response_time, self.HOMEPAGE_LOAD_THRESHOLD,
                       f"Average homepage load time {avg_response_time:.3f}s exceeds {self.HOMEPAGE_LOAD_THRESHOLD}s threshold")
        self.assertLess(max_response_time, self.HOMEPAGE_LOAD_THRESHOLD * 2,
                       f"Maximum homepage load time {max_response_time:.3f}s too high")

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_single_analysis_performance(self, mock_validate, mock_analyze):
        """Test single article analysis performance."""
        mock_validate.return_value = (True, None)
        mock_analyze.return_value = self.sample_scorecard
        
        response_times = []
        
        # Measure multiple analysis requests
        for i in range(5):
            response, response_time = self.measure_response_time(
                self.client.post, '/analyze',
                json={'url': f'https://example.com/article{i}'},
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 200)
            response_times.append(response_time)
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        print(f"Single analysis times - Avg: {avg_response_time:.3f}s, Max: {max_response_time:.3f}s")
        
        # Performance assertions
        self.assertLess(avg_response_time, self.SINGLE_ANALYSIS_THRESHOLD,
                       f"Average analysis time {avg_response_time:.3f}s exceeds {self.SINGLE_ANALYSIS_THRESHOLD}s threshold")
        self.assertLess(max_response_time, self.SINGLE_ANALYSIS_THRESHOLD * 1.5,
                       f"Maximum analysis time {max_response_time:.3f}s too high")

    @patch('curator.services.news_source.NewsSource')
    @patch('curator.services.analyzer.batch_analyze')
    @patch('curator.core.validation.validate_topic_keywords')
    def test_batch_curation_performance(self, mock_validate, mock_batch, mock_source):
        """Test batch curation performance with multiple articles."""
        mock_validate.return_value = (True, None)
        
        # Mock NewsSource
        mock_instance = Mock()
        mock_instance.get_article_urls.return_value = [
            f'https://example.com/article{i}' for i in range(5)
        ]
        mock_source.return_value = mock_instance
        
        # Mock batch analysis with realistic processing time
        def mock_batch_analyze_with_delay(*args, **kwargs):
            # Simulate realistic processing time
            time.sleep(0.05 * len(args[0]))  # 50ms per article
            return [
                ScoreCard(
                    overall_score=80.0 + i, readability_score=75.0, ner_density_score=70.0,
                    sentiment_score=65.0, tfidf_relevance_score=85.0, recency_score=90.0,
                    article=Article(url=f"https://example.com/article{i}", 
                                  title=f"Article {i}", content="Content")
                ) for i in range(5)
            ]
        
        mock_batch.side_effect = mock_batch_analyze_with_delay
        
        response_times = []
        
        # Measure multiple curation requests
        for i in range(3):
            response, response_time = self.measure_response_time(
                self.client.post, '/curate-topic',
                json={'topic': f'technology{i}', 'max_articles': 5},
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 200)
            response_times.append(response_time)
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        print(f"Batch curation times - Avg: {avg_response_time:.3f}s, Max: {max_response_time:.3f}s")
        
        # Performance assertions
        self.assertLess(avg_response_time, self.BATCH_ANALYSIS_THRESHOLD,
                       f"Average curation time {avg_response_time:.3f}s exceeds {self.BATCH_ANALYSIS_THRESHOLD}s threshold")

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_memory_usage_single_analysis(self, mock_validate, mock_analyze):
        """Test memory usage during single article analysis."""
        mock_validate.return_value = (True, None)
        mock_analyze.return_value = self.sample_scorecard
        
        memory_increases = []
        
        # Measure memory usage for multiple requests
        for i in range(5):
            response, memory_increase = self.measure_memory_usage(
                self.client.post, '/analyze',
                json={'url': f'https://example.com/article{i}'},
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 200)
            memory_increases.append(memory_increase)
        
        avg_memory_increase = statistics.mean(memory_increases)
        max_memory_increase = max(memory_increases)
        
        print(f"Memory usage - Avg: {avg_memory_increase:.2f}MB, Max: {max_memory_increase:.2f}MB")
        
        # Memory usage assertions
        self.assertLess(avg_memory_increase, self.MEMORY_INCREASE_THRESHOLD,
                       f"Average memory increase {avg_memory_increase:.2f}MB exceeds {self.MEMORY_INCREASE_THRESHOLD}MB threshold")

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_concurrent_request_performance(self, mock_validate, mock_analyze):
        """Test performance under concurrent load."""
        mock_validate.return_value = (True, None)
        
        # Create unique responses for each request
        def create_unique_scorecard(url, **kwargs):
            return ScoreCard(
                overall_score=75.0, readability_score=70.0, ner_density_score=65.0,
                sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
                article=Article(url=url, title=f"Article for {url}", content="Content")
            )
        
        mock_analyze.side_effect = create_unique_scorecard
        
        def make_request(url):
            """Make a single analysis request and measure time."""
            start_time = time.perf_counter()
            response = self.client.post('/analyze',
                                      json={'url': url},
                                      content_type='application/json')
            end_time = time.perf_counter()
            return response, end_time - start_time
        
        # Test concurrent requests
        urls = [f'https://example.com/concurrent{i}' for i in range(10)]
        
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, url) for url in urls]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        total_time = time.perf_counter() - start_time
        
        # Verify all requests succeeded
        response_times = []
        for response, response_time in results:
            self.assertEqual(response.status_code, 200)
            response_times.append(response_time)
        
        avg_response_time = statistics.mean(response_times)
        throughput = len(urls) / total_time  # requests per second
        
        print(f"Concurrent performance - Avg response: {avg_response_time:.3f}s, Throughput: {throughput:.2f} req/s")
        
        # Performance assertions
        self.assertLess(avg_response_time, self.SINGLE_ANALYSIS_THRESHOLD * 2,
                       f"Average concurrent response time {avg_response_time:.3f}s too high")
        self.assertGreater(throughput, 1.0,
                          f"Throughput {throughput:.2f} req/s too low")

    def test_static_asset_performance(self):
        """Test static asset loading performance."""
        # Test CSS loading
        response, response_time = self.measure_response_time(
            self.client.get, '/static/css/tailwind.css'
        )
        
        if response.status_code == 200:  # Only test if file exists
            print(f"CSS load time: {response_time:.3f}s")
            self.assertLess(response_time, 0.5,
                           f"CSS load time {response_time:.3f}s too high")

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_sustained_load_performance(self, mock_validate, mock_analyze):
        """Test performance under sustained load."""
        mock_validate.return_value = (True, None)
        mock_analyze.return_value = self.sample_scorecard
        
        # Measure performance over sustained requests
        num_requests = 20
        response_times = []
        
        start_time = time.perf_counter()
        
        for i in range(num_requests):
            response, response_time = self.measure_response_time(
                self.client.post, '/analyze',
                json={'url': f'https://example.com/sustained{i}'},
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 200)
            response_times.append(response_time)
        
        total_time = time.perf_counter() - start_time
        
        # Calculate performance metrics
        avg_response_time = statistics.mean(response_times)
        throughput = num_requests / total_time
        
        # Check for performance degradation over time
        first_half = response_times[:num_requests//2]
        second_half = response_times[num_requests//2:]
        
        first_half_avg = statistics.mean(first_half)
        second_half_avg = statistics.mean(second_half)
        
        print(f"Sustained load - Avg response: {avg_response_time:.3f}s, Throughput: {throughput:.2f} req/s")
        print(f"Performance degradation: {((second_half_avg - first_half_avg) / first_half_avg * 100):.1f}%")
        
        # Performance assertions
        self.assertLess(avg_response_time, self.SINGLE_ANALYSIS_THRESHOLD,
                       f"Average sustained response time {avg_response_time:.3f}s exceeds threshold")
        
        # Performance should not degrade significantly over time
        degradation_ratio = second_half_avg / first_half_avg
        self.assertLess(degradation_ratio, 1.5,
                       f"Performance degraded by {((degradation_ratio - 1) * 100):.1f}% under sustained load")

    def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        error_response_times = []
        
        # Measure error response times
        for i in range(10):
            response, response_time = self.measure_response_time(
                self.client.post, '/analyze',
                json={'url': ''},  # Invalid URL
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 400)
            error_response_times.append(response_time)
        
        avg_error_response_time = statistics.mean(error_response_times)
        
        print(f"Error handling response time: {avg_error_response_time:.3f}s")
        
        # Error responses should be fast
        self.assertLess(avg_error_response_time, 0.5,
                       f"Error response time {avg_error_response_time:.3f}s too high")

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_memory_leak_detection(self, mock_validate, mock_analyze):
        """Test for memory leaks over multiple requests."""
        mock_validate.return_value = (True, None)
        mock_analyze.return_value = self.sample_scorecard
        
        import gc
        process = psutil.Process(os.getpid())
        
        # Baseline memory measurement
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        for i in range(50):
            response = self.client.post('/analyze',
                                      json={'url': f'https://example.com/leak{i}'},
                                      content_type='application/json')
            self.assertEqual(response.status_code, 200)
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Final memory measurement
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        print(f"Memory leak test - Baseline: {baseline_memory:.2f}MB, Final: {final_memory:.2f}MB, Increase: {memory_increase:.2f}MB")
        
        # Memory increase should be reasonable for 50 requests
        self.assertLess(memory_increase, 500,
                       f"Memory increase {memory_increase:.2f}MB suggests potential memory leak")


class TestScalabilityBenchmarks(unittest.TestCase):
    """Test scalability characteristics and limits."""
    
    def setUp(self):
        """Set up test client."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_increasing_load_scalability(self, mock_validate, mock_analyze):
        """Test how performance scales with increasing load."""
        mock_validate.return_value = (True, None)
        
        article = Article(
            url="https://example.com/scalability-test",
            title="Scalability Test Article",
            content="Content for scalability testing",
            publish_date=datetime.now()
        )
        scorecard = ScoreCard(
            overall_score=75.0, readability_score=70.0, ner_density_score=65.0,
            sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
            article=article
        )
        mock_analyze.return_value = scorecard
        
        def test_load(num_concurrent, num_requests_per_thread):
            """Test performance with specific load parameters."""
            def make_requests():
                times = []
                for i in range(num_requests_per_thread):
                    start = time.perf_counter()
                    response = self.client.post('/analyze',
                                              json={'url': f'https://example.com/scale{i}'},
                                              content_type='application/json')
                    end = time.perf_counter()
                    if response.status_code == 200:
                        times.append(end - start)
                return times
            
            start_time = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(make_requests) for _ in range(num_concurrent)]
                all_times = []
                for future in concurrent.futures.as_completed(futures):
                    all_times.extend(future.result())
            total_time = time.perf_counter() - start_time
            
            if all_times:
                avg_response_time = statistics.mean(all_times)
                throughput = len(all_times) / total_time
                return avg_response_time, throughput
            return None, 0
        
        # Test with increasing load
        load_configs = [
            (1, 5),   # 1 thread, 5 requests each
            (2, 5),   # 2 threads, 5 requests each
            (5, 3),   # 5 threads, 3 requests each
        ]
        
        results = []
        for num_concurrent, num_requests in load_configs:
            avg_time, throughput = test_load(num_concurrent, num_requests)
            results.append((num_concurrent * num_requests, avg_time, throughput))
            print(f"Load {num_concurrent}x{num_requests}: Avg time {avg_time:.3f}s, Throughput {throughput:.2f} req/s")
        
        # Verify that throughput increases with load (up to a point)
        if len(results) >= 2:
            # Throughput should not decrease dramatically with moderate load increase
            throughput_ratio = results[1][2] / results[0][2] if results[0][2] > 0 else 0
            self.assertGreater(throughput_ratio, 0.5,
                             f"Throughput decreased too much with increased load: {throughput_ratio:.2f}")

    @patch('curator.services.news_source.NewsSource')
    @patch('curator.services.analyzer.batch_analyze')
    @patch('curator.core.validation.validate_topic_keywords')
    def test_batch_size_scalability(self, mock_validate, mock_batch, mock_source):
        """Test how performance scales with batch size."""
        mock_validate.return_value = (True, None)
        
        def create_mock_results(num_articles):
            return [
                ScoreCard(
                    overall_score=80.0, readability_score=75.0, ner_density_score=70.0,
                    sentiment_score=65.0, tfidf_relevance_score=85.0, recency_score=90.0,
                    article=Article(url=f"https://example.com/batch{i}", 
                                  title=f"Article {i}", content="Content")
                ) for i in range(num_articles)
            ]
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20]
        results = []
        
        for batch_size in batch_sizes:
            # Mock NewsSource
            mock_instance = Mock()
            mock_instance.get_article_urls.return_value = [
                f'https://example.com/batch{i}' for i in range(batch_size)
            ]
            mock_source.return_value = mock_instance
            
            # Mock batch analysis with size-dependent delay
            def mock_batch_with_delay(*args, **kwargs):
                time.sleep(0.01 * len(args[0]))  # 10ms per article
                return create_mock_results(len(args[0]))
            
            mock_batch.side_effect = mock_batch_with_delay
            
            start_time = time.perf_counter()
            response = self.client.post('/curate-topic',
                                      json={'topic': 'technology', 'max_articles': batch_size},
                                      content_type='application/json')
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                response_time = end_time - start_time
                throughput = batch_size / response_time  # articles per second
                results.append((batch_size, response_time, throughput))
                print(f"Batch size {batch_size}: Time {response_time:.3f}s, Throughput {throughput:.2f} articles/s")
        
        # Verify reasonable scaling characteristics
        if len(results) >= 2:
            # Response time should scale roughly linearly with batch size
            small_batch = results[0]  # (size, time, throughput)
            large_batch = results[-1]
            
            time_ratio = large_batch[1] / small_batch[1]
            size_ratio = large_batch[0] / small_batch[0]
            
            # Time ratio should be reasonable compared to size ratio
            self.assertLess(time_ratio, size_ratio * 2,
                           f"Response time scaling {time_ratio:.2f} too poor for size ratio {size_ratio:.2f}")


if __name__ == '__main__':
    # Check for required libraries
    try:
        import psutil
    except ImportError:
        print("Warning: 'psutil' library not found. Install with: pip install psutil")
        print("Memory usage tests will be skipped.")
    
    # Run performance tests with detailed output
    unittest.main(verbosity=2, buffer=True)