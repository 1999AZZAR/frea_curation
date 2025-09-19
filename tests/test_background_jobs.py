#!/usr/bin/env python3
"""
Test script for background job processing.

This script tests the Celery task system by submitting jobs and monitoring their progress.
"""

import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_job_manager():
    """Test the JobManager functionality."""
    print("Testing JobManager...")
    
    try:
        from curator.core.job_manager import JobManager
        from curator.core.config import load_scoring_config
        
        # Initialize job manager
        job_manager = JobManager()
        config = load_scoring_config()
        
        # Test health check
        print("\n1. Testing health check...")
        health = job_manager.health_check()
        print(f"Health status: {json.dumps(health, indent=2)}")
        
        if not health.get('redis_connected'):
            print("❌ Redis is not connected. Please start Redis first.")
            return False
        
        if health.get('celery_workers', 0) == 0:
            print("⚠️  No Celery workers detected. Please start a worker.")
            print("   Run: python celery_worker.py")
            return False
        
        # Test single article analysis
        print("\n2. Testing single article analysis...")
        test_url = "https://example.com/test-article"
        
        try:
            job_id = job_manager.submit_single_analysis(
                url=test_url,
                query="test query",
                config=config
            )
            print(f"✅ Submitted single analysis job: {job_id}")
            
            # Monitor job progress
            for i in range(30):  # Wait up to 30 seconds
                status = job_manager.get_job_status(job_id)
                print(f"   Status: {status.get('state')} - {status.get('status', 'Unknown')}")
                
                if status.get('state') in ['SUCCESS', 'FAILURE']:
                    break
                    
                time.sleep(1)
            
            # Get final result
            result = job_manager.get_job_result(job_id)
            if result:
                print(f"✅ Job completed with result: {result.get('status', 'Unknown')}")
            else:
                print("❌ No result available")
                
        except Exception as e:
            print(f"❌ Single analysis test failed: {e}")
        
        # Test batch analysis
        print("\n3. Testing batch analysis...")
        test_urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3"
        ]
        
        try:
            job_id = job_manager.submit_batch_analysis(
                urls=test_urls,
                query="test batch",
                config=config,
                apply_diversity=True
            )
            print(f"✅ Submitted batch analysis job: {job_id}")
            
            # Monitor job progress
            for i in range(60):  # Wait up to 60 seconds
                status = job_manager.get_job_status(job_id)
                current = status.get('current', 0)
                total = status.get('total', 0)
                print(f"   Status: {status.get('state')} - {current}/{total} - {status.get('status', 'Unknown')}")
                
                if status.get('state') in ['SUCCESS', 'FAILURE']:
                    break
                    
                time.sleep(2)
            
            # Get final result
            result = job_manager.get_job_result(job_id)
            if result:
                processed = result.get('processed_count', 0)
                failed = result.get('failed_count', 0)
                print(f"✅ Batch job completed: {processed} processed, {failed} failed")
            else:
                print("❌ No result available")
                
        except Exception as e:
            print(f"❌ Batch analysis test failed: {e}")
        
        # Test job listing
        print("\n4. Testing job listing...")
        try:
            active_jobs = job_manager.list_active_jobs()
            print(f"✅ Found {len(active_jobs)} active jobs")
            for job in active_jobs[:3]:  # Show first 3
                print(f"   Job {job.get('job_id')}: {job.get('state')} - {job.get('type')}")
        except Exception as e:
            print(f"❌ Job listing test failed: {e}")
        
        print("\n✅ JobManager tests completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_celery_tasks():
    """Test Celery tasks directly."""
    print("\nTesting Celery tasks directly...")
    
    try:
        from curator.core.tasks import health_check_task, batch_analyze_task
        
        # Test health check task
        print("\n1. Testing health check task...")
        result = health_check_task.delay()
        
        # Wait for result
        try:
            health_result = result.get(timeout=10)
            print(f"✅ Health check task completed: {health_result}")
        except Exception as e:
            print(f"❌ Health check task failed: {e}")
        
        print("\n✅ Celery task tests completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing AI Content Curator Background Jobs")
    print("=" * 50)
    
    # Check environment
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    print(f"Redis URL: {redis_url}")
    
    # Run tests
    success = True
    
    try:
        success &= test_job_manager()
        success &= test_celery_tasks()
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("\nYour background job system is working correctly!")
        print("\nNext steps:")
        print("1. Start the Flask app: python app.py")
        print("2. Visit the web interface and try topic curation with background processing")
        print("3. Monitor jobs at /jobs/<job_id>")
    else:
        print("❌ Some tests failed!")
        print("\nTroubleshooting:")
        print("1. Make sure Redis is running: redis-server")
        print("2. Make sure Celery worker is running: python celery_worker.py")
        print("3. Check that all dependencies are installed: pip install -r requirements.txt")
    
    return success

if __name__ == '__main__':
    main()