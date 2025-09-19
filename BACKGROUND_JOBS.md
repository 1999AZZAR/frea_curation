# Background Jobs System

The AI Content Curator includes a background job processing system using Celery + Redis for handling long-running analysis tasks asynchronously.

## Overview

The background job system provides:

- **Asynchronous Processing**: Large batch analysis jobs run in the background
- **Progress Tracking**: Real-time progress updates and status monitoring
- **Error Handling**: Graceful error handling with detailed error messages
- **Job Management**: Submit, monitor, cancel, and list jobs via API
- **Web Interface**: User-friendly job status pages with auto-refresh

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flask App     │    │   Redis Broker  │    │  Celery Worker  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Job Manager │◄┼────┼►│ Task Queue  │◄┼────┼►│ Task Runner │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Status API  │◄┼────┼►│ Result Store│ │    │ │ NLP Models  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### 1. Celery Tasks (`curator/core/tasks.py`)

- **`batch_analyze_task`**: Process multiple articles with progress tracking
- **`analyze_single_task`**: Process a single article asynchronously
- **`health_check_task`**: Simple health check for worker status

### 2. Job Manager (`curator/core/job_manager.py`)

- **Job Submission**: Submit analysis jobs to the queue
- **Status Tracking**: Monitor job progress and state
- **Result Retrieval**: Get completed job results
- **Job Management**: List active jobs and cancel running jobs

### 3. Flask Routes (`app.py`)

- **`POST /jobs`**: Submit new background jobs
- **`GET /jobs/<job_id>/status`**: Get job status (API)
- **`GET /jobs/<job_id>`**: Job status page (web interface)
- **`GET /jobs/<job_id>/result`**: Get job result
- **`DELETE /jobs/<job_id>`**: Cancel a job
- **`GET /jobs`**: List active jobs
- **`GET /health/jobs`**: Health check for job system

### 4. Web Interface

- **Job Status Page**: Real-time progress tracking with auto-refresh
- **Background Processing Option**: Checkbox in topic curation form
- **Progress Visualization**: Progress bars and status indicators

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Redis Server

```bash
# Ubuntu/Debian
sudo systemctl start redis-server

# macOS
brew services start redis

# Manual start
redis-server
```

### 3. Start Celery Worker

```bash
# Using the provided script
python celery_worker.py

# Or using the convenience script
./start_workers.sh
```

### 4. Start Flask Application

```bash
python app.py
```

### 5. Test the System

```bash
# Run the test suite
python test_background_jobs.py
```

## API Usage

### Submit a Background Job

```bash
curl -X POST http://localhost:5000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "topic_curation",
    "topic": "artificial intelligence",
    "max_articles": 20,
    "apply_diversity": true
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "submitted",
  "type": "topic_curation",
  "topic": "artificial intelligence",
  "urls_count": 15
}
```

### Check Job Status

```bash
curl http://localhost:5000/jobs/550e8400-e29b-41d4-a716-446655440000/status
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "state": "PROGRESS",
  "status": "Processing batch 2...",
  "current": 8,
  "total": 15,
  "processed_urls": ["https://example.com/article1", "..."],
  "failed_urls": []
}
```

### Get Job Result

```bash
curl http://localhost:5000/jobs/550e8400-e29b-41d4-a716-446655440000/result
```

## Web Interface Usage

### 1. Topic Curation with Background Processing

1. Go to the homepage and select "Curate Topic"
2. Enter your topic keywords
3. Check "Use background processing for large batches"
4. Submit the form
5. You'll be redirected to the job status page

### 2. Job Status Page

- **Real-time Updates**: Page auto-refreshes every 2 seconds
- **Progress Tracking**: Visual progress bar and current/total counts
- **Processing Details**: Lists of processed and failed URLs
- **Results Display**: Full results when job completes
- **Job Control**: Cancel button for running jobs

## Configuration

### Environment Variables

```bash
# Redis connection
REDIS_URL=redis://localhost:6379/0

# Celery worker settings
CELERY_LOG_LEVEL=INFO
CELERY_CONCURRENCY=4
CELERY_ALWAYS_EAGER=False  # Set to True for synchronous testing

# Job processing settings
DIVERSIFY_RESULTS=1
DOMAIN_CAP=2
DUP_SIM_THRESHOLD=0.97
USE_EMBEDDING_CLUSTERING=1
CLUSTER_CAP=3
CLUSTERING_THRESHOLD=0.75
```

### Celery Configuration

The Celery app is configured with:

- **JSON Serialization**: For security and compatibility
- **Task Time Limits**: 5 minute soft limit, 10 minute hard limit
- **Result Expiration**: Results expire after 1 hour
- **Retry Settings**: 3 retries with 60 second delay
- **Worker Settings**: Prefetch multiplier of 1, late acks enabled

## Monitoring and Troubleshooting

### Health Check

```bash
curl http://localhost:5000/health/jobs
```

Response:
```json
{
  "status": "healthy",
  "redis_connected": true,
  "celery_workers": 4,
  "active_jobs": 2
}
```

### Common Issues

1. **Redis Connection Failed**
   - Check if Redis server is running: `redis-cli ping`
   - Verify REDIS_URL environment variable

2. **No Celery Workers**
   - Start worker: `python celery_worker.py`
   - Check worker logs for errors

3. **Jobs Stuck in PENDING**
   - Worker may be overloaded or crashed
   - Restart worker process

4. **Import Errors**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python path and virtual environment

### Logs

- **Flask App**: Standard Flask logging
- **Celery Worker**: Configured via CELERY_LOG_LEVEL
- **Redis**: Redis server logs (usually in `/var/log/redis/`)

## Performance Considerations

### Scaling

- **Multiple Workers**: Run multiple Celery worker processes
- **Worker Concurrency**: Adjust CELERY_CONCURRENCY based on CPU cores
- **Redis Optimization**: Use Redis clustering for high throughput
- **Task Batching**: Process articles in batches to reduce overhead

### Memory Management

- **Worker Recycling**: Workers restart after 1000 tasks to prevent memory leaks
- **NLP Model Caching**: Models are cached globally to reduce memory usage
- **Result Cleanup**: Results expire automatically to prevent Redis bloat

### Monitoring

- **Flower**: Web-based Celery monitoring tool
- **Redis Monitoring**: Use Redis CLI or monitoring tools
- **Application Metrics**: Custom metrics for job success/failure rates

## Security Considerations

- **Input Validation**: All job parameters are validated before processing
- **Resource Limits**: Task time limits prevent runaway processes
- **Error Handling**: Sensitive information is not exposed in error messages
- **Access Control**: Consider adding authentication for job management endpoints

## Future Enhancements

- **Job Scheduling**: Add support for scheduled/recurring jobs
- **Priority Queues**: Different priority levels for urgent vs. batch jobs
- **Job Dependencies**: Chain jobs together with dependencies
- **Distributed Workers**: Support for workers on multiple machines
- **Advanced Monitoring**: Integration with monitoring systems like Prometheus