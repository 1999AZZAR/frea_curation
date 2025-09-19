#!/bin/bash
"""
Development script to start Redis and Celery worker.

This script starts Redis server and Celery worker for local development.
Make sure Redis is installed on your system.
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AI Content Curator Background Services${NC}"

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo -e "${RED}Error: Redis is not installed${NC}"
    echo "Please install Redis:"
    echo "  Ubuntu/Debian: sudo apt-get install redis-server"
    echo "  macOS: brew install redis"
    echo "  CentOS/RHEL: sudo yum install redis"
    exit 1
fi

# Check if Python virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo "Consider activating your virtual environment first:"
    echo "  source .venv/bin/activate"
fi

# Set default environment variables
export REDIS_URL=${REDIS_URL:-"redis://localhost:6379/0"}
export CELERY_LOG_LEVEL=${CELERY_LOG_LEVEL:-"INFO"}
export CELERY_CONCURRENCY=${CELERY_CONCURRENCY:-"4"}

echo -e "${GREEN}Configuration:${NC}"
echo "  Redis URL: $REDIS_URL"
echo "  Log Level: $CELERY_LOG_LEVEL"
echo "  Concurrency: $CELERY_CONCURRENCY"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [[ ! -z "$REDIS_PID" ]]; then
        kill $REDIS_PID 2>/dev/null || true
        echo "Redis stopped"
    fi
    if [[ ! -z "$CELERY_PID" ]]; then
        kill $CELERY_PID 2>/dev/null || true
        echo "Celery worker stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Redis server in background
echo -e "${GREEN}Starting Redis server...${NC}"
redis-server --daemonize yes --port 6379 --bind 127.0.0.1
REDIS_PID=$(pgrep redis-server | head -1)

# Wait a moment for Redis to start
sleep 2

# Test Redis connection
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}Redis is running (PID: $REDIS_PID)${NC}"
else
    echo -e "${RED}Failed to start Redis${NC}"
    exit 1
fi

# Start Celery worker
echo -e "${GREEN}Starting Celery worker...${NC}"
python celery_worker.py &
CELERY_PID=$!

echo -e "${GREEN}Celery worker started (PID: $CELERY_PID)${NC}"
echo ""
echo -e "${GREEN}Background services are running!${NC}"
echo "Press Ctrl+C to stop all services"
echo ""
echo "You can now:"
echo "  1. Start the Flask app: python app.py"
echo "  2. Submit background jobs via the web interface"
echo "  3. Monitor job status at /jobs/<job_id>"
echo ""

# Wait for Celery worker to finish (or be interrupted)
wait $CELERY_PID