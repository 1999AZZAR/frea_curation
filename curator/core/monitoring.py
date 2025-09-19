"""
Observability and monitoring utilities for AI Content Curator
Provides Sentry error tracking, Prometheus metrics, and structured logging
"""

import os
import time
import functools
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

import structlog
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# Prometheus metrics
ARTICLE_PARSE_COUNTER = Counter(
    'curator_article_parse_total',
    'Total number of article parse attempts',
    ['status', 'source']
)

ARTICLE_PARSE_DURATION = Histogram(
    'curator_article_parse_duration_seconds',
    'Time spent parsing articles',
    ['source']
)

SCORING_COUNTER = Counter(
    'curator_scoring_total',
    'Total number of scoring operations',
    ['component', 'status']
)

SCORING_DURATION = Histogram(
    'curator_scoring_duration_seconds',
    'Time spent on scoring operations',
    ['component']
)

API_REQUEST_COUNTER = Counter(
    'curator_api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']
)

API_REQUEST_DURATION = Histogram(
    'curator_api_request_duration_seconds',
    'Time spent on API requests',
    ['endpoint', 'method']
)

NEWSAPI_REQUEST_COUNTER = Counter(
    'curator_newsapi_requests_total',
    'Total number of NewsAPI requests',
    ['status']
)

NEWSAPI_REQUEST_DURATION = Histogram(
    'curator_newsapi_request_duration_seconds',
    'Time spent on NewsAPI requests'
)

CACHE_OPERATIONS = Counter(
    'curator_cache_operations_total',
    'Total number of cache operations',
    ['operation', 'status']
)

ACTIVE_JOBS = Gauge(
    'curator_active_jobs',
    'Number of active background jobs',
    ['job_type']
)

ERROR_COUNTER = Counter(
    'curator_errors_total',
    'Total number of errors',
    ['error_type', 'component']
)


def init_sentry(app=None):
    """Initialize Sentry error tracking"""
    sentry_dsn = os.environ.get('SENTRY_DSN')
    if not sentry_dsn:
        return
    
    environment = os.environ.get('SENTRY_ENVIRONMENT', 'development')
    release = os.environ.get('SENTRY_RELEASE', 'unknown')
    
    integrations = [FlaskIntegration(), CeleryIntegration()]
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=integrations,
        environment=environment,
        release=release,
        traces_sample_rate=float(os.environ.get('SENTRY_TRACES_SAMPLE_RATE', '0.1')),
        profiles_sample_rate=float(os.environ.get('SENTRY_PROFILES_SAMPLE_RATE', '0.1')),
        attach_stacktrace=True,
        send_default_pii=False,
    )


def init_structured_logging():
    """Initialize structured logging with JSON output"""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    import logging
    logging.basicConfig(level=getattr(logging, log_level))


def get_logger(name: str):
    """Get a structured logger instance"""
    return structlog.get_logger(name)


@contextmanager
def track_operation(operation_name: str, component: str = "general", **extra_context):
    """Context manager to track operation timing and success/failure"""
    logger = get_logger(component)
    start_time = time.time()
    
    try:
        logger.info(
            f"{operation_name} started",
            operation=operation_name,
            component=component,
            **extra_context
        )
        yield
        
        duration = time.time() - start_time
        logger.info(
            f"{operation_name} completed",
            operation=operation_name,
            component=component,
            duration=duration,
            status="success",
            **extra_context
        )
        
    except Exception as e:
        duration = time.time() - start_time
        ERROR_COUNTER.labels(error_type=type(e).__name__, component=component).inc()
        
        logger.error(
            f"{operation_name} failed",
            operation=operation_name,
            component=component,
            duration=duration,
            status="error",
            error=str(e),
            error_type=type(e).__name__,
            **extra_context
        )
        raise


def track_article_parsing(source: str = "unknown"):
    """Decorator to track article parsing operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ARTICLE_PARSE_DURATION.labels(source=source).time():
                try:
                    result = func(*args, **kwargs)
                    ARTICLE_PARSE_COUNTER.labels(status="success", source=source).inc()
                    return result
                except Exception as e:
                    ARTICLE_PARSE_COUNTER.labels(status="error", source=source).inc()
                    ERROR_COUNTER.labels(error_type=type(e).__name__, component="parser").inc()
                    raise
        return wrapper
    return decorator


def track_scoring_operation(component: str):
    """Decorator to track scoring operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with SCORING_DURATION.labels(component=component).time():
                try:
                    result = func(*args, **kwargs)
                    SCORING_COUNTER.labels(component=component, status="success").inc()
                    return result
                except Exception as e:
                    SCORING_COUNTER.labels(component=component, status="error").inc()
                    ERROR_COUNTER.labels(error_type=type(e).__name__, component="scoring").inc()
                    raise
        return wrapper
    return decorator


def track_newsapi_request():
    """Decorator to track NewsAPI requests"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with NEWSAPI_REQUEST_DURATION.time():
                try:
                    result = func(*args, **kwargs)
                    NEWSAPI_REQUEST_COUNTER.labels(status="success").inc()
                    return result
                except Exception as e:
                    NEWSAPI_REQUEST_COUNTER.labels(status="error").inc()
                    ERROR_COUNTER.labels(error_type=type(e).__name__, component="newsapi").inc()
                    raise
        return wrapper
    return decorator


def track_cache_operation(operation: str):
    """Decorator to track cache operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                CACHE_OPERATIONS.labels(operation=operation, status="success").inc()
                return result
            except Exception as e:
                CACHE_OPERATIONS.labels(operation=operation, status="error").inc()
                ERROR_COUNTER.labels(error_type=type(e).__name__, component="cache").inc()
                raise
        return wrapper
    return decorator


def track_api_request(endpoint: str, method: str):
    """Decorator to track Flask API requests"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with API_REQUEST_DURATION.labels(endpoint=endpoint, method=method).time():
                try:
                    result = func(*args, **kwargs)
                    # Determine status from response
                    status = "success"
                    if hasattr(result, 'status_code'):
                        status = "error" if result.status_code >= 400 else "success"
                    API_REQUEST_COUNTER.labels(endpoint=endpoint, method=method, status=status).inc()
                    return result
                except Exception as e:
                    API_REQUEST_COUNTER.labels(endpoint=endpoint, method=method, status="error").inc()
                    ERROR_COUNTER.labels(error_type=type(e).__name__, component="api").inc()
                    raise
        return wrapper
    return decorator


def log_parsing_outcome(url: str, success: bool, error: Optional[str] = None, 
                       word_count: Optional[int] = None, entities_count: Optional[int] = None):
    """Log structured information about article parsing outcomes"""
    logger = get_logger("parser")
    
    log_data = {
        "url": url,
        "success": success,
        "word_count": word_count,
        "entities_count": entities_count,
    }
    
    if error:
        log_data["error"] = error
        logger.error("Article parsing failed", **log_data)
    else:
        logger.info("Article parsing completed", **log_data)


def log_scoring_outcome(url: str, scores: Dict[str, float], overall_score: float, 
                       processing_time: float):
    """Log structured information about scoring outcomes"""
    logger = get_logger("scoring")
    
    logger.info(
        "Article scoring completed",
        url=url,
        overall_score=overall_score,
        component_scores=scores,
        processing_time=processing_time
    )


def log_newsapi_outcome(topic: str, articles_found: int, success: bool, 
                       error: Optional[str] = None):
    """Log structured information about NewsAPI outcomes"""
    logger = get_logger("newsapi")
    
    log_data = {
        "topic": topic,
        "articles_found": articles_found,
        "success": success,
    }
    
    if error:
        log_data["error"] = error
        logger.error("NewsAPI request failed", **log_data)
    else:
        logger.info("NewsAPI request completed", **log_data)


def update_active_jobs(job_type: str, count: int):
    """Update the count of active background jobs"""
    ACTIVE_JOBS.labels(job_type=job_type).set(count)


def get_metrics():
    """Get Prometheus metrics in text format"""
    return generate_latest()


def get_metrics_content_type():
    """Get the content type for Prometheus metrics"""
    return CONTENT_TYPE_LATEST


class MonitoringMiddleware:
    """Flask middleware for automatic request monitoring"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_appcontext(self.teardown)
    
    def before_request(self):
        """Called before each request"""
        from flask import g, request
        g.start_time = time.time()
        g.endpoint = request.endpoint or 'unknown'
        g.method = request.method
    
    def after_request(self, response):
        """Called after each request"""
        from flask import g
        
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            endpoint = getattr(g, 'endpoint', 'unknown')
            method = getattr(g, 'method', 'unknown')
            
            status = "error" if response.status_code >= 400 else "success"
            
            API_REQUEST_COUNTER.labels(
                endpoint=endpoint, 
                method=method, 
                status=status
            ).inc()
            
            API_REQUEST_DURATION.labels(
                endpoint=endpoint, 
                method=method
            ).observe(duration)
        
        return response
    
    def teardown(self, exception):
        """Called when request context is torn down"""
        if exception:
            from flask import g
            endpoint = getattr(g, 'endpoint', 'unknown')
            ERROR_COUNTER.labels(
                error_type=type(exception).__name__, 
                component="flask"
            ).inc()


def init_monitoring(app):
    """Initialize all monitoring components"""
    init_sentry(app)
    init_structured_logging()
    
    # Initialize middleware
    monitoring_middleware = MonitoringMiddleware(app)
    
    # Add metrics endpoint
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint"""
        from flask import Response
        return Response(get_metrics(), mimetype=get_metrics_content_type())
    
    # Add health check endpoint
    @app.route('/health')
    def health():
        """Health check endpoint"""
        from flask import jsonify
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time()
        })