# Redis Caching Layer

The AI Content Curator includes a Redis-based caching layer that significantly improves performance by avoiding redundant article parsing and scoring operations.

## Overview

The caching system stores two types of data:
- **Parsed Articles**: Raw article content, metadata, and entities
- **Scorecards**: Complete analysis results including all scoring components

This two-tier approach maximizes cache efficiency:
1. Articles are cached after parsing to avoid expensive network requests
2. Scorecards are cached per query to avoid recomputing scores

## Features

- **Graceful Degradation**: System works normally when Redis is unavailable
- **TTL Support**: Configurable expiration times for cached data
- **Query-Specific Caching**: Scorecards are cached separately for different queries
- **Error Handling**: Robust error handling with logging
- **Cache Statistics**: Built-in monitoring and health checks

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Option 1: Redis URL (recommended for production)
REDIS_URL=redis://localhost:6379/0

# Option 2: Individual Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your-redis-password

# Cache TTL settings (in seconds)
CACHE_ARTICLE_TTL=3600    # 1 hour
CACHE_SCORECARD_TTL=1800  # 30 minutes
```

### Redis Installation

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### macOS
```bash
brew install redis
brew services start redis
```

#### Docker
```bash
docker run -d -p 6379:6379 --name redis redis:alpine
```

## Usage

### Automatic Caching

Caching is automatically enabled when Redis is available. No code changes required:

```python
from curator.services._analyzer import analyze_article, batch_analyze

# First call - parses and caches
result1 = analyze_article("https://example.com/article", query="AI")

# Second call - uses cache (much faster)
result2 = analyze_article("https://example.com/article", query="AI")

# Different query - creates separate cache entry
result3 = analyze_article("https://example.com/article", query="ML")
```

### Manual Cache Operations

```python
from curator.core.cache import (
    get_cached_article, cache_article,
    get_cached_scorecard, cache_scorecard,
    cache_manager
)

# Check if article is cached
article = get_cached_article("https://example.com/article")

# Cache an article manually
cache_article("https://example.com/article", article_object)

# Get cache statistics
stats = cache_manager.get_cache_stats()
print(f"Cached articles: {stats['article_count']}")
print(f"Memory usage: {stats['memory_usage']}")
```

## Cache Keys

The system generates SHA-256 hashes for cache keys:

- **Articles**: `article:{sha256(url)}`
- **Scorecards**: `scorecard:{sha256(url + "|" + query)}`

This ensures:
- Consistent key generation
- No key collisions
- Query-specific scorecard caching

## Performance Benefits

### Single Article Analysis
- **Cold cache**: Full parsing + scoring (~2-5 seconds)
- **Warm cache**: Direct retrieval (~50-100ms)
- **Speed improvement**: 20-50x faster

### Batch Analysis
- **Mixed cache hits**: Only processes uncached articles
- **Full cache hits**: Near-instantaneous results
- **Memory efficiency**: Shared article cache across queries

## Monitoring

### Cache Statistics

```python
from curator.core.cache import cache_manager

stats = cache_manager.get_cache_stats()
print(f"""
Cache Status: {stats['available']}
Articles: {stats['article_count']}
Scorecards: {stats['scorecard_count']}
Memory: {stats['memory_usage']}
Uptime: {stats['uptime_seconds']}s
""")
```

### Logging

The cache system logs operations at different levels:

```python
import logging
logging.getLogger('curator.core.cache').setLevel(logging.DEBUG)
```

Log messages include:
- Cache hits/misses
- Serialization errors
- Redis connection issues
- Performance metrics

## Error Handling

The caching layer implements comprehensive error handling:

### Redis Unavailable
- System continues without caching
- Warning logged once at startup
- No performance impact on core functionality

### Serialization Errors
- Individual cache operations fail gracefully
- Errors logged with context
- System continues processing

### Network Issues
- Automatic retry with exponential backoff
- Circuit breaker pattern for repeated failures
- Fallback to non-cached operation

## Best Practices

### Production Deployment

1. **Use Redis Sentinel** for high availability
2. **Configure memory limits** to prevent OOM
3. **Monitor cache hit rates** for optimization
4. **Set appropriate TTLs** based on content freshness needs

### Development

1. **Use local Redis** for testing
2. **Clear cache** when changing scoring algorithms
3. **Monitor logs** for cache-related issues
4. **Test graceful degradation** by stopping Redis

### Performance Tuning

1. **Adjust TTLs** based on usage patterns:
   - Longer for stable content
   - Shorter for frequently updated sources
2. **Monitor memory usage** and eviction policies
3. **Use Redis clustering** for high-volume deployments

## Cache Management

### Clear All Cache
```python
from curator.core.cache import cache_manager
cache_manager.clear_all()
```

### Clear Specific Items
```python
# Clear article cache
cache_manager.delete_article("https://example.com/article")

# Clear scorecard cache
cache_manager.delete_scorecard("https://example.com/article", "query")
```

### Health Checks
```python
# Check if Redis is available
if cache_manager.is_available():
    print("Cache is healthy")
else:
    print("Cache is unavailable")
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis is running: `redis-cli ping`
   - Verify connection settings in `.env`
   - Check firewall/network connectivity

2. **High Memory Usage**
   - Monitor with `redis-cli info memory`
   - Adjust TTL settings to reduce retention
   - Configure Redis maxmemory policy

3. **Cache Misses**
   - Check TTL settings aren't too short
   - Verify URL normalization is consistent
   - Monitor for serialization errors in logs

4. **Performance Issues**
   - Check Redis latency: `redis-cli --latency`
   - Monitor network between app and Redis
   - Consider Redis clustering for scale

### Debug Mode

Enable debug logging to troubleshoot cache issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('curator.core.cache')
logger.setLevel(logging.DEBUG)
```

## Example Scripts

See `examples/cache_example.py` for a complete demonstration of caching functionality.

## Requirements

- Redis 5.0+ (recommended: 6.0+)
- Python `redis` package (included in requirements.txt)
- Network connectivity between application and Redis

## Security Considerations

- Use Redis AUTH in production environments
- Configure Redis to bind to specific interfaces
- Use TLS for Redis connections over networks
- Regularly update Redis to latest stable version
- Monitor for unauthorized access attempts