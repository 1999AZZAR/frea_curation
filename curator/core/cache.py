"""
Redis caching layer for parsed articles and scorecards.

Provides wrapper utilities and cache keys by URL/hash with configurable TTLs.
Implements graceful degradation when Redis is unavailable.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from dataclasses import asdict

from curator.core.models import Article, ScoreCard, Entity

logger = logging.getLogger(__name__)

# Global Redis client instance
_redis_client = None


def get_redis_client():
    """Get Redis client with lazy initialization and graceful degradation."""
    global _redis_client
    
    if _redis_client is not None:
        return _redis_client
    
    try:
        import redis
        
        # Redis configuration from environment
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            _redis_client = redis.from_url(redis_url)
        else:
            redis_host = os.environ.get('REDIS_HOST', 'localhost')
            redis_port = int(os.environ.get('REDIS_PORT', 6379))
            redis_db = int(os.environ.get('REDIS_DB', 0))
            redis_password = os.environ.get('REDIS_PASSWORD')
            
            _redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
        
        # Test connection
        _redis_client.ping()
        logger.info("Redis connection established successfully")
        return _redis_client
        
    except ImportError:
        logger.warning("Redis library not available, caching disabled")
        return None
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}, caching disabled")
        return None


def generate_cache_key(url: str, prefix: str = "") -> str:
    """Generate cache key from URL using SHA-256 hash.
    
    Args:
        url: Article URL to generate key for
        prefix: Optional prefix for key namespacing
        
    Returns:
        Cache key string in format: prefix:sha256_hash
    """
    # Normalize URL for consistent caching
    normalized_url = url.strip().lower()
    
    # Generate SHA-256 hash
    url_hash = hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()
    
    if prefix:
        return f"{prefix}:{url_hash}"
    return url_hash


def serialize_article(article: Article) -> str:
    """Serialize Article object to JSON string for caching.
    
    Args:
        article: Article object to serialize
        
    Returns:
        JSON string representation
    """
    try:
        # Convert to dict and handle datetime serialization
        data = asdict(article)
        
        # Convert datetime to ISO string
        if data.get('publish_date'):
            if isinstance(data['publish_date'], datetime):
                data['publish_date'] = data['publish_date'].isoformat()
        
        # Ensure entities are properly serialized
        if data.get('entities'):
            data['entities'] = [asdict(entity) for entity in article.entities]
        
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to serialize article: {e}")
        raise


def deserialize_article(data: str) -> Article:
    """Deserialize JSON string to Article object.
    
    Args:
        data: JSON string to deserialize
        
    Returns:
        Article object
    """
    try:
        parsed = json.loads(data)
        
        # Convert ISO string back to datetime
        if parsed.get('publish_date'):
            if isinstance(parsed['publish_date'], str):
                parsed['publish_date'] = datetime.fromisoformat(parsed['publish_date'])
        
        # Reconstruct entities
        entities = []
        if parsed.get('entities'):
            for entity_data in parsed['entities']:
                entities.append(Entity(**entity_data))
        parsed['entities'] = entities
        
        return Article(**parsed)
    except Exception as e:
        logger.error(f"Failed to deserialize article: {e}")
        raise


def serialize_scorecard(scorecard: ScoreCard) -> str:
    """Serialize ScoreCard object to JSON string for caching.
    
    Args:
        scorecard: ScoreCard object to serialize
        
    Returns:
        JSON string representation
    """
    try:
        # Convert to dict and handle nested Article serialization
        data = asdict(scorecard)
        
        # Handle Article serialization within ScoreCard
        if data.get('article'):
            article_data = data['article']
            
            # Convert datetime to ISO string
            if article_data.get('publish_date'):
                if isinstance(scorecard.article.publish_date, datetime):
                    article_data['publish_date'] = scorecard.article.publish_date.isoformat()
            
            # Ensure entities are properly serialized
            if article_data.get('entities'):
                article_data['entities'] = [asdict(entity) for entity in scorecard.article.entities]
        
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to serialize scorecard: {e}")
        raise


def deserialize_scorecard(data: str) -> ScoreCard:
    """Deserialize JSON string to ScoreCard object.
    
    Args:
        data: JSON string to deserialize
        
    Returns:
        ScoreCard object
    """
    try:
        parsed = json.loads(data)
        
        # Reconstruct Article within ScoreCard
        if parsed.get('article'):
            article_data = parsed['article']
            
            # Convert ISO string back to datetime
            if article_data.get('publish_date'):
                if isinstance(article_data['publish_date'], str):
                    article_data['publish_date'] = datetime.fromisoformat(article_data['publish_date'])
            
            # Reconstruct entities
            entities = []
            if article_data.get('entities'):
                for entity_data in article_data['entities']:
                    entities.append(Entity(**entity_data))
            article_data['entities'] = entities
            
            parsed['article'] = Article(**article_data)
        
        return ScoreCard(**parsed)
    except Exception as e:
        logger.error(f"Failed to deserialize scorecard: {e}")
        raise


class CacheManager:
    """Redis cache manager with TTL support and graceful degradation."""
    
    def __init__(self):
        self.client = get_redis_client()
        
        # Default TTLs in seconds
        self.article_ttl = int(os.environ.get('CACHE_ARTICLE_TTL', 3600))  # 1 hour
        self.scorecard_ttl = int(os.environ.get('CACHE_SCORECARD_TTL', 1800))  # 30 minutes
    
    def is_available(self) -> bool:
        """Check if Redis caching is available."""
        return self.client is not None
    
    def get_article(self, url: str) -> Optional[Article]:
        """Get cached article by URL.
        
        Args:
            url: Article URL to lookup
            
        Returns:
            Cached Article object or None if not found/unavailable
        """
        if not self.is_available():
            return None
        
        try:
            key = generate_cache_key(url, "article")
            data = self.client.get(key)
            
            if data:
                logger.debug(f"Cache hit for article: {url}")
                return deserialize_article(data)
            else:
                logger.debug(f"Cache miss for article: {url}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get cached article for {url}: {e}")
            return None
    
    def set_article(self, url: str, article: Article, ttl: Optional[int] = None) -> bool:
        """Cache article with TTL.
        
        Args:
            url: Article URL as cache key
            article: Article object to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            key = generate_cache_key(url, "article")
            data = serialize_article(article)
            ttl = ttl or self.article_ttl
            
            result = self.client.setex(key, ttl, data)
            if result:
                logger.debug(f"Cached article: {url} (TTL: {ttl}s)")
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Failed to cache article for {url}: {e}")
            return False
    
    def get_scorecard(self, url: str, query: str = "") -> Optional[ScoreCard]:
        """Get cached scorecard by URL and query.
        
        Args:
            url: Article URL
            query: Query string used for relevance scoring
            
        Returns:
            Cached ScoreCard object or None if not found/unavailable
        """
        if not self.is_available():
            return None
        
        try:
            # Include query in cache key for relevance-specific caching
            cache_input = f"{url}|{query}"
            key = generate_cache_key(cache_input, "scorecard")
            data = self.client.get(key)
            
            if data:
                logger.debug(f"Cache hit for scorecard: {url} (query: {query})")
                return deserialize_scorecard(data)
            else:
                logger.debug(f"Cache miss for scorecard: {url} (query: {query})")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get cached scorecard for {url}: {e}")
            return None
    
    def set_scorecard(self, url: str, scorecard: ScoreCard, query: str = "", ttl: Optional[int] = None) -> bool:
        """Cache scorecard with TTL.
        
        Args:
            url: Article URL
            scorecard: ScoreCard object to cache
            query: Query string used for relevance scoring
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            # Include query in cache key for relevance-specific caching
            cache_input = f"{url}|{query}"
            key = generate_cache_key(cache_input, "scorecard")
            data = serialize_scorecard(scorecard)
            ttl = ttl or self.scorecard_ttl
            
            result = self.client.setex(key, ttl, data)
            if result:
                logger.debug(f"Cached scorecard: {url} (query: {query}, TTL: {ttl}s)")
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Failed to cache scorecard for {url}: {e}")
            return False
    
    def delete_article(self, url: str) -> bool:
        """Delete cached article.
        
        Args:
            url: Article URL to delete from cache
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            key = generate_cache_key(url, "article")
            result = self.client.delete(key)
            if result:
                logger.debug(f"Deleted cached article: {url}")
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Failed to delete cached article for {url}: {e}")
            return False
    
    def delete_scorecard(self, url: str, query: str = "") -> bool:
        """Delete cached scorecard.
        
        Args:
            url: Article URL
            query: Query string used for relevance scoring
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            cache_input = f"{url}|{query}"
            key = generate_cache_key(cache_input, "scorecard")
            result = self.client.delete(key)
            if result:
                logger.debug(f"Deleted cached scorecard: {url} (query: {query})")
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Failed to delete cached scorecard for {url}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all cached data (use with caution).
        
        Returns:
            True if successfully cleared, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            # Delete all keys with our prefixes
            article_keys = self.client.keys("article:*")
            scorecard_keys = self.client.keys("scorecard:*")
            
            all_keys = article_keys + scorecard_keys
            if all_keys:
                result = self.client.delete(*all_keys)
                logger.info(f"Cleared {result} cached items")
                return True
            return True
            
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and health information.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.is_available():
            return {"available": False, "error": "Redis not available"}
        
        try:
            info = self.client.info()
            article_count = len(self.client.keys("article:*"))
            scorecard_count = len(self.client.keys("scorecard:*"))
            
            return {
                "available": True,
                "article_count": article_count,
                "scorecard_count": scorecard_count,
                "total_keys": article_count + scorecard_count,
                "memory_usage": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }
            
        except Exception as e:
            return {"available": False, "error": str(e)}


# Global cache manager instance
cache_manager = CacheManager()


def get_cached_article(url: str) -> Optional[Article]:
    """Convenience function to get cached article."""
    return cache_manager.get_article(url)


def cache_article(url: str, article: Article, ttl: Optional[int] = None) -> bool:
    """Convenience function to cache article."""
    return cache_manager.set_article(url, article, ttl)


def get_cached_scorecard(url: str, query: str = "") -> Optional[ScoreCard]:
    """Convenience function to get cached scorecard."""
    return cache_manager.get_scorecard(url, query)


def cache_scorecard(url: str, scorecard: ScoreCard, query: str = "", ttl: Optional[int] = None) -> bool:
    """Convenience function to cache scorecard."""
    return cache_manager.set_scorecard(url, scorecard, query, ttl)