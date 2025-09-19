"""
Celery background tasks for content analysis.

Provides asynchronous processing capabilities for batch article analysis
with progress tracking and error handling.
"""

import logging
from typing import List, Optional, Dict, Any
from celery import current_task
from celery.exceptions import Retry

from curator.core.celery_app import celery_app
from curator.core.models import ScoringConfig, ScoreCard
from curator.services.analyzer import batch_analyze
from curator.core.nlp import get_spacy_model, get_vader_analyzer

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='curator.batch_analyze_task')
def batch_analyze_task(
    self,
    urls: List[str],
    query: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    apply_diversity: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Background task for batch article analysis with progress tracking.
    
    Args:
        urls: List of article URLs to analyze
        query: Optional query for relevance scoring
        config_dict: Scoring configuration as dictionary
        apply_diversity: Whether to apply diversity filtering
        
    Returns:
        Dictionary containing results and metadata
        
    Raises:
        Retry: If temporary failures occur and retries are available
    """
    try:
        # Update task state to indicate processing has started
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': len(urls),
                'status': 'Initializing analysis...',
                'processed_urls': [],
                'failed_urls': [],
            }
        )
        
        # Load configuration
        if config_dict:
            config = ScoringConfig(**config_dict)
        else:
            try:
                from curator.core.config import load_scoring_config
                config = load_scoring_config()
            except Exception:
                from config import load_scoring_config
                config = load_scoring_config()
        
        # Initialize NLP components
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': len(urls),
                'status': 'Loading NLP models...',
                'processed_urls': [],
                'failed_urls': [],
            }
        )
        
        nlp_model = get_spacy_model()
        vader = get_vader_analyzer()
        
        # Process articles in batches to provide progress updates
        batch_size = min(10, max(1, len(urls) // 10))  # Process in 10% chunks, min 1, max 10
        results = []
        processed_count = 0
        failed_urls = []
        
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            
            try:
                # Update progress
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': processed_count,
                        'total': len(urls),
                        'status': f'Processing batch {i//batch_size + 1}...',
                        'processed_urls': [r.article.url for r in results],
                        'failed_urls': failed_urls,
                    }
                )
                
                # Process batch
                batch_results = batch_analyze(
                    urls=batch_urls,
                    query=query,
                    config=config,
                    nlp=nlp_model,
                    vader_analyzer=vader,
                    apply_diversity=False  # Apply diversity at the end
                )
                
                results.extend(batch_results)
                processed_count += len(batch_urls)
                
                logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_results)}/{len(batch_urls)} articles")
                
            except Exception as e:
                logger.error(f"Batch processing failed for URLs {batch_urls}: {e}")
                failed_urls.extend(batch_urls)
                processed_count += len(batch_urls)
                
                # Continue with other batches rather than failing entirely
                continue
        
        # Apply diversity filtering if requested
        if apply_diversity and results:
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': processed_count,
                    'total': len(urls),
                    'status': 'Applying diversity filtering...',
                    'processed_urls': [r.article.url for r in results],
                    'failed_urls': failed_urls,
                }
            )
            
            try:
                from curator.services._analyzer import _apply_diversity_and_dedup
                results = _apply_diversity_and_dedup(results)
            except Exception as e:
                logger.warning(f"Diversity filtering failed: {e}")
        
        # Sort results by overall score
        results.sort(key=lambda r: r.overall_score, reverse=True)
        
        # Convert results to serializable format
        serialized_results = []
        for result in results:
            try:
                serialized_results.append({
                    'overall_score': result.overall_score,
                    'readability_score': result.readability_score,
                    'ner_density_score': result.ner_density_score,
                    'sentiment_score': result.sentiment_score,
                    'tfidf_relevance_score': result.tfidf_relevance_score,
                    'recency_score': result.recency_score,
                    'reputation_score': getattr(result, 'reputation_score', 0.0),
                    'topic_coherence_score': getattr(result, 'topic_coherence_score', 0.0),
                    'article': {
                        'url': result.article.url,
                        'title': result.article.title or '',
                        'author': result.article.author or '',
                        'summary': result.article.summary or '',
                        'publish_date': result.article.publish_date.isoformat() if result.article.publish_date else None,
                        'entities': [
                            {'text': e.text, 'label': e.label, 'confidence': e.confidence}
                            for e in (result.article.entities or [])
                        ],
                    }
                })
            except Exception as e:
                logger.error(f"Failed to serialize result for {result.article.url}: {e}")
                continue
        
        # Final result
        final_result = {
            'status': 'SUCCESS',
            'total_urls': len(urls),
            'processed_count': len(results),
            'failed_count': len(failed_urls),
            'failed_urls': failed_urls,
            'results': serialized_results,
            'query': query,
            'diversity_applied': bool(apply_diversity),
        }
        
        logger.info(f"Batch analysis completed: {len(results)}/{len(urls)} articles processed successfully")
        
        return final_result
        
    except Exception as e:
        logger.error(f"Batch analysis task failed: {e}")
        
        # Update task state to indicate failure
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'status': 'Task failed',
            }
        )
        
        # Re-raise the exception to mark task as failed
        raise


@celery_app.task(bind=True, name='curator.analyze_single_task')
def analyze_single_task(
    self,
    url: str,
    query: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Background task for single article analysis.
    
    Args:
        url: Article URL to analyze
        query: Optional query for relevance scoring
        config_dict: Scoring configuration as dictionary
        
    Returns:
        Dictionary containing analysis result
    """
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Analyzing article...',
                'url': url,
            }
        )
        
        # Load configuration
        if config_dict:
            config = ScoringConfig(**config_dict)
        else:
            try:
                from curator.core.config import load_scoring_config
                config = load_scoring_config()
            except Exception:
                from config import load_scoring_config
                config = load_scoring_config()
        
        # Initialize NLP components
        nlp_model = get_spacy_model()
        vader = get_vader_analyzer()
        
        # Analyze article
        from curator.services.analyzer import analyze_article
        result = analyze_article(
            url=url,
            query=query,
            config=config,
            nlp=nlp_model,
            vader_analyzer=vader,
        )
        
        # Convert to serializable format
        serialized_result = {
            'overall_score': result.overall_score,
            'readability_score': result.readability_score,
            'ner_density_score': result.ner_density_score,
            'sentiment_score': result.sentiment_score,
            'tfidf_relevance_score': result.tfidf_relevance_score,
            'recency_score': result.recency_score,
            'reputation_score': getattr(result, 'reputation_score', 0.0),
            'topic_coherence_score': getattr(result, 'topic_coherence_score', 0.0),
            'article': {
                'url': result.article.url,
                'title': result.article.title or '',
                'author': result.article.author or '',
                'summary': result.article.summary or '',
                'publish_date': result.article.publish_date.isoformat() if result.article.publish_date else None,
                'entities': [
                    {'text': e.text, 'label': e.label, 'confidence': e.confidence}
                    for e in (result.article.entities or [])
                ],
            }
        }
        
        return {
            'status': 'SUCCESS',
            'result': serialized_result,
            'url': url,
            'query': query,
        }
        
    except Exception as e:
        logger.error(f"Single analysis task failed for {url}: {e}")
        
        # Update task state to indicate failure
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'status': 'Analysis failed',
                'url': url,
            }
        )
        
        # Re-raise the exception to mark task as failed
        raise


@celery_app.task(name='curator.health_check')
def health_check_task() -> Dict[str, Any]:
    """
    Simple health check task to verify Celery worker is functioning.
    
    Returns:
        Dictionary with health status information
    """
    import time
    from datetime import datetime
    
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'worker_id': current_task.request.id if current_task else None,
    }