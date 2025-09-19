"""
User feedback collection and learning-to-rank system for AI Content Curator.
Handles user interactions (clicks, saves, likes) and uses them to improve scoring.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from curator.core.database import (
    get_db_session, UserFeedback, ScoringWeights, 
    Article, ScoreCard
)
from curator.core.models import ScoringConfig

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """Data class for feedback events."""
    article_url: str
    session_id: str
    feedback_type: str  # 'click', 'save', 'like', 'share'
    query_context: Optional[str] = None
    position_in_results: Optional[int] = None
    total_results: Optional[int] = None
    time_on_page: Optional[float] = None
    user_id: Optional[str] = None


@dataclass
class LearningMetrics:
    """Metrics for evaluating learning-to-rank performance."""
    ndcg_score: float  # Normalized Discounted Cumulative Gain
    map_score: float   # Mean Average Precision
    click_through_rate: float
    total_samples: int
    version: str


class FeedbackCollector:
    """Collects and stores user feedback on articles."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.FeedbackCollector')
    
    def record_feedback(self, event: FeedbackEvent) -> bool:
        """Record a user feedback event."""
        try:
            with get_db_session() as session:
                # Find the article
                article = session.query(Article).filter_by(url=event.article_url).first()
                if not article:
                    self.logger.warning(f"Article not found for feedback: {event.article_url}")
                    return False
                
                # Find the most recent scorecard for this article and query context
                scorecard = None
                if event.query_context:
                    scorecard = session.query(ScoreCard).filter(
                        and_(
                            ScoreCard.article_id == article.id,
                            ScoreCard.query_context == event.query_context
                        )
                    ).order_by(desc(ScoreCard.created_at)).first()
                
                if not scorecard:
                    # Fallback to most recent scorecard for this article
                    scorecard = session.query(ScoreCard).filter_by(
                        article_id=article.id
                    ).order_by(desc(ScoreCard.created_at)).first()
                
                # Check if feedback already exists for this session/article
                existing_feedback = session.query(UserFeedback).filter(
                    and_(
                        UserFeedback.article_id == article.id,
                        UserFeedback.session_id == event.session_id
                    )
                ).first()
                
                if existing_feedback:
                    # Update existing feedback
                    if event.feedback_type == 'click':
                        existing_feedback.clicked = True
                    elif event.feedback_type == 'save':
                        existing_feedback.saved = True
                    elif event.feedback_type == 'like':
                        existing_feedback.liked = True
                    elif event.feedback_type == 'share':
                        existing_feedback.shared = True
                    
                    if event.time_on_page is not None:
                        existing_feedback.time_on_page = event.time_on_page
                    
                    if event.position_in_results is not None:
                        existing_feedback.position_in_results = event.position_in_results
                    
                    if event.total_results is not None:
                        existing_feedback.total_results = event.total_results
                        
                else:
                    # Create new feedback record
                    feedback = UserFeedback(
                        article_id=article.id,
                        scorecard_id=scorecard.id if scorecard else None,
                        session_id=event.session_id,
                        user_id=event.user_id,
                        clicked=(event.feedback_type == 'click'),
                        saved=(event.feedback_type == 'save'),
                        liked=(event.feedback_type == 'like'),
                        shared=(event.feedback_type == 'share'),
                        query_context=event.query_context,
                        position_in_results=event.position_in_results,
                        total_results=event.total_results,
                        time_on_page=event.time_on_page
                    )
                    session.add(feedback)
                
                session.commit()
                self.logger.info(f"Recorded {event.feedback_type} feedback for article {event.article_url}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to record feedback: {str(e)}")
            return False
    
    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback statistics for the last N days."""
        try:
            with get_db_session() as session:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                # Total feedback events
                total_feedback = session.query(UserFeedback).filter(
                    UserFeedback.created_at >= cutoff_date
                ).count()
                
                # Feedback by type
                clicks = session.query(UserFeedback).filter(
                    and_(
                        UserFeedback.created_at >= cutoff_date,
                        UserFeedback.clicked == True
                    )
                ).count()
                
                saves = session.query(UserFeedback).filter(
                    and_(
                        UserFeedback.created_at >= cutoff_date,
                        UserFeedback.saved == True
                    )
                ).count()
                
                likes = session.query(UserFeedback).filter(
                    and_(
                        UserFeedback.created_at >= cutoff_date,
                        UserFeedback.liked == True
                    )
                ).count()
                
                shares = session.query(UserFeedback).filter(
                    and_(
                        UserFeedback.created_at >= cutoff_date,
                        UserFeedback.shared == True
                    )
                ).count()
                
                # Unique sessions and articles
                unique_sessions = session.query(UserFeedback.session_id).filter(
                    UserFeedback.created_at >= cutoff_date
                ).distinct().count()
                
                unique_articles = session.query(UserFeedback.article_id).filter(
                    UserFeedback.created_at >= cutoff_date
                ).distinct().count()
                
                return {
                    'total_feedback': total_feedback,
                    'clicks': clicks,
                    'saves': saves,
                    'likes': likes,
                    'shares': shares,
                    'unique_sessions': unique_sessions,
                    'unique_articles': unique_articles,
                    'days': days
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get feedback stats: {str(e)}")
            return {}


class LearningToRank:
    """Learning-to-rank system that optimizes scoring weights based on user feedback."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.LearningToRank')
    
    def collect_training_data(self, min_samples: int = 100) -> List[Dict[str, Any]]:
        """Collect training data from user feedback."""
        try:
            with get_db_session() as session:
                # Get feedback with associated scorecards
                query = session.query(
                    UserFeedback,
                    ScoreCard,
                    Article
                ).join(
                    ScoreCard, UserFeedback.scorecard_id == ScoreCard.id
                ).join(
                    Article, UserFeedback.article_id == Article.id
                ).filter(
                    UserFeedback.scorecard_id.isnot(None)
                )
                
                feedback_data = query.all()
                
                if len(feedback_data) < min_samples:
                    self.logger.warning(f"Insufficient training data: {len(feedback_data)} < {min_samples}")
                    return []
                
                training_samples = []
                for feedback, scorecard, article in feedback_data:
                    # Calculate relevance score based on user actions
                    relevance_score = self._calculate_relevance_score(feedback)
                    
                    sample = {
                        'features': {
                            'readability_score': scorecard.readability_score,
                            'ner_density_score': scorecard.ner_density_score,
                            'sentiment_score': scorecard.sentiment_score,
                            'tfidf_relevance_score': scorecard.tfidf_relevance_score,
                            'recency_score': scorecard.recency_score,
                            'reputation_score': scorecard.reputation_score,
                            'topic_coherence_score': scorecard.topic_coherence_score,
                        },
                        'relevance': relevance_score,
                        'position': feedback.position_in_results or 0,
                        'query_context': feedback.query_context or '',
                        'session_id': feedback.session_id,
                        'article_url': article.url
                    }
                    training_samples.append(sample)
                
                self.logger.info(f"Collected {len(training_samples)} training samples")
                return training_samples
                
        except Exception as e:
            self.logger.error(f"Failed to collect training data: {str(e)}")
            return []
    
    def _calculate_relevance_score(self, feedback: UserFeedback) -> float:
        """Calculate relevance score based on user feedback."""
        score = 0.0
        
        # Weight different types of feedback
        if feedback.clicked:
            score += 1.0
        if feedback.saved:
            score += 2.0  # Saving indicates higher relevance
        if feedback.liked:
            score += 2.5  # Liking indicates high relevance
        if feedback.shared:
            score += 3.0  # Sharing indicates highest relevance
        
        # Consider time on page (if available)
        if feedback.time_on_page:
            if feedback.time_on_page > 30:  # More than 30 seconds
                score += 1.0
            if feedback.time_on_page > 120:  # More than 2 minutes
                score += 1.0
        
        # Consider position bias (lower positions need higher engagement to be relevant)
        if feedback.position_in_results and feedback.position_in_results > 3:
            score *= 1.2  # Boost score for lower-positioned items that got engagement
        
        # Normalize to 0-5 scale
        return min(5.0, score)
    
    def optimize_weights(self, training_data: List[Dict[str, Any]]) -> Optional[ScoringConfig]:
        """Optimize scoring weights using training data."""
        if not training_data:
            self.logger.warning("No training data available for weight optimization")
            return None
        
        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import mean_squared_error
            
            # Prepare features and targets
            features = []
            targets = []
            
            for sample in training_data:
                feature_vector = [
                    sample['features']['readability_score'],
                    sample['features']['ner_density_score'],
                    sample['features']['sentiment_score'],
                    sample['features']['tfidf_relevance_score'],
                    sample['features']['recency_score'],
                    sample['features']['reputation_score'],
                    sample['features']['topic_coherence_score'],
                ]
                features.append(feature_vector)
                targets.append(sample['relevance'])
            
            X = np.array(features)
            y = np.array(targets)
            
            # Train linear regression model to learn optimal weights
            model = LinearRegression(fit_intercept=False, positive=True)
            model.fit(X, y)
            
            # Get learned weights
            raw_weights = model.coef_
            
            # Normalize weights to sum to 1.0
            weight_sum = np.sum(raw_weights)
            if weight_sum > 0:
                normalized_weights = raw_weights / weight_sum
            else:
                self.logger.warning("All weights are zero, using default weights")
                return None
            
            # Create new scoring config
            new_config = ScoringConfig(
                readability_weight=float(normalized_weights[0]),
                ner_density_weight=float(normalized_weights[1]),
                sentiment_weight=float(normalized_weights[2]),
                tfidf_relevance_weight=float(normalized_weights[3]),
                recency_weight=float(normalized_weights[4]),
                reputation_weight=float(normalized_weights[5]),
                topic_coherence_weight=float(normalized_weights[6])
            )
            
            # Evaluate performance using cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            performance_score = float(-np.mean(cv_scores))
            
            self.logger.info(f"Optimized weights with performance score: {performance_score}")
            self.logger.info(f"New weights: {normalized_weights}")
            
            return new_config, performance_score
            
        except ImportError:
            self.logger.error("scikit-learn not available for weight optimization")
            return None
        except Exception as e:
            self.logger.error(f"Failed to optimize weights: {str(e)}")
            return None
    
    def save_weights(self, config: ScoringConfig, performance_score: float, 
                    training_samples: int, notes: str = None) -> str:
        """Save optimized weights to database."""
        try:
            with get_db_session() as session:
                # Generate version string
                version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                
                # Deactivate previous active weights
                session.query(ScoringWeights).filter_by(is_active=True).update(
                    {'is_active': False}
                )
                
                # Create new weights record
                weights = ScoringWeights(
                    version=version,
                    readability_weight=config.readability_weight,
                    ner_density_weight=config.ner_density_weight,
                    sentiment_weight=config.sentiment_weight,
                    tfidf_relevance_weight=config.tfidf_relevance_weight,
                    recency_weight=config.recency_weight,
                    reputation_weight=config.reputation_weight,
                    topic_coherence_weight=config.topic_coherence_weight,
                    training_samples=training_samples,
                    performance_score=performance_score,
                    is_active=True,
                    notes=notes
                )
                
                session.add(weights)
                session.commit()
                
                self.logger.info(f"Saved new weights version: {version}")
                return version
                
        except Exception as e:
            self.logger.error(f"Failed to save weights: {str(e)}")
            raise
    
    def get_active_weights(self) -> Optional[ScoringConfig]:
        """Get the currently active scoring weights."""
        try:
            with get_db_session() as session:
                weights = session.query(ScoringWeights).filter_by(is_active=True).first()
                
                if weights:
                    return ScoringConfig(
                        readability_weight=weights.readability_weight,
                        ner_density_weight=weights.ner_density_weight,
                        sentiment_weight=weights.sentiment_weight,
                        tfidf_relevance_weight=weights.tfidf_relevance_weight,
                        recency_weight=weights.recency_weight,
                        reputation_weight=weights.reputation_weight,
                        topic_coherence_weight=weights.topic_coherence_weight
                    )
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get active weights: {str(e)}")
            return None
    
    def run_learning_cycle(self, min_samples: int = 100) -> Optional[LearningMetrics]:
        """Run a complete learning cycle: collect data, optimize weights, evaluate."""
        try:
            self.logger.info("Starting learning-to-rank cycle")
            
            # Collect training data
            training_data = self.collect_training_data(min_samples)
            if not training_data:
                return None
            
            # Optimize weights
            result = self.optimize_weights(training_data)
            if not result:
                return None
            
            new_config, performance_score = result
            
            # Save new weights
            version = self.save_weights(
                config=new_config,
                performance_score=performance_score,
                training_samples=len(training_data),
                notes=f"Automated learning cycle with {len(training_data)} samples"
            )
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(training_data, new_config)
            metrics.version = version
            metrics.total_samples = len(training_data)
            
            self.logger.info(f"Learning cycle completed. Version: {version}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Learning cycle failed: {str(e)}")
            return None
    
    def _calculate_metrics(self, training_data: List[Dict[str, Any]], 
                          config: ScoringConfig) -> LearningMetrics:
        """Calculate evaluation metrics for the learned model."""
        try:
            # Group by query context for ranking evaluation
            queries = {}
            for sample in training_data:
                query = sample['query_context'] or 'default'
                if query not in queries:
                    queries[query] = []
                queries[query].append(sample)
            
            ndcg_scores = []
            map_scores = []
            total_clicks = 0
            total_impressions = 0
            
            for query, samples in queries.items():
                if len(samples) < 2:
                    continue
                
                # Calculate predicted scores using new weights
                for sample in samples:
                    features = sample['features']
                    predicted_score = (
                        features['readability_score'] * config.readability_weight +
                        features['ner_density_score'] * config.ner_density_weight +
                        features['sentiment_score'] * config.sentiment_weight +
                        features['tfidf_relevance_score'] * config.tfidf_relevance_weight +
                        features['recency_score'] * config.recency_weight +
                        features['reputation_score'] * config.reputation_weight +
                        features['topic_coherence_score'] * config.topic_coherence_weight
                    )
                    sample['predicted_score'] = predicted_score
                
                # Sort by predicted score (descending)
                samples.sort(key=lambda x: x['predicted_score'], reverse=True)
                
                # Calculate NDCG and MAP for this query
                relevance_scores = [s['relevance'] for s in samples]
                ndcg = self._calculate_ndcg(relevance_scores)
                map_score = self._calculate_map(relevance_scores)
                
                ndcg_scores.append(ndcg)
                map_scores.append(map_score)
                
                # Count clicks and impressions
                for sample in samples:
                    total_impressions += 1
                    if sample['relevance'] > 0:
                        total_clicks += 1
            
            # Calculate average metrics
            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
            avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
            ctr = total_clicks / total_impressions if total_impressions > 0 else 0.0
            
            return LearningMetrics(
                ndcg_score=avg_ndcg,
                map_score=avg_map,
                click_through_rate=ctr,
                total_samples=len(training_data),
                version=""  # Will be set by caller
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {str(e)}")
            return LearningMetrics(0.0, 0.0, 0.0, 0, "")
    
    def _calculate_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        try:
            import numpy as np
            
            def dcg(scores):
                return sum(
                    (2**score - 1) / np.log2(i + 2)
                    for i, score in enumerate(scores[:k])
                )
            
            actual_dcg = dcg(relevance_scores)
            ideal_dcg = dcg(sorted(relevance_scores, reverse=True))
            
            return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_map(self, relevance_scores: List[float]) -> float:
        """Calculate Mean Average Precision."""
        try:
            relevant_items = 0
            precision_sum = 0.0
            
            for i, score in enumerate(relevance_scores):
                if score > 0:  # Relevant item
                    relevant_items += 1
                    precision_at_i = relevant_items / (i + 1)
                    precision_sum += precision_at_i
            
            return precision_sum / relevant_items if relevant_items > 0 else 0.0
            
        except Exception:
            return 0.0


def generate_session_id() -> str:
    """Generate a unique session ID for anonymous tracking."""
    return str(uuid.uuid4())