"""
Repository layer for data access operations.
Provides high-level interface for storing and retrieving articles, scorecards, and entities.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func

from curator.core.database import (
    Article as DBArticle, 
    ScoreCard as DBScoreCard, 
    Entity as DBEntity,
    get_db_session
)
from curator.core.models import Article, ScoreCard, Entity


class ArticleRepository:
    """Repository for Article operations."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self.session = session or get_db_session()
        self._should_close_session = session is None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_session:
            self.session.close()
    
    def save_article(self, article: Article) -> int:
        """Save an article and return its ID."""
        try:
            # Check if article already exists
            existing = self.session.query(DBArticle).filter_by(url=article.url).first()
            
            if existing:
                # Update existing article
                existing.title = article.title
                existing.author = article.author
                existing.publish_date = article.publish_date
                existing.content = article.content
                existing.summary = article.summary
                existing.updated_at = datetime.now(timezone.utc)
                
                # Clear existing entities and add new ones
                self.session.query(DBEntity).filter_by(article_id=existing.id).delete()
                for entity in article.entities:
                    db_entity = DBEntity.from_model(entity, existing.id)
                    self.session.add(db_entity)
                
                self.session.commit()
                return existing.id
            else:
                # Create new article
                db_article = DBArticle.from_model(article)
                self.session.add(db_article)
                self.session.flush()  # Get the ID
                
                # Add entities
                for entity in article.entities:
                    db_entity = DBEntity.from_model(entity, db_article.id)
                    self.session.add(db_entity)
                
                self.session.commit()
                return db_article.id
                
        except Exception as e:
            self.session.rollback()
            raise e
    
    def get_article_by_url(self, url: str) -> Optional[Article]:
        """Get article by URL."""
        db_article = self.session.query(DBArticle).filter_by(url=url).first()
        return db_article.to_model() if db_article else None
    
    def get_article_by_id(self, article_id: int) -> Optional[Article]:
        """Get article by ID."""
        db_article = self.session.query(DBArticle).filter_by(id=article_id).first()
        return db_article.to_model() if db_article else None
    
    def get_articles_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Article]:
        """Get articles within a date range."""
        db_articles = (
            self.session.query(DBArticle)
            .filter(and_(
                DBArticle.publish_date >= start_date,
                DBArticle.publish_date <= end_date
            ))
            .order_by(desc(DBArticle.publish_date))
            .all()
        )
        return [article.to_model() for article in db_articles]
    
    def get_recent_articles(self, limit: int = 100) -> List[Article]:
        """Get most recently created articles."""
        db_articles = (
            self.session.query(DBArticle)
            .order_by(desc(DBArticle.created_at))
            .limit(limit)
            .all()
        )
        return [article.to_model() for article in db_articles]
    
    def search_articles(self, query: str, limit: int = 50) -> List[Article]:
        """Search articles by title or content."""
        search_term = f"%{query}%"
        db_articles = (
            self.session.query(DBArticle)
            .filter(or_(
                DBArticle.title.ilike(search_term),
                DBArticle.content.ilike(search_term)
            ))
            .order_by(desc(DBArticle.created_at))
            .limit(limit)
            .all()
        )
        return [article.to_model() for article in db_articles]
    
    def delete_old_articles(self, older_than_days: int) -> int:
        """Delete articles older than specified days. Returns count of deleted articles."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        
        # Get count before deletion
        count = (
            self.session.query(DBArticle)
            .filter(DBArticle.created_at < cutoff_date)
            .count()
        )
        
        # Delete articles (cascades to scorecards and entities)
        self.session.query(DBArticle).filter(DBArticle.created_at < cutoff_date).delete()
        self.session.commit()
        
        return count


class ScoreCardRepository:
    """Repository for ScoreCard operations."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self.session = session or get_db_session()
        self._should_close_session = session is None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_session:
            self.session.close()
    
    def save_scorecard(self, scorecard: ScoreCard, article_url: str, query_context: Optional[str] = None) -> int:
        """Save a scorecard for an article and return its ID."""
        try:
            # Get or create article first
            with ArticleRepository(self.session) as article_repo:
                article_id = article_repo.save_article(scorecard.article)
            
            # Create scorecard
            db_scorecard = DBScoreCard.from_model(scorecard, article_id, query_context)
            self.session.add(db_scorecard)
            self.session.commit()
            
            return db_scorecard.id
            
        except Exception as e:
            self.session.rollback()
            raise e
    
    def get_scorecard_by_id(self, scorecard_id: int) -> Optional[ScoreCard]:
        """Get scorecard by ID."""
        db_scorecard = self.session.query(DBScoreCard).filter_by(id=scorecard_id).first()
        return db_scorecard.to_model() if db_scorecard else None
    
    def get_scorecards_by_article_url(self, article_url: str) -> List[ScoreCard]:
        """Get all scorecards for an article."""
        db_scorecards = (
            self.session.query(DBScoreCard)
            .join(DBArticle)
            .filter(DBArticle.url == article_url)
            .order_by(desc(DBScoreCard.created_at))
            .all()
        )
        return [scorecard.to_model() for scorecard in db_scorecards]
    
    def get_latest_scorecard_by_article_url(self, article_url: str) -> Optional[ScoreCard]:
        """Get the most recent scorecard for an article."""
        db_scorecard = (
            self.session.query(DBScoreCard)
            .join(DBArticle)
            .filter(DBArticle.url == article_url)
            .order_by(desc(DBScoreCard.created_at))
            .first()
        )
        return db_scorecard.to_model() if db_scorecard else None
    
    def get_top_scorecards(self, limit: int = 50, min_score: float = 0.0) -> List[ScoreCard]:
        """Get top-scoring articles."""
        db_scorecards = (
            self.session.query(DBScoreCard)
            .filter(DBScoreCard.overall_score >= min_score)
            .order_by(desc(DBScoreCard.overall_score))
            .limit(limit)
            .all()
        )
        return [scorecard.to_model() for scorecard in db_scorecards]
    
    def get_scorecards_by_query(self, query_context: str, limit: int = 50) -> List[ScoreCard]:
        """Get scorecards that were created with a specific query context."""
        search_term = f"%{query_context}%"
        db_scorecards = (
            self.session.query(DBScoreCard)
            .filter(DBScoreCard.query_context.ilike(search_term))
            .order_by(desc(DBScoreCard.overall_score))
            .limit(limit)
            .all()
        )
        return [scorecard.to_model() for scorecard in db_scorecards]
    
    def delete_old_scorecards(self, older_than_days: int) -> int:
        """Delete scorecards older than specified days. Returns count of deleted scorecards."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        
        # Get count before deletion
        count = (
            self.session.query(DBScoreCard)
            .filter(DBScoreCard.created_at < cutoff_date)
            .count()
        )
        
        # Delete scorecards
        self.session.query(DBScoreCard).filter(DBScoreCard.created_at < cutoff_date).delete()
        self.session.commit()
        
        return count


class EntityRepository:
    """Repository for Entity operations."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self.session = session or get_db_session()
        self._should_close_session = session is None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_session:
            self.session.close()
    
    def get_entities_by_article_url(self, article_url: str) -> List[Entity]:
        """Get all entities for an article."""
        db_entities = (
            self.session.query(DBEntity)
            .join(DBArticle)
            .filter(DBArticle.url == article_url)
            .all()
        )
        return [entity.to_model() for entity in db_entities]
    
    def get_entities_by_label(self, label: str, limit: int = 100) -> List[Tuple[Entity, str]]:
        """Get entities by label with their article URLs."""
        results = (
            self.session.query(DBEntity, DBArticle.url)
            .join(DBArticle)
            .filter(DBEntity.label == label)
            .order_by(desc(DBEntity.confidence))
            .limit(limit)
            .all()
        )
        return [(entity.to_model(), url) for entity, url in results]
    
    def get_most_common_entities(self, limit: int = 50) -> List[Tuple[str, str, int]]:
        """Get most common entities (text, label, count)."""
        results = (
            self.session.query(
                DBEntity.text,
                DBEntity.label,
                func.count(DBEntity.id).label('count')
            )
            .group_by(DBEntity.text, DBEntity.label)
            .order_by(desc('count'))
            .limit(limit)
            .all()
        )
        return [(text, label, count) for text, label, count in results]


class RetentionPolicy:
    """Handles data retention and cleanup operations."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize retention policy with optional session."""
        self.session = session or get_db_session()
        self._should_close_session = session is None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close_session:
            self.session.close()
    
    def cleanup_old_data(self, 
                        article_retention_days: int = 90,
                        scorecard_retention_days: int = 30) -> dict:
        """
        Clean up old data based on retention policies.
        
        Args:
            article_retention_days: Keep articles newer than this many days
            scorecard_retention_days: Keep scorecards newer than this many days
            
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'articles_deleted': 0,
            'scorecards_deleted': 0,
            'cleanup_date': datetime.now(timezone.utc)
        }
        
        try:
            # Clean up old scorecards first (they reference articles)
            with ScoreCardRepository(self.session) as scorecard_repo:
                stats['scorecards_deleted'] = scorecard_repo.delete_old_scorecards(scorecard_retention_days)
            
            # Clean up old articles (this will cascade to remaining scorecards and entities)
            with ArticleRepository(self.session) as article_repo:
                stats['articles_deleted'] = article_repo.delete_old_articles(article_retention_days)
            
            return stats
            
        except Exception as e:
            self.session.rollback()
            raise e
    
    def get_storage_stats(self) -> dict:
        """Get database storage statistics."""
        stats = {}
        
        # Count articles
        stats['total_articles'] = self.session.query(DBArticle).count()
        stats['articles_last_7_days'] = (
            self.session.query(DBArticle)
            .filter(DBArticle.created_at >= datetime.now(timezone.utc) - timedelta(days=7))
            .count()
        )
        
        # Count scorecards
        stats['total_scorecards'] = self.session.query(DBScoreCard).count()
        stats['scorecards_last_7_days'] = (
            self.session.query(DBScoreCard)
            .filter(DBScoreCard.created_at >= datetime.now(timezone.utc) - timedelta(days=7))
            .count()
        )
        
        # Count entities
        stats['total_entities'] = self.session.query(DBEntity).count()
        
        # Get oldest and newest articles
        oldest_article = (
            self.session.query(DBArticle.created_at)
            .order_by(DBArticle.created_at)
            .first()
        )
        newest_article = (
            self.session.query(DBArticle.created_at)
            .order_by(desc(DBArticle.created_at))
            .first()
        )
        
        stats['oldest_article_date'] = oldest_article[0] if oldest_article else None
        stats['newest_article_date'] = newest_article[0] if newest_article else None
        
        return stats