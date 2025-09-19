"""
Database models and configuration for AI Content Curator.
Provides SQLAlchemy models for Article, ScoreCard, and Entity persistence.
"""

from datetime import datetime, timezone
from typing import List, Optional
import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, 
    ForeignKey, Boolean, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from curator.core.models import Article as ArticleModel, ScoreCard as ScoreCardModel, Entity as EntityModel

Base = declarative_base()


class Article(Base):
    """SQLAlchemy model for Article persistence."""
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True)
    url = Column(String(2048), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=True)
    author = Column(String(255), nullable=True)
    publish_date = Column(DateTime(timezone=True), nullable=True)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    scorecards = relationship("ScoreCard", back_populates="article", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="article", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_articles_publish_date', 'publish_date'),
        Index('idx_articles_created_at', 'created_at'),
    )
    
    def to_model(self) -> ArticleModel:
        """Convert SQLAlchemy model to dataclass model."""
        entities = [entity.to_model() for entity in self.entities]
        return ArticleModel(
            url=self.url,
            title=self.title or "",
            author=self.author or "",
            publish_date=self.publish_date,
            content=self.content or "",
            summary=self.summary or "",
            entities=entities
        )
    
    @classmethod
    def from_model(cls, article: ArticleModel) -> 'Article':
        """Create SQLAlchemy model from dataclass model."""
        return cls(
            url=article.url,
            title=article.title,
            author=article.author,
            publish_date=article.publish_date,
            content=article.content,
            summary=article.summary
        )


class ScoreCard(Base):
    """SQLAlchemy model for ScoreCard persistence."""
    __tablename__ = 'scorecards'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('articles.id'), nullable=False)
    overall_score = Column(Float, nullable=False)
    readability_score = Column(Float, nullable=False)
    ner_density_score = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    tfidf_relevance_score = Column(Float, nullable=False)
    recency_score = Column(Float, nullable=False)
    reputation_score = Column(Float, nullable=False, default=0.0)
    topic_coherence_score = Column(Float, nullable=False, default=0.0)
    query_context = Column(String(500), nullable=True)  # Store the query used for relevance scoring
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    article = relationship("Article", back_populates="scorecards")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_scorecards_overall_score', 'overall_score'),
        Index('idx_scorecards_created_at', 'created_at'),
        Index('idx_scorecards_article_id', 'article_id'),
    )
    
    def to_model(self) -> ScoreCardModel:
        """Convert SQLAlchemy model to dataclass model."""
        return ScoreCardModel(
            overall_score=self.overall_score,
            readability_score=self.readability_score,
            ner_density_score=self.ner_density_score,
            sentiment_score=self.sentiment_score,
            tfidf_relevance_score=self.tfidf_relevance_score,
            recency_score=self.recency_score,
            reputation_score=self.reputation_score,
            topic_coherence_score=self.topic_coherence_score,
            article=self.article.to_model()
        )
    
    @classmethod
    def from_model(cls, scorecard: ScoreCardModel, article_id: int, query_context: Optional[str] = None) -> 'ScoreCard':
        """Create SQLAlchemy model from dataclass model."""
        return cls(
            article_id=article_id,
            overall_score=scorecard.overall_score,
            readability_score=scorecard.readability_score,
            ner_density_score=scorecard.ner_density_score,
            sentiment_score=scorecard.sentiment_score,
            tfidf_relevance_score=scorecard.tfidf_relevance_score,
            recency_score=scorecard.recency_score,
            reputation_score=scorecard.reputation_score,
            topic_coherence_score=scorecard.topic_coherence_score,
            query_context=query_context
        )


class Entity(Base):
    """SQLAlchemy model for Entity persistence."""
    __tablename__ = 'entities'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('articles.id'), nullable=False)
    text = Column(String(255), nullable=False)
    label = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    article = relationship("Article", back_populates="entities")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_entities_article_id', 'article_id'),
        Index('idx_entities_label', 'label'),
        Index('idx_entities_text', 'text'),
    )
    
    def to_model(self) -> EntityModel:
        """Convert SQLAlchemy model to dataclass model."""
        return EntityModel(
            text=self.text,
            label=self.label,
            confidence=self.confidence
        )
    
    @classmethod
    def from_model(cls, entity: EntityModel, article_id: int) -> 'Entity':
        """Create SQLAlchemy model from dataclass model."""
        return cls(
            article_id=article_id,
            text=entity.text,
            label=entity.label,
            confidence=entity.confidence
        )


class UserFeedback(Base):
    """SQLAlchemy model for user feedback on articles."""
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('articles.id'), nullable=False)
    scorecard_id = Column(Integer, ForeignKey('scorecards.id'), nullable=True)
    session_id = Column(String(255), nullable=True)  # Anonymous session tracking
    user_id = Column(String(255), nullable=True)  # Optional user identification
    
    # Feedback types
    clicked = Column(Boolean, default=False, nullable=False)
    saved = Column(Boolean, default=False, nullable=False)
    liked = Column(Boolean, default=False, nullable=False)
    shared = Column(Boolean, default=False, nullable=False)
    
    # Context information
    query_context = Column(String(500), nullable=True)  # Search query or topic
    position_in_results = Column(Integer, nullable=True)  # Position in ranked list
    total_results = Column(Integer, nullable=True)  # Total results shown
    
    # Timing information
    time_on_page = Column(Float, nullable=True)  # Seconds spent on article
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    article = relationship("Article")
    scorecard = relationship("ScoreCard")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_feedback_article_id', 'article_id'),
        Index('idx_feedback_scorecard_id', 'scorecard_id'),
        Index('idx_feedback_session_id', 'session_id'),
        Index('idx_feedback_created_at', 'created_at'),
        Index('idx_feedback_query_context', 'query_context'),
    )


class ScoringWeights(Base):
    """SQLAlchemy model for storing learned scoring weights."""
    __tablename__ = 'scoring_weights'
    
    id = Column(Integer, primary_key=True)
    version = Column(String(50), nullable=False, unique=True)
    
    # Scoring weights
    readability_weight = Column(Float, nullable=False, default=0.12)
    ner_density_weight = Column(Float, nullable=False, default=0.12)
    sentiment_weight = Column(Float, nullable=False, default=0.12)
    tfidf_relevance_weight = Column(Float, nullable=False, default=0.20)
    recency_weight = Column(Float, nullable=False, default=0.12)
    reputation_weight = Column(Float, nullable=False, default=0.12)
    topic_coherence_weight = Column(Float, nullable=False, default=0.20)
    
    # Metadata
    training_samples = Column(Integer, nullable=False, default=0)
    performance_score = Column(Float, nullable=True)  # Validation metric
    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    notes = Column(Text, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_weights_version', 'version'),
        Index('idx_weights_active', 'is_active'),
        Index('idx_weights_created_at', 'created_at'),
    )


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager with connection URL."""
        self.database_url = database_url or self._get_database_url()
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # Check for full database URL first
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            return database_url
        
        # Build URL from components
        db_type = os.environ.get('DB_TYPE', 'sqlite')
        
        if db_type.lower() == 'sqlite':
            db_path = os.environ.get('DB_PATH', 'curator.db')
            return f'sqlite:///{db_path}'
        
        elif db_type.lower() == 'postgresql':
            host = os.environ.get('DB_HOST', 'localhost')
            port = os.environ.get('DB_PORT', '5432')
            user = os.environ.get('DB_USER', 'curator')
            password = os.environ.get('DB_PASSWORD', '')
            database = os.environ.get('DB_NAME', 'curator')
            return f'postgresql://{user}:{password}@{host}:{port}/{database}'
        
        else:
            # Default to SQLite
            return 'sqlite:///curator.db'
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def close(self):
        """Close database connections."""
        self.engine.dispose()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db_session() -> Session:
    """Get a new database session."""
    return get_db_manager().get_session()


def init_database():
    """Initialize the database by creating all tables."""
    db_manager = get_db_manager()
    db_manager.create_tables()


def close_database():
    """Close database connections."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None