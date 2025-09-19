"""
Unit tests for database persistence layer.
"""

import pytest
import tempfile
import os
import shutil
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from curator.core.database import Base, DatabaseManager, Article as DBArticle, ScoreCard as DBScoreCard, Entity as DBEntity
from curator.core.repository import ArticleRepository, ScoreCardRepository, EntityRepository, RetentionPolicy
from curator.core.models import Article, ScoreCard, Entity
from curator.core.migrations import DatabaseInitializer


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    # Create a temporary directory and database file
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    # Set environment variable for test database
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
    
    # Create database manager and initialize
    db_manager = DatabaseManager()
    db_manager.create_tables()
    
    yield db_manager
    
    # Cleanup
    db_manager.close()
    
    # Clean up files and directory
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Clean up environment
    if 'DATABASE_URL' in os.environ:
        del os.environ['DATABASE_URL']


@pytest.fixture
def sample_article():
    """Create a sample article for testing."""
    return Article(
        url="https://example.com/test-article",
        title="Test Article",
        author="Test Author",
        publish_date=datetime.now(timezone.utc),
        content="This is test content for the article. " * 50,  # 300+ words
        summary="This is a test summary.",
        entities=[
            Entity(text="Test Company", label="ORG", confidence=0.95),
            Entity(text="John Doe", label="PERSON", confidence=0.88),
        ]
    )


@pytest.fixture
def sample_scorecard(sample_article):
    """Create a sample scorecard for testing."""
    return ScoreCard(
        overall_score=75.5,
        readability_score=80.0,
        ner_density_score=70.0,
        sentiment_score=60.0,
        tfidf_relevance_score=85.0,
        recency_score=90.0,
        reputation_score=65.0,
        topic_coherence_score=75.0,
        article=sample_article
    )


class TestDatabaseModels:
    """Test SQLAlchemy database models."""
    
    def test_article_model_conversion(self, sample_article):
        """Test conversion between dataclass and SQLAlchemy models."""
        # Convert to SQLAlchemy model
        db_article = DBArticle.from_model(sample_article)
        
        assert db_article.url == sample_article.url
        assert db_article.title == sample_article.title
        assert db_article.author == sample_article.author
        assert db_article.content == sample_article.content
        assert db_article.summary == sample_article.summary
        
        # Convert back to dataclass
        converted_article = db_article.to_model()
        
        assert converted_article.url == sample_article.url
        assert converted_article.title == sample_article.title
        assert converted_article.author == sample_article.author
        assert converted_article.content == sample_article.content
        assert converted_article.summary == sample_article.summary
    
    def test_scorecard_model_conversion(self, sample_scorecard):
        """Test conversion between dataclass and SQLAlchemy scorecard models."""
        # Convert to SQLAlchemy model
        db_scorecard = DBScoreCard.from_model(sample_scorecard, article_id=1, query_context="test query")
        
        assert db_scorecard.overall_score == sample_scorecard.overall_score
        assert db_scorecard.readability_score == sample_scorecard.readability_score
        assert db_scorecard.ner_density_score == sample_scorecard.ner_density_score
        assert db_scorecard.query_context == "test query"
    
    def test_entity_model_conversion(self, sample_article):
        """Test conversion between dataclass and SQLAlchemy entity models."""
        entity = sample_article.entities[0]
        
        # Convert to SQLAlchemy model
        db_entity = DBEntity.from_model(entity, article_id=1)
        
        assert db_entity.text == entity.text
        assert db_entity.label == entity.label
        assert db_entity.confidence == entity.confidence
        assert db_entity.article_id == 1
        
        # Convert back to dataclass
        converted_entity = db_entity.to_model()
        
        assert converted_entity.text == entity.text
        assert converted_entity.label == entity.label
        assert converted_entity.confidence == entity.confidence


class TestArticleRepository:
    """Test ArticleRepository operations."""
    
    def test_save_and_retrieve_article(self, temp_db, sample_article):
        """Test saving and retrieving an article."""
        with ArticleRepository() as repo:
            # Save article
            article_id = repo.save_article(sample_article)
            assert article_id > 0
            
            # Retrieve by URL
            retrieved = repo.get_article_by_url(sample_article.url)
            assert retrieved is not None
            assert retrieved.url == sample_article.url
            assert retrieved.title == sample_article.title
            assert len(retrieved.entities) == len(sample_article.entities)
            
            # Retrieve by ID
            retrieved_by_id = repo.get_article_by_id(article_id)
            assert retrieved_by_id is not None
            assert retrieved_by_id.url == sample_article.url
    
    def test_update_existing_article(self, temp_db, sample_article):
        """Test updating an existing article."""
        with ArticleRepository() as repo:
            # Save original article
            article_id = repo.save_article(sample_article)
            
            # Update article
            sample_article.title = "Updated Title"
            sample_article.content = "Updated content"
            
            # Save again (should update)
            updated_id = repo.save_article(sample_article)
            assert updated_id == article_id  # Same ID
            
            # Retrieve and verify update
            retrieved = repo.get_article_by_url(sample_article.url)
            assert retrieved.title == "Updated Title"
            assert retrieved.content == "Updated content"
    
    def test_search_articles(self, temp_db, sample_article):
        """Test article search functionality."""
        with ArticleRepository() as repo:
            # Save article
            repo.save_article(sample_article)
            
            # Search by title
            results = repo.search_articles("Test Article")
            assert len(results) == 1
            assert results[0].title == sample_article.title
            
            # Search by content
            results = repo.search_articles("test content")
            assert len(results) == 1
            
            # Search with no matches
            results = repo.search_articles("nonexistent")
            assert len(results) == 0
    
    def test_get_articles_by_date_range(self, temp_db, sample_article):
        """Test retrieving articles by date range."""
        with ArticleRepository() as repo:
            # Save article
            repo.save_article(sample_article)
            
            # Search within date range
            start_date = datetime.now(timezone.utc) - timedelta(hours=1)
            end_date = datetime.now(timezone.utc) + timedelta(hours=1)
            
            results = repo.get_articles_by_date_range(start_date, end_date)
            assert len(results) == 1
            
            # Search outside date range
            start_date = datetime.now(timezone.utc) - timedelta(days=2)
            end_date = datetime.now(timezone.utc) - timedelta(days=1)
            
            results = repo.get_articles_by_date_range(start_date, end_date)
            assert len(results) == 0
    
    def test_delete_old_articles(self, temp_db, sample_article):
        """Test deleting old articles."""
        with ArticleRepository() as repo:
            # Save article
            repo.save_article(sample_article)
            
            # Delete articles older than 0 days (should delete the article)
            deleted_count = repo.delete_old_articles(older_than_days=0)
            assert deleted_count == 1
            
            # Verify article is deleted
            retrieved = repo.get_article_by_url(sample_article.url)
            assert retrieved is None


class TestScoreCardRepository:
    """Test ScoreCardRepository operations."""
    
    def test_save_and_retrieve_scorecard(self, temp_db, sample_scorecard):
        """Test saving and retrieving a scorecard."""
        with ScoreCardRepository() as repo:
            # Save scorecard
            scorecard_id = repo.save_scorecard(sample_scorecard, sample_scorecard.article.url, "test query")
            assert scorecard_id > 0
            
            # Retrieve by ID
            retrieved = repo.get_scorecard_by_id(scorecard_id)
            assert retrieved is not None
            assert retrieved.overall_score == sample_scorecard.overall_score
            assert retrieved.article.url == sample_scorecard.article.url
            
            # Retrieve by article URL
            scorecards = repo.get_scorecards_by_article_url(sample_scorecard.article.url)
            assert len(scorecards) == 1
            assert scorecards[0].overall_score == sample_scorecard.overall_score
    
    def test_get_top_scorecards(self, temp_db, sample_scorecard):
        """Test retrieving top scoring articles."""
        with ScoreCardRepository() as repo:
            # Save scorecard
            repo.save_scorecard(sample_scorecard, sample_scorecard.article.url)
            
            # Get top scorecards
            top_scorecards = repo.get_top_scorecards(limit=10, min_score=70.0)
            assert len(top_scorecards) == 1
            assert top_scorecards[0].overall_score == sample_scorecard.overall_score
            
            # Get top scorecards with higher minimum score
            top_scorecards = repo.get_top_scorecards(limit=10, min_score=80.0)
            assert len(top_scorecards) == 0
    
    def test_get_scorecards_by_query(self, temp_db, sample_scorecard):
        """Test retrieving scorecards by query context."""
        with ScoreCardRepository() as repo:
            # Save scorecard with query context
            repo.save_scorecard(sample_scorecard, sample_scorecard.article.url, "artificial intelligence")
            
            # Search by query context
            results = repo.get_scorecards_by_query("artificial")
            assert len(results) == 1
            
            # Search with no matches
            results = repo.get_scorecards_by_query("nonexistent")
            assert len(results) == 0


class TestEntityRepository:
    """Test EntityRepository operations."""
    
    def test_get_entities_by_article_url(self, temp_db, sample_article):
        """Test retrieving entities by article URL."""
        with ArticleRepository() as article_repo:
            article_repo.save_article(sample_article)
        
        with EntityRepository() as entity_repo:
            entities = entity_repo.get_entities_by_article_url(sample_article.url)
            assert len(entities) == len(sample_article.entities)
            
            entity_texts = [e.text for e in entities]
            assert "Test Company" in entity_texts
            assert "John Doe" in entity_texts
    
    def test_get_entities_by_label(self, temp_db, sample_article):
        """Test retrieving entities by label."""
        with ArticleRepository() as article_repo:
            article_repo.save_article(sample_article)
        
        with EntityRepository() as entity_repo:
            # Get ORG entities
            org_entities = entity_repo.get_entities_by_label("ORG")
            assert len(org_entities) == 1
            assert org_entities[0][0].text == "Test Company"
            assert org_entities[0][1] == sample_article.url  # Article URL
            
            # Get PERSON entities
            person_entities = entity_repo.get_entities_by_label("PERSON")
            assert len(person_entities) == 1
            assert person_entities[0][0].text == "John Doe"
    
    def test_get_most_common_entities(self, temp_db, sample_article):
        """Test retrieving most common entities."""
        with ArticleRepository() as article_repo:
            article_repo.save_article(sample_article)
        
        with EntityRepository() as entity_repo:
            common_entities = entity_repo.get_most_common_entities()
            assert len(common_entities) == 2
            
            # Check that entities are returned with counts
            entity_data = {(text, label): count for text, label, count in common_entities}
            assert entity_data[("Test Company", "ORG")] == 1
            assert entity_data[("John Doe", "PERSON")] == 1


class TestRetentionPolicy:
    """Test RetentionPolicy operations."""
    
    def test_cleanup_old_data(self, temp_db, sample_scorecard):
        """Test data cleanup based on retention policy."""
        with ScoreCardRepository() as repo:
            repo.save_scorecard(sample_scorecard, sample_scorecard.article.url)
        
        with RetentionPolicy() as retention:
            # Get initial stats
            initial_stats = retention.get_storage_stats()
            assert initial_stats['total_articles'] == 1
            assert initial_stats['total_scorecards'] == 1
            
            # Cleanup with 0 days retention (should delete everything)
            cleanup_stats = retention.cleanup_old_data(
                article_retention_days=0,
                scorecard_retention_days=0
            )
            
            assert cleanup_stats['articles_deleted'] == 1
            assert cleanup_stats['scorecards_deleted'] == 1
            
            # Verify cleanup
            final_stats = retention.get_storage_stats()
            assert final_stats['total_articles'] == 0
            assert final_stats['total_scorecards'] == 0
    
    def test_get_storage_stats(self, temp_db, sample_scorecard):
        """Test storage statistics retrieval."""
        with ScoreCardRepository() as repo:
            repo.save_scorecard(sample_scorecard, sample_scorecard.article.url)
        
        with RetentionPolicy() as retention:
            stats = retention.get_storage_stats()
            
            assert 'total_articles' in stats
            assert 'total_scorecards' in stats
            assert 'total_entities' in stats
            assert 'articles_last_7_days' in stats
            assert 'scorecards_last_7_days' in stats
            
            assert stats['total_articles'] == 1
            assert stats['total_scorecards'] == 1
            assert stats['total_entities'] == 2  # Two entities in sample article


class TestDatabaseInitializer:
    """Test DatabaseInitializer operations."""
    
    def test_initialize_database(self, temp_db):
        """Test database initialization."""
        initializer = DatabaseInitializer()
        
        # Initialize database
        initializer.initialize_database()
        
        # Verify health
        health = initializer.verify_database_health()
        assert health['status'] == 'healthy'
        assert health['connection_ok'] is True
        assert health['tables_exist'] is True
    
    def test_verify_database_health(self, temp_db):
        """Test database health verification."""
        initializer = DatabaseInitializer()
        
        health = initializer.verify_database_health()
        
        assert 'status' in health
        assert 'connection_ok' in health
        assert 'tables_exist' in health
        assert 'migration_status' in health
        
        # Should be healthy after temp_db fixture setup
        assert health['connection_ok'] is True
        assert health['tables_exist'] is True