#!/usr/bin/env python3
"""
Example demonstrating the persistence layer functionality.
Shows how to store and retrieve articles, scorecards, and entities.
"""

import os
import sys
from datetime import datetime, timezone, timedelta

# Add the parent directory to the path so we can import curator modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curator.core.models import Article, ScoreCard, Entity
from curator.core.database import init_database
from curator.core.repository import ArticleRepository, ScoreCardRepository, EntityRepository, RetentionPolicy
from curator.core.migrations import DatabaseInitializer


def create_sample_data():
    """Create sample articles and scorecards for demonstration."""
    
    # Sample articles
    articles = [
        Article(
            url="https://example.com/ai-breakthrough",
            title="Major AI Breakthrough in Natural Language Processing",
            author="Dr. Jane Smith",
            publish_date=datetime.now(timezone.utc) - timedelta(hours=2),
            content="Researchers at Tech University have announced a significant breakthrough in natural language processing. The new model demonstrates unprecedented accuracy in understanding context and generating human-like responses. This development could revolutionize how we interact with AI systems in the future.",
            summary="Researchers announce major breakthrough in NLP with new model showing unprecedented accuracy.",
            entities=[
                Entity(text="Tech University", label="ORG", confidence=0.95),
                Entity(text="Dr. Jane Smith", label="PERSON", confidence=0.92),
                Entity(text="natural language processing", label="TECHNOLOGY", confidence=0.88),
            ]
        ),
        Article(
            url="https://example.com/climate-report",
            title="New Climate Report Shows Accelerating Changes",
            author="Environmental News Team",
            publish_date=datetime.now(timezone.utc) - timedelta(hours=6),
            content="The latest climate report from the International Panel reveals that climate changes are accelerating faster than previously predicted. Rising sea levels, increasing temperatures, and extreme weather events are becoming more frequent. Immediate action is needed to address these challenges.",
            summary="Latest climate report reveals accelerating changes requiring immediate action.",
            entities=[
                Entity(text="International Panel", label="ORG", confidence=0.90),
                Entity(text="Environmental News Team", label="PERSON", confidence=0.85),
            ]
        ),
        Article(
            url="https://example.com/tech-startup",
            title="Startup Raises $50M for Revolutionary Battery Technology",
            author="Business Reporter",
            publish_date=datetime.now(timezone.utc) - timedelta(days=1),
            content="EnergyTech Solutions has successfully raised $50 million in Series B funding to develop their revolutionary solid-state battery technology. The new batteries promise 10x longer life and faster charging times compared to current lithium-ion batteries.",
            summary="EnergyTech Solutions raises $50M for solid-state battery technology development.",
            entities=[
                Entity(text="EnergyTech Solutions", label="ORG", confidence=0.98),
                Entity(text="Business Reporter", label="PERSON", confidence=0.80),
                Entity(text="$50 million", label="MONEY", confidence=0.95),
            ]
        )
    ]
    
    # Create corresponding scorecards
    scorecards = []
    for article in articles:
        scorecard = ScoreCard(
            overall_score=75.0 + (hash(article.url) % 20),  # Random score 75-95
            readability_score=80.0,
            ner_density_score=70.0,
            sentiment_score=65.0,
            tfidf_relevance_score=85.0,
            recency_score=90.0,
            reputation_score=75.0,
            topic_coherence_score=80.0,
            article=article
        )
        scorecards.append(scorecard)
    
    return articles, scorecards


def demonstrate_article_operations():
    """Demonstrate article repository operations."""
    print("\n=== Article Repository Operations ===")
    
    articles, _ = create_sample_data()
    
    with ArticleRepository() as repo:
        print("Saving articles...")
        article_ids = []
        for article in articles:
            article_id = repo.save_article(article)
            article_ids.append(article_id)
            print(f"  Saved article: {article.title} (ID: {article_id})")
        
        print(f"\nSaved {len(article_ids)} articles")
        
        # Retrieve articles
        print("\nRetrieving articles...")
        for article_id in article_ids:
            retrieved = repo.get_article_by_id(article_id)
            if retrieved:
                print(f"  Retrieved: {retrieved.title}")
                print(f"    Entities: {len(retrieved.entities)}")
        
        # Search articles
        print("\nSearching articles...")
        search_results = repo.search_articles("AI", limit=5)
        print(f"  Found {len(search_results)} articles matching 'AI'")
        for article in search_results:
            print(f"    - {article.title}")
        
        # Get recent articles
        print("\nRecent articles...")
        recent = repo.get_recent_articles(limit=5)
        print(f"  Found {len(recent)} recent articles")
        for article in recent:
            print(f"    - {article.title} ({article.publish_date})")


def demonstrate_scorecard_operations():
    """Demonstrate scorecard repository operations."""
    print("\n=== ScoreCard Repository Operations ===")
    
    articles, scorecards = create_sample_data()
    
    with ScoreCardRepository() as repo:
        print("Saving scorecards...")
        scorecard_ids = []
        for i, scorecard in enumerate(scorecards):
            query_context = ["AI technology", "climate change", "startup funding"][i]
            scorecard_id = repo.save_scorecard(scorecard, scorecard.article.url, query_context)
            scorecard_ids.append(scorecard_id)
            print(f"  Saved scorecard: {scorecard.overall_score:.1f} for {scorecard.article.title}")
        
        print(f"\nSaved {len(scorecard_ids)} scorecards")
        
        # Get top scorecards
        print("\nTop scoring articles...")
        top_scorecards = repo.get_top_scorecards(limit=5, min_score=70.0)
        print(f"  Found {len(top_scorecards)} top scorecards")
        for scorecard in top_scorecards:
            print(f"    - Score: {scorecard.overall_score:.1f} - {scorecard.article.title}")
        
        # Search by query context
        print("\nScorécards by query context...")
        ai_scorecards = repo.get_scorecards_by_query("AI")
        print(f"  Found {len(ai_scorecards)} scorecards for 'AI' queries")
        for scorecard in ai_scorecards:
            print(f"    - {scorecard.article.title}")


def demonstrate_entity_operations():
    """Demonstrate entity repository operations."""
    print("\n=== Entity Repository Operations ===")
    
    articles, _ = create_sample_data()
    
    # First save articles to get entities in database
    with ArticleRepository() as article_repo:
        for article in articles:
            article_repo.save_article(article)
    
    with EntityRepository() as repo:
        # Get entities by label
        print("Entities by label...")
        org_entities = repo.get_entities_by_label("ORG", limit=10)
        print(f"  Found {len(org_entities)} ORG entities:")
        for entity, article_url in org_entities:
            print(f"    - {entity.text} (confidence: {entity.confidence:.2f})")
        
        person_entities = repo.get_entities_by_label("PERSON", limit=10)
        print(f"  Found {len(person_entities)} PERSON entities:")
        for entity, article_url in person_entities:
            print(f"    - {entity.text} (confidence: {entity.confidence:.2f})")
        
        # Get most common entities
        print("\nMost common entities...")
        common_entities = repo.get_most_common_entities(limit=10)
        print(f"  Found {len(common_entities)} unique entities:")
        for text, label, count in common_entities:
            print(f"    - {text} ({label}): {count} occurrences")


def demonstrate_retention_policy():
    """Demonstrate retention policy operations."""
    print("\n=== Retention Policy Operations ===")
    
    with RetentionPolicy() as retention:
        # Get storage statistics
        print("Storage statistics...")
        stats = retention.get_storage_stats()
        print(f"  Total articles: {stats['total_articles']}")
        print(f"  Total scorecards: {stats['total_scorecards']}")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Articles in last 7 days: {stats['articles_last_7_days']}")
        print(f"  Scorecards in last 7 days: {stats['scorecards_last_7_days']}")
        
        if stats.get('oldest_article_date'):
            print(f"  Oldest article: {stats['oldest_article_date']}")
        if stats.get('newest_article_date'):
            print(f"  Newest article: {stats['newest_article_date']}")
        
        # Note: We won't actually run cleanup in this demo to preserve data
        print("\nRetention cleanup (simulation)...")
        print("  Would clean up articles older than 90 days")
        print("  Would clean up scorecards older than 30 days")
        print("  (Skipping actual cleanup to preserve demo data)")


def demonstrate_database_health():
    """Demonstrate database health checking."""
    print("\n=== Database Health Check ===")
    
    initializer = DatabaseInitializer()
    health = initializer.verify_database_health()
    
    print(f"Database status: {health['status']}")
    print(f"Connection OK: {health['connection_ok']}")
    print(f"Tables exist: {health['tables_exist']}")
    print(f"Migration status: {health['migration_status']}")
    
    if health.get('existing_tables'):
        print(f"Existing tables: {', '.join(health['existing_tables'])}")
    
    if health.get('applied_migrations'):
        print(f"Applied migrations: {', '.join(health['applied_migrations'])}")
    
    if health.get('storage_stats'):
        stats = health['storage_stats']
        print(f"Storage: {stats['total_articles']} articles, {stats['total_scorecards']} scorecards")


def main():
    """Main demonstration function."""
    print("AI Content Curator - Persistence Layer Demo")
    print("=" * 50)
    
    # Initialize database
    print("Initializing database...")
    try:
        init_database()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return 1
    
    try:
        # Run demonstrations
        demonstrate_database_health()
        demonstrate_article_operations()
        demonstrate_scorecard_operations()
        demonstrate_entity_operations()
        demonstrate_retention_policy()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nYou can now:")
        print("- Use the manage_db.py script for database management")
        print("- Integrate persistence into your analysis workflows")
        print("- Set up retention policies for data cleanup")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())