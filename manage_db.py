#!/usr/bin/env python3
"""
Database management CLI for AI Content Curator.
Provides commands for initializing, migrating, and maintaining the database.
"""

import sys
import argparse
import logging
from datetime import datetime

from curator.core.migrations import DatabaseInitializer
from curator.core.repository import ArticleRepository, ScoreCardRepository, RetentionPolicy
from curator.core.config import load_database_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def init_command(args):
    """Initialize database command."""
    print("Initializing database...")
    
    try:
        initializer = DatabaseInitializer()
        initializer.initialize_database(force_recreate=args.force)
        
        print("✓ Database initialization completed successfully")
        
        # Show health status
        health = initializer.verify_database_health()
        print(f"✓ Database status: {health['status']}")
        
        if health.get('storage_stats'):
            stats = health['storage_stats']
            print(f"  - Articles: {stats['total_articles']} total")
            print(f"  - Scorecards: {stats['total_scorecards']} total")
            print(f"  - Entities: {stats['total_entities']} total")
        
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return 1
    
    return 0


def status_command(args):
    """Show database status command."""
    print("Checking database status...")
    
    try:
        initializer = DatabaseInitializer()
        health = initializer.verify_database_health()
        
        print(f"Status: {health['status']}")
        print(f"Connection: {'✓' if health['connection_ok'] else '✗'}")
        print(f"Tables exist: {'✓' if health['tables_exist'] else '✗'}")
        print(f"Migration status: {health['migration_status']}")
        
        if health.get('existing_tables'):
            print(f"Existing tables: {', '.join(health['existing_tables'])}")
        
        if health.get('applied_migrations'):
            print(f"Applied migrations: {', '.join(health['applied_migrations'])}")
        
        if health.get('storage_stats'):
            stats = health['storage_stats']
            print("\nStorage Statistics:")
            print(f"  Articles: {stats['total_articles']} total, {stats['articles_last_7_days']} in last 7 days")
            print(f"  Scorecards: {stats['total_scorecards']} total, {stats['scorecards_last_7_days']} in last 7 days")
            print(f"  Entities: {stats['total_entities']} total")
            
            if stats.get('oldest_article_date'):
                print(f"  Oldest article: {stats['oldest_article_date']}")
            if stats.get('newest_article_date'):
                print(f"  Newest article: {stats['newest_article_date']}")
        
        if health.get('error'):
            print(f"Error: {health['error']}")
            return 1
        
    except Exception as e:
        print(f"✗ Status check failed: {e}")
        return 1
    
    return 0


def cleanup_command(args):
    """Run retention cleanup command."""
    print("Running retention cleanup...")
    
    try:
        db_config = load_database_config()
        article_retention = args.article_days or db_config['article_retention_days']
        scorecard_retention = args.scorecard_days or db_config['scorecard_retention_days']
        
        print(f"Article retention: {article_retention} days")
        print(f"Scorecard retention: {scorecard_retention} days")
        
        if not args.force:
            response = input("Continue with cleanup? (y/N): ")
            if response.lower() != 'y':
                print("Cleanup cancelled")
                return 0
        
        with RetentionPolicy() as retention:
            stats = retention.cleanup_old_data(
                article_retention_days=article_retention,
                scorecard_retention_days=scorecard_retention
            )
        
        print("✓ Retention cleanup completed successfully")
        print(f"  Articles deleted: {stats['articles_deleted']}")
        print(f"  Scorecards deleted: {stats['scorecards_deleted']}")
        print(f"  Cleanup date: {stats['cleanup_date']}")
        
    except Exception as e:
        print(f"✗ Retention cleanup failed: {e}")
        return 1
    
    return 0


def search_command(args):
    """Search articles command."""
    print(f"Searching for articles matching: '{args.query}'")
    
    try:
        with ArticleRepository() as repo:
            articles = repo.search_articles(args.query, limit=args.limit)
        
        if not articles:
            print("No articles found")
            return 0
        
        print(f"Found {len(articles)} articles:")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article.title}")
            print(f"   URL: {article.url}")
            print(f"   Author: {article.author}")
            if article.publish_date:
                print(f"   Published: {article.publish_date}")
            print()
        
    except Exception as e:
        print(f"✗ Search failed: {e}")
        return 1
    
    return 0


def top_scores_command(args):
    """Show top scoring articles command."""
    print(f"Top {args.limit} scoring articles (min score: {args.min_score}):")
    
    try:
        with ScoreCardRepository() as repo:
            scorecards = repo.get_top_scorecards(limit=args.limit, min_score=args.min_score)
        
        if not scorecards:
            print("No scorecards found")
            return 0
        
        for i, scorecard in enumerate(scorecards, 1):
            article = scorecard.article
            print(f"{i}. Score: {scorecard.overall_score:.1f} - {article.title}")
            print(f"   URL: {article.url}")
            print(f"   Readability: {scorecard.readability_score:.1f}, "
                  f"NER: {scorecard.ner_density_score:.1f}, "
                  f"Sentiment: {scorecard.sentiment_score:.1f}")
            print(f"   Relevance: {scorecard.tfidf_relevance_score:.1f}, "
                  f"Recency: {scorecard.recency_score:.1f}")
            print()
        
    except Exception as e:
        print(f"✗ Failed to get top scores: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Database management for AI Content Curator')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize database')
    init_parser.add_argument('--force', action='store_true', help='Force recreate tables')
    
    # Status command
    subparsers.add_parser('status', help='Show database status')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Run retention cleanup')
    cleanup_parser.add_argument('--article-days', type=int, help='Article retention days')
    cleanup_parser.add_argument('--scorecard-days', type=int, help='Scorecard retention days')
    cleanup_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search articles')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Max results')
    
    # Top scores command
    top_parser = subparsers.add_parser('top', help='Show top scoring articles')
    top_parser.add_argument('--limit', type=int, default=10, help='Max results')
    top_parser.add_argument('--min-score', type=float, default=0.0, help='Minimum score')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    # Command dispatch
    commands = {
        'init': init_command,
        'status': status_command,
        'cleanup': cleanup_command,
        'search': search_command,
        'top': top_scores_command,
    }
    
    command_func = commands.get(args.command)
    if command_func:
        return command_func(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())