"""
Database migration utilities for AI Content Curator.
Provides simple migration management without Alembic for basic use cases.
"""

import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

from curator.core.database import get_db_manager, Base
from curator.core.repository import RetentionPolicy

logger = logging.getLogger(__name__)


class MigrationManager:
    """Simple migration manager for database schema changes."""
    
    def __init__(self):
        """Initialize migration manager."""
        self.db_manager = get_db_manager()
        self.engine = self.db_manager.engine
    
    def create_migration_table(self):
        """Create migrations tracking table if it doesn't exist."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY,
            version VARCHAR(50) NOT NULL UNIQUE,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version FROM migrations ORDER BY applied_at"))
                return [row[0] for row in result]
        except SQLAlchemyError:
            # Migration table doesn't exist yet
            return []
    
    def apply_migration(self, version: str, description: str, sql_commands: List[str]):
        """Apply a migration with the given SQL commands."""
        try:
            with self.engine.connect() as conn:
                # Execute migration commands
                for command in sql_commands:
                    if command.strip():
                        conn.execute(text(command))
                
                # Record migration
                conn.execute(
                    text("INSERT INTO migrations (version, description) VALUES (:version, :description)"),
                    {"version": version, "description": description}
                )
                conn.commit()
                
                logger.info(f"Applied migration {version}: {description}")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to apply migration {version}: {e}")
            raise
    
    def check_and_create_tables(self):
        """Check if tables exist and create them if they don't."""
        inspector = inspect(self.engine)
        existing_tables = inspector.get_table_names()
        
        required_tables = ['articles', 'scorecards', 'entities']
        missing_tables = [table for table in required_tables if table not in existing_tables]
        
        if missing_tables:
            logger.info(f"Creating missing tables: {missing_tables}")
            Base.metadata.create_all(bind=self.engine)
            return True
        
        return False
    
    def run_migrations(self):
        """Run all pending migrations."""
        # Create migration tracking table
        self.create_migration_table()
        
        # Get applied migrations
        applied = set(self.get_applied_migrations())
        
        # Define available migrations
        migrations = self._get_available_migrations()
        
        # Apply pending migrations
        for version, description, commands in migrations:
            if version not in applied:
                logger.info(f"Applying migration {version}: {description}")
                self.apply_migration(version, description, commands)
    
    def _get_available_migrations(self) -> List[tuple]:
        """Get list of available migrations (version, description, commands)."""
        migrations = []
        
        # Migration 001: Initial schema
        migrations.append((
            "001_initial_schema",
            "Create initial tables for articles, scorecards, and entities",
            []  # Tables are created by SQLAlchemy metadata
        ))
        
        # Migration 002: Add indexes for performance
        migrations.append((
            "002_add_indexes",
            "Add performance indexes",
            [
                "CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)",
                "CREATE INDEX IF NOT EXISTS idx_articles_publish_date ON articles(publish_date)",
                "CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_scorecards_overall_score ON scorecards(overall_score)",
                "CREATE INDEX IF NOT EXISTS idx_scorecards_created_at ON scorecards(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_scorecards_article_id ON scorecards(article_id)",
                "CREATE INDEX IF NOT EXISTS idx_entities_article_id ON entities(article_id)",
                "CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label)",
                "CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(text)",
            ]
        ))
        
        return migrations


class DatabaseInitializer:
    """Database initialization and setup utilities."""
    
    def __init__(self):
        """Initialize database initializer."""
        self.db_manager = get_db_manager()
        self.migration_manager = MigrationManager()
    
    def initialize_database(self, force_recreate: bool = False):
        """Initialize database with tables and migrations."""
        try:
            if force_recreate:
                logger.warning("Force recreating database tables")
                self.db_manager.drop_tables()
            
            # Create tables if they don't exist
            tables_created = self.migration_manager.check_and_create_tables()
            
            if tables_created:
                logger.info("Database tables created successfully")
            
            # Run migrations
            self.migration_manager.run_migrations()
            
            logger.info("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def verify_database_health(self) -> Dict[str, Any]:
        """Verify database health and return status information."""
        health_info = {
            'status': 'unknown',
            'tables_exist': False,
            'connection_ok': False,
            'migration_status': 'unknown',
            'error': None
        }
        
        try:
            # Test connection
            with self.db_manager.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                health_info['connection_ok'] = True
            
            # Check if tables exist
            inspector = inspect(self.db_manager.engine)
            existing_tables = inspector.get_table_names()
            required_tables = ['articles', 'scorecards', 'entities']
            
            health_info['tables_exist'] = all(table in existing_tables for table in required_tables)
            health_info['existing_tables'] = existing_tables
            
            # Check migration status
            try:
                applied_migrations = self.migration_manager.get_applied_migrations()
                health_info['applied_migrations'] = applied_migrations
                health_info['migration_status'] = 'ok'
            except Exception as e:
                health_info['migration_status'] = f'error: {e}'
            
            # Get storage stats
            try:
                with RetentionPolicy() as retention:
                    health_info['storage_stats'] = retention.get_storage_stats()
            except Exception as e:
                health_info['storage_stats_error'] = str(e)
            
            # Overall status
            if health_info['connection_ok'] and health_info['tables_exist']:
                health_info['status'] = 'healthy'
            else:
                health_info['status'] = 'needs_initialization'
                
        except Exception as e:
            health_info['status'] = 'error'
            health_info['error'] = str(e)
            logger.error(f"Database health check failed: {e}")
        
        return health_info
    
    def run_retention_cleanup(self) -> Dict[str, Any]:
        """Run retention policy cleanup and return statistics."""
        try:
            from curator.core.config import load_database_config
            
            db_config = load_database_config()
            article_retention_days = db_config['article_retention_days']
            scorecard_retention_days = db_config['scorecard_retention_days']
            
            with RetentionPolicy() as retention:
                cleanup_stats = retention.cleanup_old_data(
                    article_retention_days=article_retention_days,
                    scorecard_retention_days=scorecard_retention_days
                )
            
            logger.info(f"Retention cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Retention cleanup failed: {e}")
            raise


def init_database_cli():
    """CLI function to initialize database."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    force_recreate = '--force' in sys.argv
    
    try:
        initializer = DatabaseInitializer()
        initializer.initialize_database(force_recreate=force_recreate)
        print("Database initialization completed successfully")
        
        # Show health status
        health = initializer.verify_database_health()
        print(f"Database status: {health['status']}")
        
        if health.get('storage_stats'):
            stats = health['storage_stats']
            print(f"Articles: {stats['total_articles']} total, {stats['articles_last_7_days']} in last 7 days")
            print(f"Scorecards: {stats['total_scorecards']} total, {stats['scorecards_last_7_days']} in last 7 days")
            print(f"Entities: {stats['total_entities']} total")
        
    except Exception as e:
        print(f"Database initialization failed: {e}")
        sys.exit(1)


def cleanup_database_cli():
    """CLI function to run retention cleanup."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        initializer = DatabaseInitializer()
        stats = initializer.run_retention_cleanup()
        
        print("Retention cleanup completed successfully")
        print(f"Articles deleted: {stats['articles_deleted']}")
        print(f"Scorecards deleted: {stats['scorecards_deleted']}")
        print(f"Cleanup date: {stats['cleanup_date']}")
        
    except Exception as e:
        print(f"Retention cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'cleanup':
        cleanup_database_cli()
    else:
        init_database_cli()