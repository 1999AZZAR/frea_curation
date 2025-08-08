"""
Unit tests for configuration loading and validation.

Tests the configuration management and environment variable
loading functionality.
"""

import unittest
import os
from unittest.mock import patch
from config import load_scoring_config, validate_required_env_vars
from models import ScoringConfig


class TestConfigLoading(unittest.TestCase):
    """Test cases for configuration loading."""
    
    def test_load_default_scoring_config(self):
        """Test loading default scoring configuration."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_scoring_config()
            self.assertIsInstance(config, ScoringConfig)
            self.assertEqual(config.readability_weight, 0.2)
            self.assertEqual(config.ner_density_weight, 0.2)
            self.assertEqual(config.sentiment_weight, 0.15)
            self.assertEqual(config.tfidf_relevance_weight, 0.25)
            self.assertEqual(config.recency_weight, 0.2)
            self.assertEqual(config.min_word_count, 300)
            self.assertEqual(config.max_articles_per_topic, 20)
    
    def test_load_custom_scoring_config(self):
        """Test loading custom scoring configuration from environment."""
        env_vars = {
            'READABILITY_WEIGHT': '0.3',
            'NER_DENSITY_WEIGHT': '0.25',
            'SENTIMENT_WEIGHT': '0.1',
            'TFIDF_RELEVANCE_WEIGHT': '0.2',
            'RECENCY_WEIGHT': '0.15',
            'MIN_WORD_COUNT': '500',
            'MAX_ARTICLES_PER_TOPIC': '15'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_scoring_config()
            self.assertEqual(config.readability_weight, 0.3)
            self.assertEqual(config.ner_density_weight, 0.25)
            self.assertEqual(config.sentiment_weight, 0.1)
            self.assertEqual(config.tfidf_relevance_weight, 0.2)
            self.assertEqual(config.recency_weight, 0.15)
            self.assertEqual(config.min_word_count, 500)
            self.assertEqual(config.max_articles_per_topic, 15)
    
    def test_load_config_with_invalid_values(self):
        """Test loading configuration with invalid environment values."""
        # Invalid float value
        with patch.dict(os.environ, {'READABILITY_WEIGHT': 'invalid'}, clear=True):
            with self.assertRaises(ValueError):
                load_scoring_config()
        
        # Invalid int value
        with patch.dict(os.environ, {'MIN_WORD_COUNT': 'invalid'}, clear=True):
            with self.assertRaises(ValueError):
                load_scoring_config()
        
        # Invalid weight sum
        env_vars = {
            'READABILITY_WEIGHT': '0.5',
            'NER_DENSITY_WEIGHT': '0.5',
            'SENTIMENT_WEIGHT': '0.5',
            'TFIDF_RELEVANCE_WEIGHT': '0.5',
            'RECENCY_WEIGHT': '0.5'
        }
        with patch.dict(os.environ, env_vars, clear=True):
            with self.assertRaises(ValueError):
                load_scoring_config()
    
    def test_partial_environment_override(self):
        """Test loading config with partial environment variable override."""
        env_vars = {
            'READABILITY_WEIGHT': '0.3',
            'MIN_WORD_COUNT': '400'
            # Other values should use defaults
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_scoring_config()
            self.assertEqual(config.readability_weight, 0.3)
            self.assertEqual(config.min_word_count, 400)
            # These should be defaults
            self.assertEqual(config.ner_density_weight, 0.2)
            self.assertEqual(config.sentiment_weight, 0.15)


class TestEnvironmentValidation(unittest.TestCase):
    """Test cases for environment variable validation."""
    
    def test_validate_with_required_vars_present(self):
        """Test validation when all required variables are present."""
        with patch.dict(os.environ, {'NEWS_API_KEY': 'test-key'}, clear=True):
            result = validate_required_env_vars()
            self.assertTrue(result)
    
    def test_validate_with_missing_required_vars(self):
        """Test validation when required variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                validate_required_env_vars()
            self.assertIn('NEWS_API_KEY', str(context.exception))
    
    def test_validate_with_empty_required_vars(self):
        """Test validation when required variables are empty."""
        with patch.dict(os.environ, {'NEWS_API_KEY': ''}, clear=True):
            with self.assertRaises(ValueError) as context:
                validate_required_env_vars()
            self.assertIn('NEWS_API_KEY', str(context.exception))


if __name__ == '__main__':
    unittest.main()