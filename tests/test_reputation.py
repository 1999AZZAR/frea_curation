"""
Unit tests for reputation scoring system.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import patch

from curator.core.reputation import (
    ReputationManager, 
    compute_reputation_score,
    get_reputation_manager
)


class TestReputationManager:
    """Test cases for ReputationManager class."""
    
    def test_initialization_with_defaults(self):
        """Test that manager initializes with default reputation data."""
        manager = ReputationManager()
        
        # Should have some default high-credibility domains
        assert manager.get_domain_reputation('nytimes.com') > 70.0
        assert manager.get_domain_reputation('bbc.com') > 70.0
        assert manager.get_domain_reputation('reuters.com') > 70.0
        
        # Should have some default low-credibility domains
        assert manager.get_domain_reputation('infowars.com') < 40.0
        
        # Unknown domains should get neutral score
        assert manager.get_domain_reputation('unknown-domain.com') == 50.0
    
    def test_government_domain_scoring(self):
        """Test that government domains get high reputation scores."""
        manager = ReputationManager()
        
        assert manager.get_domain_reputation('cdc.gov') >= 85.0
        assert manager.get_domain_reputation('nasa.gov') >= 85.0
        assert manager.get_domain_reputation('example.edu') >= 85.0
    
    def test_www_prefix_handling(self):
        """Test that www prefixes are handled correctly."""
        manager = ReputationManager()
        
        # Should get same score with or without www
        score1 = manager.get_domain_reputation('nytimes.com')
        score2 = manager.get_domain_reputation('www.nytimes.com')
        assert score1 == score2
    
    def test_subdomain_scoring(self):
        """Test that subdomains get slightly lower scores than parent domains."""
        manager = ReputationManager()
        
        parent_score = manager.get_domain_reputation('bbc.com')
        subdomain_score = manager.get_domain_reputation('news.bbc.com')
        
        # Subdomain should have slightly lower score
        assert subdomain_score < parent_score
        assert subdomain_score >= parent_score - 10.0  # But not too much lower
    
    def test_author_reputation_default(self):
        """Test author reputation defaults to neutral."""
        manager = ReputationManager()
        
        assert manager.get_author_reputation('Unknown Author') == 50.0
        assert manager.get_author_reputation('') == 50.0
        assert manager.get_author_reputation(None) == 50.0
    
    def test_update_domain_reputation(self):
        """Test updating domain reputation scores."""
        manager = ReputationManager()
        
        # Update a domain reputation
        manager.update_domain_reputation('example.com', 75.0)
        assert manager.get_domain_reputation('example.com') == 75.0
        
        # Test validation
        with pytest.raises(ValueError):
            manager.update_domain_reputation('example.com', 150.0)  # Invalid score
        
        with pytest.raises(ValueError):
            manager.update_domain_reputation('example.com', -10.0)  # Invalid score
    
    def test_update_author_reputation(self):
        """Test updating author reputation scores."""
        manager = ReputationManager()
        
        # Update an author reputation
        manager.update_author_reputation('John Doe', 80.0)
        assert manager.get_author_reputation('John Doe') == 80.0
        assert manager.get_author_reputation('john doe') == 80.0  # Case insensitive
        
        # Test validation
        with pytest.raises(ValueError):
            manager.update_author_reputation('Jane Doe', 150.0)  # Invalid score
    
    def test_add_credibility_domains(self):
        """Test adding domains to credibility lists."""
        manager = ReputationManager()
        
        # Add high credibility domain
        manager.add_high_credibility_domain('newtrusted.com')
        assert manager.get_domain_reputation('newtrusted.com') == 85.0
        
        # Add low credibility domain
        manager.add_low_credibility_domain('newuntrusted.com')
        assert manager.get_domain_reputation('newuntrusted.com') == 25.0
    
    def test_reputation_stats(self):
        """Test getting reputation statistics."""
        manager = ReputationManager()
        stats = manager.get_reputation_stats()
        
        assert 'total_domains' in stats
        assert 'high_credibility_domains' in stats
        assert 'low_credibility_domains' in stats
        assert 'total_authors' in stats
        
        assert stats['total_domains'] >= 0
        assert stats['high_credibility_domains'] >= 0
        assert stats['low_credibility_domains'] >= 0
        assert stats['total_authors'] >= 0
    
    def test_save_and_load_reputation_data(self):
        """Test saving and loading reputation data from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create manager and add some data
            manager1 = ReputationManager(temp_path)
            manager1.update_domain_reputation('test.com', 75.0)
            manager1.update_author_reputation('Test Author', 85.0)
            manager1.save_reputation_data()
            
            # Create new manager and load the data
            manager2 = ReputationManager(temp_path)
            assert manager2.get_domain_reputation('test.com') == 75.0
            assert manager2.get_author_reputation('Test Author') == 85.0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestComputeReputationScore:
    """Test cases for compute_reputation_score function."""
    
    def test_basic_reputation_scoring(self):
        """Test basic reputation scoring with URL and author."""
        # Test with known high-reputation domain
        score1 = compute_reputation_score('https://www.nytimes.com/article', 'John Doe')
        assert 40.0 <= score1 <= 100.0  # Should be reasonably high
        
        # Test with unknown domain
        score2 = compute_reputation_score('https://unknown-site.com/article', '')
        assert 40.0 <= score2 <= 60.0  # Should be around neutral
    
    def test_url_parsing_edge_cases(self):
        """Test reputation scoring with various URL formats."""
        # Test with different URL formats
        score1 = compute_reputation_score('https://bbc.com/news/article')
        score2 = compute_reputation_score('http://www.bbc.com/news/article')
        score3 = compute_reputation_score('bbc.com/news/article')
        
        # All should give similar scores (domain is the same)
        assert abs(score1 - score2) <= 5.0
        assert abs(score1 - score3) <= 5.0
    
    def test_author_weight_in_scoring(self):
        """Test that author reputation affects the composite score."""
        with patch.object(ReputationManager, 'get_author_reputation') as mock_author:
            # Test with high author reputation
            mock_author.return_value = 90.0
            score_high_author = compute_reputation_score('https://unknown.com/article', 'Famous Author')
            
            # Test with low author reputation
            mock_author.return_value = 20.0
            score_low_author = compute_reputation_score('https://unknown.com/article', 'Unknown Author')
            
            # High author reputation should increase overall score
            assert score_high_author > score_low_author
    
    def test_error_handling(self):
        """Test that reputation scoring handles errors gracefully."""
        # Test with invalid URL
        score = compute_reputation_score('not-a-url', 'Author')
        assert score == 50.0  # Should return neutral fallback
        
        # Test with empty URL
        score = compute_reputation_score('', 'Author')
        assert score == 50.0  # Should return neutral fallback


class TestGlobalReputationManager:
    """Test cases for global reputation manager."""
    
    def test_get_reputation_manager_singleton(self):
        """Test that get_reputation_manager returns singleton instance."""
        manager1 = get_reputation_manager()
        manager2 = get_reputation_manager()
        
        # Should be the same instance
        assert manager1 is manager2
    
    @patch('curator.core.reputation._reputation_manager', None)
    def test_lazy_initialization(self):
        """Test that reputation manager is lazily initialized."""
        # Reset global instance
        import curator.core.reputation
        curator.core.reputation._reputation_manager = None
        
        # Should create new instance on first call
        manager = get_reputation_manager()
        assert manager is not None
        assert isinstance(manager, ReputationManager)


if __name__ == '__main__':
    pytest.main([__file__])