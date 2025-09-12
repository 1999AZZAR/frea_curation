"""
Source reputation and credibility scoring system.

Maintains domain reputation tables and optional author reputation
to provide credibility signals for content curation.
"""

import os
import json
from typing import Dict, Optional, Set
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class ReputationManager:
    """Manages domain and author reputation scoring."""
    
    def __init__(self, reputation_data_path: Optional[str] = None):
        """Initialize reputation manager with optional data file path."""
        self.reputation_data_path = reputation_data_path or os.environ.get(
            'REPUTATION_DATA_PATH', 
            'curator/data/reputation.json'
        )
        self._domain_reputation: Dict[str, float] = {}
        self._author_reputation: Dict[str, float] = {}
        self._high_credibility_domains: Set[str] = set()
        self._low_credibility_domains: Set[str] = set()
        self._load_reputation_data()
    
    def _load_reputation_data(self) -> None:
        """Load reputation data from file or initialize with defaults."""
        try:
            if os.path.exists(self.reputation_data_path):
                with open(self.reputation_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._domain_reputation = data.get('domain_reputation', {})
                    self._author_reputation = data.get('author_reputation', {})
                    self._high_credibility_domains = set(data.get('high_credibility_domains', []))
                    self._low_credibility_domains = set(data.get('low_credibility_domains', []))
                logger.info(f"Loaded reputation data from {self.reputation_data_path}")
            else:
                # Initialize with default high-credibility domains
                self._initialize_default_reputation()
                logger.info("Initialized with default reputation data")
        except Exception as e:
            logger.warning(f"Failed to load reputation data: {e}, using defaults")
            self._initialize_default_reputation()
    
    def _initialize_default_reputation(self) -> None:
        """Initialize with default high-credibility news sources."""
        # High-credibility news sources (major newspapers, news agencies, academic)
        high_credibility = {
            # Major newspapers
            'nytimes.com', 'washingtonpost.com', 'wsj.com', 'ft.com',
            'theguardian.com', 'bbc.com', 'reuters.com', 'apnews.com',
            'npr.org', 'pbs.org', 'cnn.com', 'abcnews.go.com',
            'cbsnews.com', 'nbcnews.com', 'usatoday.com',
            
            # International quality sources
            'economist.com', 'bloomberg.com', 'politico.com',
            'theatlantic.com', 'newyorker.com', 'time.com',
            'newsweek.com', 'latimes.com', 'chicagotribune.com',
            
            # Tech and science
            'nature.com', 'science.org', 'scientificamerican.com',
            'arstechnica.com', 'wired.com', 'techcrunch.com',
            
            # Academic and research
            'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com',
            
            # Government and official sources
            'gov.uk', 'gov.ca', 'europa.eu', 'who.int', 'cdc.gov',
            'fda.gov', 'nasa.gov', 'noaa.gov'
        }
        
        # Low-credibility domains (known for misinformation or low quality)
        low_credibility = {
            # Common content farms and clickbait sites
            'buzzfeed.com', 'upworthy.com', 'viralnova.com',
            'clickhole.com', 'theonion.com',  # satire
            
            # Known misinformation sources (examples)
            'infowars.com', 'naturalnews.com', 'beforeitsnews.com'
        }
        
        self._high_credibility_domains = high_credibility
        self._low_credibility_domains = low_credibility
        
        # Set reputation scores based on categories
        for domain in high_credibility:
            self._domain_reputation[domain] = 85.0  # High reputation
        
        for domain in low_credibility:
            self._domain_reputation[domain] = 25.0  # Low reputation
    
    def save_reputation_data(self) -> None:
        """Save current reputation data to file."""
        try:
            os.makedirs(os.path.dirname(self.reputation_data_path), exist_ok=True)
            data = {
                'domain_reputation': self._domain_reputation,
                'author_reputation': self._author_reputation,
                'high_credibility_domains': list(self._high_credibility_domains),
                'low_credibility_domains': list(self._low_credibility_domains)
            }
            with open(self.reputation_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved reputation data to {self.reputation_data_path}")
        except Exception as e:
            logger.error(f"Failed to save reputation data: {e}")
    
    def get_domain_reputation(self, domain: str) -> float:
        """Get reputation score for a domain (0-100 scale).
        
        Args:
            domain: Domain name (e.g., 'example.com')
            
        Returns:
            Reputation score between 0-100, with 50 as neutral default
        """
        if not domain:
            return 50.0
        
        domain = domain.lower().strip()
        
        # Remove www prefix for consistency
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check explicit reputation scores first
        if domain in self._domain_reputation:
            return self._domain_reputation[domain]
        
        # Check category-based scoring
        if domain in self._high_credibility_domains:
            return 85.0
        
        if domain in self._low_credibility_domains:
            return 25.0
        
        # Check for subdomain patterns (e.g., news.example.com)
        domain_parts = domain.split('.')
        if len(domain_parts) >= 2:
            parent_domain = '.'.join(domain_parts[-2:])
            if parent_domain in self._domain_reputation:
                # Slightly lower score for subdomains
                return max(0.0, self._domain_reputation[parent_domain] - 5.0)
            
            if parent_domain in self._high_credibility_domains:
                return 80.0  # Slightly lower for subdomains
            
            if parent_domain in self._low_credibility_domains:
                return 30.0  # Slightly higher for subdomains
        
        # Check for government domains (.gov, .edu)
        if domain.endswith('.gov') or domain.endswith('.edu'):
            return 90.0
        
        # Check for academic domains (.ac.uk, .edu.au, etc.)
        if any(domain.endswith(suffix) for suffix in ['.ac.uk', '.edu.au', '.ac.jp']):
            return 88.0
        
        # Default neutral reputation for unknown domains
        return 50.0
    
    def get_author_reputation(self, author: str) -> float:
        """Get reputation score for an author (0-100 scale).
        
        Args:
            author: Author name
            
        Returns:
            Reputation score between 0-100, with 50 as neutral default
        """
        if not author or not author.strip():
            return 50.0
        
        author_key = author.lower().strip()
        return self._author_reputation.get(author_key, 50.0)
    
    def update_domain_reputation(self, domain: str, score: float) -> None:
        """Update reputation score for a domain.
        
        Args:
            domain: Domain name
            score: New reputation score (0-100)
        """
        if not domain:
            return
        
        domain = domain.lower().strip()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        if not 0.0 <= score <= 100.0:
            raise ValueError("Reputation score must be between 0 and 100")
        
        self._domain_reputation[domain] = float(score)
        logger.info(f"Updated domain reputation: {domain} = {score}")
    
    def update_author_reputation(self, author: str, score: float) -> None:
        """Update reputation score for an author.
        
        Args:
            author: Author name
            score: New reputation score (0-100)
        """
        if not author or not author.strip():
            return
        
        if not 0.0 <= score <= 100.0:
            raise ValueError("Reputation score must be between 0 and 100")
        
        author_key = author.lower().strip()
        self._author_reputation[author_key] = float(score)
        logger.info(f"Updated author reputation: {author} = {score}")
    
    def add_high_credibility_domain(self, domain: str) -> None:
        """Add domain to high credibility list."""
        if domain:
            domain = domain.lower().strip()
            if domain.startswith('www.'):
                domain = domain[4:]
            self._high_credibility_domains.add(domain)
            self._domain_reputation[domain] = 85.0
    
    def add_low_credibility_domain(self, domain: str) -> None:
        """Add domain to low credibility list."""
        if domain:
            domain = domain.lower().strip()
            if domain.startswith('www.'):
                domain = domain[4:]
            self._low_credibility_domains.add(domain)
            self._domain_reputation[domain] = 25.0
    
    def get_reputation_stats(self) -> Dict[str, int]:
        """Get statistics about reputation data."""
        return {
            'total_domains': len(self._domain_reputation),
            'high_credibility_domains': len(self._high_credibility_domains),
            'low_credibility_domains': len(self._low_credibility_domains),
            'total_authors': len(self._author_reputation)
        }


# Global reputation manager instance
_reputation_manager: Optional[ReputationManager] = None


def get_reputation_manager() -> ReputationManager:
    """Get global reputation manager instance (lazy initialization)."""
    global _reputation_manager
    if _reputation_manager is None:
        _reputation_manager = ReputationManager()
    return _reputation_manager


def compute_reputation_score(url: str, author: str = "") -> float:
    """Compute reputation score for an article based on domain and author.
    
    Args:
        url: Article URL
        author: Article author (optional)
        
    Returns:
        Composite reputation score (0-100)
    """
    try:
        manager = get_reputation_manager()
        
        # Extract domain from URL
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Get domain reputation (70% weight)
        domain_score = manager.get_domain_reputation(domain)
        
        # Get author reputation (30% weight) if author provided
        author_score = 50.0  # Default neutral
        if author and author.strip():
            author_score = manager.get_author_reputation(author)
        
        # Weighted combination: 70% domain, 30% author
        composite_score = (domain_score * 0.7) + (author_score * 0.3)
        
        return round(max(0.0, min(100.0, composite_score)), 2)
        
    except Exception as e:
        logger.warning(f"Failed to compute reputation score for {url}: {e}")
        return 50.0  # Neutral fallback