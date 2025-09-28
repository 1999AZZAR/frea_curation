"""
Web Search Service
Provides web search capabilities for plagiarism detection and content analysis
"""

import requests
import time
import logging
from typing import List, Dict, Optional
from urllib.parse import urlencode, urlparse
import json
import os
from bs4 import BeautifulSoup

class WebSearchService:
    """Service for performing web searches"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # API keys (in production, these would be in environment variables)
        self.google_api_key = os.environ.get('GOOGLE_API_KEY')
        self.google_cse_id = os.environ.get('GOOGLE_CSE_ID')
        self.serp_api_key = os.environ.get('SERP_API_KEY')
    
    def search(self, query: str, num_results: int = 10, source_type: str = "web") -> List[Dict[str, str]]:
        """
        Perform web search for given query
        
        Args:
            query: Search query
            num_results: Number of results to return
            source_type: Type of search (web, academic, news)
            
        Returns:
            List of search results with url, title, snippet
        """
        try:
            if source_type == "academic":
                return self._search_academic(query, num_results)
            elif source_type == "news":
                return self._search_news(query, num_results)
            else:
                return self._search_web(query, num_results)
        except Exception as e:
            self.logger.error(f"Error performing search: {e}")
            return []
    
    def _search_web(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Perform general web search"""
        # Try Google Custom Search API first
        if self.google_api_key and self.google_cse_id:
            return self._google_custom_search(query, num_results)
        
        # Fallback to SERP API
        if self.serp_api_key:
            return self._serp_search(query, num_results)
        
        # Final fallback to mock results
        return self._mock_search_results(query, num_results)
    
    def _search_academic(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Search academic sources"""
        # This would integrate with Google Scholar, arXiv, etc.
        # For now, return mock academic results
        return self._mock_academic_results(query, num_results)
    
    def _search_news(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Search news sources"""
        # This would integrate with news APIs
        # For now, return mock news results
        return self._mock_news_results(query, num_results)
    
    def _google_custom_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Use Google Custom Search API"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': min(num_results, 10)  # Google API limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    'url': item.get('link', ''),
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', '')
                })
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Google Custom Search API error: {e}")
            return self._mock_search_results(query, num_results)
    
    def _serp_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Use SERP API for search"""
        try:
            url = "https://serpapi.com/search"
            params = {
                'api_key': self.serp_api_key,
                'engine': 'google',
                'q': query,
                'num': num_results
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for result in data.get('organic_results', []):
                results.append({
                    'url': result.get('link', ''),
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', '')
                })
            
            return results
            
        except Exception as e:
            self.logger.warning(f"SERP API error: {e}")
            return self._mock_search_results(query, num_results)
    
    def _mock_search_results(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Generate mock search results for demo purposes"""
        mock_results = [
            {
                'url': f'https://example.com/article-{i}',
                'title': f'Article about {query[:30]} - Part {i}',
                'snippet': f'This article discusses {query} in detail, covering various aspects and providing comprehensive analysis. Content includes detailed explanations and examples.'
            }
            for i in range(1, min(num_results + 1, 6))
        ]
        
        return mock_results
    
    def _mock_academic_results(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Generate mock academic search results"""
        academic_results = []
        
        # Add scholar results
        academic_results.append({
            'url': f'https://scholar.google.com/scholar?q={query}',
            'title': f'Academic Study: {query[:40]}',
            'snippet': f'Peer-reviewed research paper examining {query} with comprehensive methodology and findings.'
        })
        
        # Add arxiv results
        for i in range(1, min(num_results, 3)):
            academic_results.append({
                'url': f'https://arxiv.org/abs/2023.1234{i}',
                'title': f'Research Paper: {query[:35]} Analysis',
                'snippet': f'Preprint research study analyzing {query} using advanced statistical methods and machine learning techniques.'
            })
        
        return academic_results[:num_results]
    
    def _mock_news_results(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Generate mock news search results"""
        news_results = [
            {
                'url': f'https://news.example.com/{query.replace(" ", "-")}-{i}',
                'title': f'Breaking: {query[:40]} News Update',
                'snippet': f'Latest news coverage on {query} with expert analysis and current developments.'
            }
            for i in range(1, min(num_results + 1, 4))
        ]
        
        return news_results[:num_results]
    
    def extract_page_content(self, url: str) -> Optional[str]:
        """Extract main content from a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Try to find main content
            main_content = None
            
            # Look for common content containers
            content_selectors = [
                'main', 'article', '.content', '.post', '.entry-content',
                '.article-content', '.post-content', '#content', '.main-content'
            ]
            
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element.get_text()
                    break
            
            # Fallback to body content
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text()
            
            if main_content:
                # Clean and normalize
                main_content = ' '.join(main_content.split())
                return main_content[:3000]  # Limit content length
            
        except Exception as e:
            self.logger.warning(f"Error extracting content from {url}: {e}")
        
        return None
    
    def validate_url(self, url: str) -> bool:
        """Validate if URL is accessible and relevant"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check if it's a valid web URL
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Basic content type check
            response = self.session.head(url, timeout=5)
            content_type = response.headers.get('content-type', '').lower()
            
            # Accept text-based content
            if 'text/html' in content_type or 'application/json' in content_type:
                return True
            
            return False
            
        except Exception:
            return False
