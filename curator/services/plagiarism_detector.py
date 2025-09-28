"""
Plagiarism Detection Service
Advanced content originality verification using multiple detection methods
"""

import re
import hashlib
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

@dataclass
class SimilaritySource:
    """Represents a source with similarity to the checked content"""
    url: str
    title: str
    similarity_percentage: float
    matched_text: str
    source_type: str = "web"  # web, academic, news, etc.

@dataclass
class PlagiarismResult:
    """Complete plagiarism detection result"""
    originality_score: float
    similarity_sources: List[SimilaritySource]
    total_words: int
    analyzed_sources: int
    detection_method: str
    recommendations: List[str]
    processing_time: float

class PlagiarismDetector:
    """Advanced plagiarism detection engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Common academic and news domains for source validation
        self.trusted_domains = [
            'scholar.google.com', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov',
            'news.bbc.co.uk', 'reuters.com', 'ap.org', 'cnn.com',
            'wikipedia.org', 'github.com', 'stackoverflow.com'
        ]
        
        # Stop words for text preprocessing
        self.stop_words = set(stopwords.words('english'))
    
    def detect_plagiarism(self, content: str, options: Dict[str, bool] = None) -> PlagiarismResult:
        """
        Main plagiarism detection method
        
        Args:
            content: Text content to check for plagiarism
            options: Detection options (deep_search, citation_check, paraphrase_check)
            
        Returns:
            PlagiarismResult with detailed analysis
        """
        start_time = time.time()
        
        if options is None:
            options = {
                'deep_search': True,
                'citation_check': False,
                'paraphrase_check': True
            }
        
        # Preprocess content
        processed_content = self._preprocess_text(content)
        content_sentences = sent_tokenize(processed_content)
        
        # Extract potential search queries
        search_queries = self._extract_search_queries(processed_content, content_sentences)
        
        # Search for similar content
        similarity_sources = []
        if options.get('deep_search', True):
            similarity_sources = self._search_similar_content(search_queries, processed_content)
        
        # Check for paraphrasing if enabled
        if options.get('paraphrase_check', True):
            paraphrase_sources = self._detect_paraphrasing(content_sentences)
            similarity_sources.extend(paraphrase_sources)
        
        # Calculate originality score
        originality_score = self._calculate_originality_score(similarity_sources)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(originality_score, similarity_sources, options)
        
        processing_time = time.time() - start_time
        
        return PlagiarismResult(
            originality_score=originality_score,
            similarity_sources=similarity_sources,
            total_words=len(content.split()),
            analyzed_sources=len(similarity_sources),
            detection_method="tfidf_cosine_similarity",
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        return text.lower()
    
    def _extract_search_queries(self, content: str, sentences: List[str]) -> List[str]:
        """Extract meaningful search queries from content"""
        queries = []
        
        # Use longest sentences as primary queries
        sorted_sentences = sorted(sentences, key=len, reverse=True)
        
        for sentence in sorted_sentences[:5]:  # Top 5 longest sentences
            if len(sentence.split()) >= 8:  # Minimum 8 words
                # Clean sentence for search
                clean_sentence = re.sub(r'[^\w\s]', ' ', sentence)
                clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
                queries.append(clean_sentence)
        
        # Extract key phrases (2-4 word combinations)
        words = word_tokenize(content)
        phrases = []
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+4])
            if len(phrase) > 10:  # Minimum phrase length
                phrases.append(phrase)
        
        # Add top phrases
        queries.extend(phrases[:3])
        
        return queries[:8]  # Limit to 8 queries
    
    def _search_similar_content(self, queries: List[str], original_content: str) -> List[SimilaritySource]:
        """Search for similar content using web search"""
        similarity_sources = []
        
        try:
            from .web_search import WebSearchService
            search_service = WebSearchService()
        except ImportError:
            self.logger.warning("WebSearchService not available, using fallback search")
            search_service = None
        
        # Use web search service or fallback
        for query in queries[:3]:  # Limit to 3 queries to avoid rate limiting
            try:
                if search_service:
                    search_results = search_service.search(query, num_results=5)
                else:
                    search_results = self._google_search(query)
                
                for result in search_results:
                    try:
                        # Extract content from URL
                        if search_service:
                            page_content = search_service.extract_page_content(result['url'])
                        else:
                            page_content = self._extract_page_content(result['url'])
                        
                        if page_content:
                            # Calculate similarity
                            similarity = self._calculate_text_similarity(original_content, page_content)
                            
                            if similarity > 0.1:  # 10% similarity threshold
                                # Extract matched text snippet
                                matched_text = self._extract_matching_snippet(original_content, page_content)
                                
                                source = SimilaritySource(
                                    url=result['url'],
                                    title=result.get('title', 'Unknown Title'),
                                    similarity_percentage=similarity * 100,
                                    matched_text=matched_text,
                                    source_type=self._classify_source_type(result['url'])
                                )
                                similarity_sources.append(source)
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing URL {result['url']}: {e}")
                        continue
                        
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Error searching for query '{query}': {e}")
                continue
        
        # Sort by similarity and return top results
        similarity_sources.sort(key=lambda x: x.similarity_percentage, reverse=True)
        return similarity_sources[:10]  # Top 10 results
    
    def _google_search(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """Perform Google search (simplified version for demo)"""
        # This is a simplified implementation
        # In production, you would use Google Custom Search API or similar
        
        # For demo purposes, return some mock results
        mock_results = [
            {
                'url': 'https://example.com/article1',
                'title': f'Article about {query[:20]}...',
                'snippet': f'Content related to {query} with detailed information...'
            },
            {
                'url': 'https://example.com/article2', 
                'title': f'Discussion on {query[:20]}...',
                'snippet': f'Analysis of {query} covering various aspects...'
            }
        ]
        
        return mock_results
    
    def _extract_page_content(self, url: str) -> Optional[str]:
        """Extract main content from a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            main_content = None
            
            # Look for common content containers
            for selector in ['main', 'article', '.content', '.post', '.entry-content']:
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
                main_content = self._preprocess_text(main_content)
                return main_content[:2000]  # Limit content length
            
        except Exception as e:
            self.logger.warning(f"Error extracting content from {url}: {e}")
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using TF-IDF and cosine similarity"""
        try:
            # Tokenize and preprocess
            texts = [text1, text2]
            
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _extract_matching_snippet(self, original_text: str, source_text: str, snippet_length: int = 100) -> str:
        """Extract a snippet that shows the matching content"""
        # Simple implementation - find the most similar sentence
        original_sentences = sent_tokenize(original_text)
        source_sentences = sent_tokenize(source_text)
        
        best_similarity = 0
        best_snippet = ""
        
        for orig_sent in original_sentences:
            for source_sent in source_sentences:
                similarity = self._calculate_text_similarity(orig_sent, source_sent)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_snippet = source_sent
        
        # Truncate if too long
        if len(best_snippet) > snippet_length:
            best_snippet = best_snippet[:snippet_length] + "..."
        
        return best_snippet
    
    def _detect_paraphrasing(self, sentences: List[str]) -> List[SimilaritySource]:
        """Detect potential paraphrasing (simplified implementation)"""
        # This would integrate with more sophisticated paraphrasing detection
        # For now, return empty list as it's complex to implement properly
        return []
    
    def _classify_source_type(self, url: str) -> str:
        """Classify the type of source based on URL"""
        domain = urlparse(url).netloc.lower()
        
        if any(trusted in domain for trusted in ['scholar', 'arxiv', 'pubmed']):
            return 'academic'
        elif any(trusted in domain for trusted in ['news', 'bbc', 'reuters', 'cnn']):
            return 'news'
        elif 'wikipedia' in domain:
            return 'encyclopedia'
        else:
            return 'web'
    
    def _calculate_originality_score(self, similarity_sources: List[SimilaritySource]) -> float:
        """Calculate overall originality score based on similarity sources"""
        if not similarity_sources:
            return 100.0
        
        # Calculate weighted average of similarities
        total_similarity = sum(source.similarity_percentage for source in similarity_sources)
        max_similarity = max(source.similarity_percentage for source in similarity_sources)
        
        # Originality score is inverse of maximum similarity found
        originality_score = max(0, 100 - max_similarity)
        
        # Apply penalty for multiple similar sources
        if len(similarity_sources) > 1:
            penalty = min(10, len(similarity_sources) * 2)
            originality_score = max(0, originality_score - penalty)
        
        return round(originality_score, 1)
    
    def _generate_recommendations(self, originality_score: float, 
                                similarity_sources: List[SimilaritySource], 
                                options: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if originality_score < 70:
            recommendations.append("High similarity detected. Consider significant rewriting of flagged sections.")
            recommendations.append("Review and properly cite all sources that inspired your content.")
        elif originality_score < 85:
            recommendations.append("Moderate similarity found. Consider paraphrasing similar sections.")
            recommendations.append("Ensure all borrowed ideas are properly attributed.")
        elif originality_score < 95:
            recommendations.append("Good originality with minor similarities. Consider reviewing flagged sections.")
        else:
            recommendations.append("Excellent! Your content appears to be highly original.")
        
        # Add specific recommendations based on source types
        academic_sources = [s for s in similarity_sources if s.source_type == 'academic']
        if academic_sources:
            recommendations.append("Academic sources found. Ensure proper citation format for academic writing.")
        
        # Add citation recommendations
        if options.get('citation_check', False):
            recommendations.append("Use citation detection features to identify properly referenced content.")
        
        return recommendations
