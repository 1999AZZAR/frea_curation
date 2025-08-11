"""
NewsAPI integration service.
"""

import os
import time
import random
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsAPIError(Exception):
    pass


class RateLimitError(NewsAPIError):
    pass


class NewsSource:
    BASE_URL = "https://newsapi.org/v2"
    DEFAULT_LANGUAGE = "en"
    DEFAULT_SORT_BY = "publishedAt"
    DEFAULT_PAGE_SIZE = 20
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 1.0
    TIMEOUT = 30

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('NEWS_API_KEY')
        if not self.api_key:
            raise NewsAPIError("NewsAPI key is required. Set NEWS_API_KEY environment variable.")
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            'X-API-Key': self.api_key,
            'User-Agent': 'AI-Content-Curator/1.0'
        })
        self.last_request_time = 0
        self.min_request_interval = 1.0

    def _handle_rate_limiting(self) -> None:
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            jitter = random.uniform(0, 0.5)
            time.sleep(sleep_time + jitter)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._handle_rate_limiting()
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            logger.info(f"Making request to {endpoint} with params: {params}")
            response = self.session.get(url, params=params, timeout=self.TIMEOUT)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit exceeded. Retry after {retry_after} seconds.")
                raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds.")
            response.raise_for_status()
            data = response.json()
            self._validate_response(data)
            return data
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            raise NewsAPIError("Request timeout. Please try again later.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for {url}")
            raise NewsAPIError("Connection error. Please check your internet connection.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}")
            if e.response.status_code == 401:
                raise NewsAPIError("Invalid API key. Please check your NewsAPI key.")
            elif e.response.status_code == 400:
                raise NewsAPIError("Invalid request parameters.")
            else:
                raise NewsAPIError(f"HTTP error: {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {str(e)}")
            raise NewsAPIError(f"Request failed: {str(e)}")

    def _validate_response(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise NewsAPIError("Invalid response format: expected JSON object")
        status = data.get('status')
        if status != 'ok':
            error_code = data.get('code', 'unknown')
            error_message = data.get('message', 'Unknown error')
            logger.error(f"API error {error_code}: {error_message}")
            raise NewsAPIError(f"API error {error_code}: {error_message}")
        if 'articles' not in data:
            raise NewsAPIError("Invalid response: missing articles field")
        if not isinstance(data['articles'], list):
            raise NewsAPIError("Invalid response: articles must be a list")

    def fetch_articles_by_topic(
        self,
        topic: str,
        language: str = DEFAULT_LANGUAGE,
        sort_by: str = DEFAULT_SORT_BY,
        page_size: int = DEFAULT_PAGE_SIZE,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        if page_size < 1 or page_size > 100:
            raise ValueError("Page size must be between 1 and 100")
        if sort_by not in ['publishedAt', 'relevancy', 'popularity']:
            raise ValueError("sort_by must be one of: publishedAt, relevancy, popularity")
        params = {
            'q': topic.strip(),
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size,
        }
        if from_date:
            params['from'] = from_date.strftime('%Y-%m-%d')
        if to_date:
            params['to'] = to_date.strftime('%Y-%m-%d')
        if not from_date and not to_date:
            from_date = datetime.now() - timedelta(days=7)
            params['from'] = from_date.strftime('%Y-%m-%d')
        try:
            data = self._make_request('everything', params)
            articles = data['articles']
            valid_articles = []
            for article in articles:
                if self._is_valid_article(article):
                    valid_articles.append(article)
                else:
                    logger.debug(f"Skipping invalid article: {article.get('title', 'Unknown')}")
            logger.info(f"Fetched {len(valid_articles)} valid articles for topic: {topic}")
            return valid_articles
        except RateLimitError:
            raise
        except NewsAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching articles: {str(e)}")
            raise NewsAPIError(f"Unexpected error: {str(e)}")

    def _is_valid_article(self, article: Dict[str, Any]) -> bool:
        required_fields = ['title', 'url', 'publishedAt']
        for field in required_fields:
            value = article.get(field)
            if not value or (isinstance(value, str) and not value.strip()):
                return False
        url = article.get('url', '')
        if not url.startswith(('http://', 'https://')):
            return False
        description = article.get('description', '')
        content = article.get('content', '')
        if not description and not content:
            return False
        return True

    def get_article_urls(self, topic: str, max_articles: int = 20) -> List[str]:
        if max_articles < 1:
            raise ValueError("max_articles must be positive")
        page_size = min(max_articles, 100)
        articles = self.fetch_articles_by_topic(
            topic=topic,
            page_size=page_size,
            sort_by='relevancy',
        )
        # Canonicalize and deduplicate while preserving order
        seen = set()
        urls: List[str] = []
        for article in articles:
            url = article.get('url') or ''
            if not url:
                continue
            cu = self._canonicalize_url(url)
            if cu in seen:
                continue
            seen.add(cu)
            urls.append(cu)
            if len(urls) >= max_articles:
                break
        logger.info(f"Extracted {len(urls)} URLs for topic: {topic}")
        return urls

    def check_api_status(self) -> Dict[str, Any]:
        try:
            params = {'q': 'test', 'pageSize': 1, 'language': 'en'}
            response = self.session.get(f"{self.BASE_URL}/everything", params=params, timeout=self.TIMEOUT)
            status_info = {
                'status_code': response.status_code,
                'rate_limit_remaining': response.headers.get('X-RateLimit-Remaining'),
                'rate_limit_reset': response.headers.get('X-RateLimit-Reset'),
                'api_accessible': response.status_code == 200,
            }
            logger.info(f"API status check: {status_info}")
            return status_info
        except Exception as e:
            logger.error(f"API status check failed: {str(e)}")
            raise NewsAPIError(f"Status check failed: {str(e)}")

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        """Normalize URL to reduce duplicates from tracking params, fragments, and case.

        - Lowercase scheme and host
        - Strip fragment
        - Remove common tracking query params (utm_*, gclid, fbclid)
        - Sort remaining query parameters for stability
        """
        try:
            parsed = urlparse(url)
            scheme = (parsed.scheme or 'https').lower()
            netloc = (parsed.netloc or '').lower()
            if netloc.startswith('www.'):
                netloc = netloc[4:]
            # Filter query params
            params = []
            for k, v in parse_qsl(parsed.query, keep_blank_values=True):
                lk = k.lower()
                if lk.startswith('utm_') or lk in {'gclid', 'fbclid', 'mc_eid', 'mc_cid'}:
                    continue
                params.append((k, v))
            params.sort()
            canonical = urlunparse((
                scheme,
                netloc,
                parsed.path or '/',
                '',
                urlencode(params, doseq=True),
                ''  # no fragment
            ))
            return canonical
        except Exception:
            return url
