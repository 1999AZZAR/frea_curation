# Design Document

## Overview

Curator is architected as a Flask-based web application with a modular design that separates concerns between web routing, content analysis, and external data sourcing. The system follows a three-tier architecture with presentation (Flask templates), business logic (scoring engine), and data access (NewsAPI integration) layers.

The application emphasizes real-time processing for manual analysis while supporting batch processing capabilities for topic-based curation. The scoring engine uses a weighted composite approach combining multiple NLP and metadata-based metrics to produce reliable content quality assessments.

## Architecture

### High-Level System Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        UI[Web Interface]
        Templates[Jinja2 Templates]
    end
    
    subgraph "Application Layer"
        Flask[Flask App Router]
        Analyzer[Content Analyzer]
        NewsSource[News Source Manager]
    end
    
    subgraph "External Services"
        NewsAPI[NewsAPI.org]
        Articles[Article URLs]
    end
    
    subgraph "Processing Pipeline"
        Parser[newspaper3k Parser]
        NLP[NLP Processing]
        Scorer[Scoring Engine]
    end
    
    UI --> Flask
    Flask --> Analyzer
    Flask --> NewsSource
    NewsSource --> NewsAPI
    Analyzer --> Parser
    Parser --> NLP
    NLP --> Scorer
    Scorer --> Templates
    Templates --> UI
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Flask
    participant Analyzer
    participant NewsSource
    participant NewsAPI
    participant Parser
    
    Note over User,Parser: Manual Analysis Flow
    User->>Flask: Submit URL
    Flask->>Analyzer: analyze_article(url)
    Analyzer->>Parser: parse_article(url)
    Parser-->>Analyzer: article_content
    Analyzer->>Analyzer: calculate_scores()
    Analyzer-->>Flask: scorecard_data
    Flask-->>User: results.html
    
    Note over User,Parser: Topic Curation Flow
    User->>Flask: Submit topic
    Flask->>NewsSource: get_articles(topic)
    NewsSource->>NewsAPI: fetch_articles()
    NewsAPI-->>NewsSource: article_urls[]
    NewsSource-->>Flask: article_list
    Flask->>Analyzer: batch_analyze(articles)
    Analyzer->>Parser: parse_multiple()
    Parser-->>Analyzer: parsed_articles[]
    Analyzer-->>Flask: ranked_results
    Flask-->>User: curation_results.html
```

## Components and Interfaces

### Core Application Components

#### 1. Flask Application Router (`app.py`)
- **Purpose**: HTTP request handling and route management
- **Key Routes**:
  - `GET /` - Homepage with input forms
  - `POST /analyze` - Manual article analysis endpoint
  - `POST /curate-topic` - Topic-based curation endpoint
- **Dependencies**: Flask, python-dotenv for configuration
- **Error Handling**: Custom error pages for 400/500 status codes

#### 2. Content Analyzer (`analyzer.py`)
- **Purpose**: Core scoring logic and NLP processing orchestration
- **Key Methods**:
  - `analyze_article(url)` - Single article analysis
  - `batch_analyze(articles)` - Multiple article processing
  - `calculate_composite_score(metrics)` - Weighted score calculation
- **Scoring Components**:
  - Readability Score (word count, content depth)
  - NER Density Score (spaCy named entity recognition)
  - Sentiment Score (NLTK VADER sentiment analysis)
  - TF-IDF Relevance Score (scikit-learn cosine similarity)
  - Recency Score (publication date decay function)

#### 3. News Source Manager (`news_source.py`)
- **Purpose**: NewsAPI integration and article URL management
- **Key Methods**:
  - `fetch_articles_by_topic(topic, region='id')` - Topic-based article retrieval
  - `validate_response(response)` - API response validation
  - `handle_rate_limits()` - Rate limiting and retry logic
- **Configuration**: API key management, request parameters, error handling

### Data Models and Structures

#### Article Data Model
```python
@dataclass
class Article:
    url: str
    title: str
    author: str
    publish_date: datetime
    content: str
    summary: str
    entities: List[Entity]
    
@dataclass
class Entity:
    text: str
    label: str  # PERSON, ORG, GPE, etc.
    confidence: float

@dataclass
class ScoreCard:
    overall_score: float
    readability_score: float
    ner_density_score: float
    sentiment_score: float
    tfidf_relevance_score: float
    recency_score: float
    article: Article
```

#### Configuration Model
```python
@dataclass
class ScoringConfig:
    readability_weight: float = 0.2
    ner_density_weight: float = 0.2
    sentiment_weight: float = 0.15
    tfidf_relevance_weight: float = 0.25
    recency_weight: float = 0.2
    min_word_count: int = 300
    max_articles_per_topic: int = 20
```

### External Service Interfaces

#### NewsAPI Integration
- **Endpoint**: `https://newsapi.org/v2/everything`
- **Authentication**: API key via header
- **Rate Limits**: 1000 requests/day (developer tier)
- **Response Format**: JSON with articles array
- **Error Handling**: HTTP status codes, rate limit headers

#### Article Parsing Interface
- **Library**: newspaper3k
- **Input**: Article URL
- **Output**: Structured article data (title, content, metadata)
- **Error Handling**: Parsing failures, timeout handling, encoding issues

## Data Models

### Database Schema (Future Enhancement)
While the initial implementation uses in-memory processing, the design supports future database integration:

```sql
-- Articles table for caching parsed content
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    url VARCHAR(2048) UNIQUE NOT NULL,
    title TEXT,
    author VARCHAR(255),
    publish_date TIMESTAMP,
    content TEXT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scores table for caching analysis results
CREATE TABLE scores (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    overall_score FLOAT,
    readability_score FLOAT,
    ner_density_score FLOAT,
    sentiment_score FLOAT,
    tfidf_relevance_score FLOAT,
    recency_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Named entities table
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    text VARCHAR(255),
    label VARCHAR(50),
    confidence FLOAT
);
```

### Scoring Algorithm Design

#### Composite Score Calculation
```python
def calculate_composite_score(metrics: Dict[str, float], config: ScoringConfig) -> float:
    """
    Weighted composite score calculation
    All component scores normalized to 0-100 range
    """
    weighted_sum = (
        metrics['readability'] * config.readability_weight +
        metrics['ner_density'] * config.ner_density_weight +
        metrics['sentiment'] * config.sentiment_weight +
        metrics['tfidf_relevance'] * config.tfidf_relevance_weight +
        metrics['recency'] * config.recency_weight
    )
    return min(100.0, max(0.0, weighted_sum))
```

#### Individual Score Calculations

**Readability Score**:
- Word count threshold (300+ words = higher score)
- Sentence complexity analysis
- Paragraph structure evaluation

**NER Density Score**:
- Entity count per 100 words
- Entity type diversity (PERSON, ORG, GPE, etc.)
- Entity confidence weighting

**Sentiment Score**:
- VADER compound score normalization
- Neutral sentiment preference for news content
- Extreme sentiment penalty for bias detection

**TF-IDF Relevance Score**:
- Cosine similarity between article and query terms
- Term frequency weighting
- Document frequency normalization

**Recency Score**:
- Exponential decay function based on publication date
- Configurable half-life parameter
- Maximum age threshold

## Error Handling

### Error Classification and Response Strategy

#### Client Errors (4xx)
- **Invalid URL Format**: Return 400 with validation message
- **Article Not Found**: Return 404 with alternative suggestions
- **Rate Limit Exceeded**: Return 429 with retry-after header

#### Server Errors (5xx)
- **NewsAPI Unavailable**: Return 503 with fallback message
- **Parsing Failures**: Return 500 with generic error page
- **NLP Processing Errors**: Graceful degradation with partial scores

#### Error Handling Implementation
```python
@app.errorhandler(400)
def bad_request(error):
    return render_template('errors/400.html', 
                         message="Invalid input provided"), 400

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html', 
                         message="An unexpected error occurred"), 500

def safe_analyze_article(url: str) -> Optional[ScoreCard]:
    """
    Wrapper function with comprehensive error handling
    """
    try:
        return analyze_article(url)
    except requests.RequestException:
        logger.error(f"Network error fetching {url}")
        return None
    except Exception as e:
        logger.error(f"Analysis error for {url}: {str(e)}")
        return None
```

### Retry Logic and Circuit Breaker Pattern
- **NewsAPI Requests**: Exponential backoff with jitter
- **Article Parsing**: 3 retry attempts with different user agents
- **Circuit Breaker**: Fail fast after consecutive failures

## Testing Strategy

### Unit Testing Approach

#### Component Testing
- **Analyzer Module**: Mock article content, test scoring algorithms
- **News Source Module**: Mock API responses, test error handling
- **Flask Routes**: Test request/response cycles with test client

#### Test Data Strategy
```python
# Sample test fixtures
SAMPLE_ARTICLE = {
    'url': 'https://example.com/article',
    'title': 'Sample Technology Article',
    'content': 'Lorem ipsum...' * 100,  # 300+ words
    'publish_date': datetime.now() - timedelta(hours=2)
}

SAMPLE_NEWSAPI_RESPONSE = {
    'status': 'ok',
    'totalResults': 5,
    'articles': [...]
}
```

#### Integration Testing
- **End-to-End Workflows**: Test complete user journeys
- **External Service Mocking**: Use responses library for API mocking
- **Database Integration**: Test with in-memory SQLite for future features

### Performance Testing Considerations
- **Load Testing**: Simulate concurrent user requests
- **Memory Profiling**: Monitor NLP model memory usage
- **Response Time Benchmarks**: Target <3s for single article analysis

### Security Testing
- **Input Validation**: Test URL injection attempts
- **API Key Protection**: Verify environment variable isolation
- **XSS Prevention**: Test template output escaping

## Deployment Architecture

### Development Environment
- **Local Development**: Flask development server
- **Dependencies**: Virtual environment with requirements.txt
- **Configuration**: .env file for local settings

### Production Deployment Options

#### Option 1: Traditional Server Deployment
- **Web Server**: Nginx reverse proxy
- **WSGI Server**: Gunicorn with multiple workers
- **Process Management**: Systemd service files
- **Monitoring**: Application logs and health checks

#### Option 2: Containerized Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

#### Option 3: Cloud Platform Deployment
- **Heroku**: Procfile with web dyno configuration
- **AWS Elastic Beanstalk**: Application bundle deployment
- **Google Cloud Run**: Containerized serverless deployment

### Environment Configuration Management
```python
# Production configuration
class ProductionConfig:
    DEBUG = False
    TESTING = False
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    SECRET_KEY = os.environ.get('SECRET_KEY')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
# Development configuration  
class DevelopmentConfig:
    DEBUG = True
    TESTING = False
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY', 'dev-key')
```

### Scalability Considerations

#### Horizontal Scaling
- **Stateless Design**: No server-side session storage
- **Load Balancing**: Multiple application instances
- **Caching Layer**: Redis for frequently accessed articles

#### Performance Optimization
- **Async Processing**: Celery task queue for batch operations
- **Content Delivery**: CDN for static assets
- **Database Optimization**: Connection pooling and query optimization

#### Monitoring and Observability
- **Application Metrics**: Response times, error rates
- **Business Metrics**: Articles processed, user engagement
- **Infrastructure Metrics**: CPU, memory, network usage