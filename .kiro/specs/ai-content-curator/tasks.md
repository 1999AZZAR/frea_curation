# Implementation Plan

- [x] 1. Set up project structure and core dependencies
  - Create directory structure for Flask application with templates and static folders
  - Initialize requirements.txt with all necessary dependencies (Flask, newspaper3k, spaCy, nltk, scikit-learn, requests, python-dotenv)
  - Create .env template file for configuration management
  - Set up basic Flask application entry point with development configuration
  - _Requirements: 5.1, 5.2_

- [x] 2. Implement core data models and configuration
  - Create data classes for Article, Entity, ScoreCard, and ScoringConfig using Python dataclasses
  - Implement configuration loading from environment variables using python-dotenv
  - Create validation functions for URL input using validators library
  - Write unit tests for data models and configuration loading
  - _Requirements: 5.1, 5.2, 6.5_

- [x] 3. Build article parsing and content extraction
  - Implement article parsing functionality using newspaper3k library
  - Create error handling for unparsable articles and network timeouts
  - Add content validation to ensure minimum word count requirements
  - Implement retry logic for failed parsing attempts with different user agents
  - Write unit tests for parsing functionality with mocked article content
  - _Requirements: 1.1, 6.2, 6.4_

- [x] 4. Develop NewsAPI integration module
  - Create NewsSource class with methods for fetching articles by topic
  - Implement API key authentication and request parameter configuration
  - Add rate limiting handling and retry logic with exponential backoff
  - Create response validation and error handling for empty or failed API responses
  - Write unit tests using mocked API responses
  - _Requirements: 2.1, 2.4, 2.5, 5.3, 6.3_

- [ ] 5. Implement NLP processing components
- [x] 5.1 Set up spaCy named entity recognition
  - Download and configure spaCy language model for entity extraction
  - Implement NER processing function to extract and categorize named entities
  - Create entity confidence scoring and filtering logic
  - Write unit tests for entity extraction with sample text content
  - _Requirements: 3.2_

- [x] 5.2 Implement NLTK sentiment analysis
  - Set up NLTK VADER sentiment analyzer with required data downloads
  - Create sentiment scoring function that normalizes compound scores to 0-100 range
  - Implement neutral sentiment preference logic for news content
  - Write unit tests for sentiment analysis with various text samples
  - _Requirements: 3.3_

- [x] 5.3 Build TF-IDF relevance scoring
  - Implement TF-IDF vectorization using scikit-learn TfidfVectorizer
  - Create cosine similarity calculation between article content and query terms
  - Add text preprocessing and normalization for better relevance matching
  - Write unit tests for relevance scoring with known article-query pairs
  - _Requirements: 3.4_

- [ ] 6. Create comprehensive scoring engine
- [x] 6.1 Implement individual scoring algorithms
  - Create readability scoring based on word count and content structure analysis
  - Implement NER density scoring using entity count per word ratio
  - Build recency scoring with exponential decay function based on publication date
  - Write unit tests for each individual scoring component
  - _Requirements: 3.1, 3.2, 3.5_

- [x] 6.2 Build composite scoring system
  - Implement weighted composite score calculation using configurable weights
  - Create score normalization functions to ensure 0-100 range for all components
  - Add configuration management for adjustable scoring weights
  - Write comprehensive unit tests for composite scoring with various input combinations
  - _Requirements: 3.6_

- [ ] 6.3 Integrate complete analysis pipeline
  - Create main analyze_article function that orchestrates all scoring components
  - Implement batch_analyze function for processing multiple articles efficiently
  - Add comprehensive error handling and graceful degradation for partial failures
  - Write integration tests for complete analysis workflow
  - _Requirements: 1.2, 2.2, 6.2_

- [ ] 7. Integrate Flask routes with analysis pipeline
- [x] 7.1 Implement manual article analysis endpoint
  - Integrate article parsing and scoring pipeline into POST /analyze route
  - Add URL validation using existing validation functions
  - Implement error handling for invalid URLs and parsing failures
  - Return structured scorecard data to results template
  - Write unit tests for analysis endpoint with various input scenarios
  - _Requirements: 1.1, 1.3, 1.4, 1.5, 6.1, 6.4_

- [x] 7.2 Build topic-based curation endpoint
  - Integrate NewsAPI fetching with article parsing and scoring pipeline into POST /curate-topic route
  - Add topic keyword validation using existing validation functions
  - Implement batch processing for multiple articles with progress tracking
  - Sort and rank articles by composite score in descending order
  - Write unit tests for curation endpoint with mocked NewsAPI responses
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 8. Design and implement user interface templates
  - [x] 8.1 Create base template with modern styling
    - Set up Tailwind CSS integration (compiled) for utility-first styling approach
    - Create `base.html` with dark theme and glassmorphism effects
    - Implement responsive navigation and layout (mobile-first)
    - Add Font Awesome icons integration for visual elements
    - _Requirements: 4.4_

  - [x] 8.2 Build homepage interface
    - Create `index.html` with two clear entry points: Analyze and Curate
    - Implement client-side validation, loading states, and error feedback
    - Ensure responsive design across devices
    - _Requirements: 4.1_

  - [x] 8.3 Design scorecard results interface
    - Create `results.html` for individual analysis
    - Prominent overall score with progress bars for each component
    - Component breakdowns with clear visual hierarchy
    - Placeholder for entities (hidden until back-end provides data)
    - _Requirements: 1.3, 4.2_

  - [x] 8.4 Build topic curation results interface
    - Create `curation_results.html` for ranked article lists
    - Article cards with title, score, summary, and source link
    - Sorting and min-score filtering, search, and pagination controls
    - _Requirements: 2.3, 4.3_

  - [x] 9. Create custom error page templates
    - `400.html`, `404.html`, `500.html` extend base template
    - Consistent UI and helpful messaging
    - _Requirements: 6.4_

  - [x] 10. Add static assets and styling
    - Tailwind CSS build pipeline (assets -> compiled `static/css/tailwind.css`)
    - Inter font and refined typography for professional look
    - Minor transitions and hover states aligned with glassmorphism
    - _Requirements: 4.4_

- [ ] 11. Create integration tests for complete workflows
  - Create end-to-end tests for complete user workflows using Flask test client
  - Test external service integration with mocked responses using responses library
  - Implement performance tests to ensure response time requirements are met
  - Write tests for concurrent user scenarios and resource management
  - _Requirements: 7.3_

- [ ] 12. Prepare deployment configuration
  - Create production-ready configuration with environment-specific settings
  - Write Dockerfile for containerized deployment with optimized image size
  - Create docker-compose.yml for local development and testing environment
  - Add deployment scripts and documentation for various hosting platforms
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 13. Documentation and final integration
  - Write comprehensive README.md with setup instructions and API documentation
  - Add code comments and docstrings following PEP8 standards
  - Perform final integration testing and bug fixes before deployment
  - _Requirements: 5.4, 5.5_

- [x] 14. Preserve API compatibility alongside UI
  - Keep JSON responses for `POST /analyze` and `POST /curate-topic` when `Content-Type: application/json` is used
  - Render server-side templates for non-JSON form submissions
  - _Requirements: 1.1, 2.2, 4.1_

- [x] 15. Front-end UX enhancements
  - Loading indicators and inline error messages for both forms
  - Client-side sorting, filtering, search, and pagination for curated results
  - _Requirements: 4.3_

- [x] 16. Bugfix: timezone-safe recency scoring
  - Normalize naive/aware datetimes to UTC in `compute_recency_score`
  - Prevents offset-naive/aware subtraction errors
  - _Requirements: 3.5_

- [x] 17. Upgrade relevance scoring to embeddings
  - Integrate SentenceTransformers (MiniLM) for semantic similarity
  - Add lazy model initialization and config flag to fallback to TF‑IDF
  - Write unit tests with mocked embeddings to verify ranking improvements
  - _Requirements: 3.4, 3.6_

- [ ] 18. Duplicate detection and diversity controls
  - URL canonicalization (strip UTM, normalize hosts) before parsing
  - Near‑duplicate collapse using simhash or embedding threshold
  - Domain diversity cap in final ranking (e.g., max 2 per domain)
  - _Requirements: 2.3, 7.1_

- [ ] 19. Topic‑aware recency calibration
  - Add per‑topic half‑life config (e.g., finance=1d, tech=3d, research=14d)
  - Pass calibrated half‑life into `compute_recency_score`
  - _Requirements: 3.5, 5.2_

- [ ] 20. Parser resilience improvements
  - Add `readability-lxml` fallback when newspaper3k fails
  - Harden retry/backoff and user‑agent rotation
  - Expand error messages for better UX and logs
  - _Requirements: 1.1, 6.2, 6.4_

- [ ] 21. Source reputation and credibility signals
  - Maintain domain reputation table and optional author reputation
  - Add reputation weight into composite score via `ScoringConfig`
  - _Requirements: 3.6_

- [ ] 22. Topic coherence and coverage
  - Keyword coverage ratio and keyphrase extraction (e.g., YAKE)
  - Blend as an additional metric in composite score
  - _Requirements: 3.4, 3.6_

- [ ] 23. Summarization for display
  - Add summarization utility (extractive or lightweight abstractive)
  - Use when newspaper3k summary is missing/low quality
  - _Requirements: 1.3, 4.2_

- [ ] 24. Diversity‑constrained ranking
  - Cluster by embeddings and cap results per cluster/domain
  - Ensure varied perspectives in top‑N
  - _Requirements: 2.3, 7.1_

- [ ] 25. Caching layer
  - Redis cache for parsed articles and scorecards with TTLs
  - Wrapper utilities and cache keys by URL/hash
  - _Requirements: 7.1, 7.2_

- [ ] 26. Background jobs for batch processing
  - Celery + Redis worker for `batch_analyze`
  - Add status endpoints and progress polling in UI
  - _Requirements: 7.1, 7.2_

- [ ] 27. Persistence layer
  - Store `Article`, `ScoreCard`, `Entity` in Postgres via SQLAlchemy
  - Simple migrations and retention policy
  - _Requirements: 5.1, 5.2_

- [ ] 28. Feedback loop and learning‑to‑rank
  - Endpoints to capture clicks/saves/likes
  - Periodic job to tune `ScoringConfig` weights from feedback
  - _Requirements: 5.4, 7.3_

- [ ] 29. Observability and quality monitoring
  - Add Sentry for errors, Prometheus metrics for timings and success rates
  - Structured logs for parsing/scoring outcomes
  - _Requirements: 6.4, 7.3_

- [ ] 30. UI enhancements (professional editorial UX)
  - Refine article card typography and spacing; add faceted filters (domain/date/word count)
  - Actions: save to list, copy/share links
  - _Requirements: 4.1, 4.3, 4.4_

- [ ] 31. CI/CD and deployment hardening
  - Dockerfile and docker‑compose with Redis/Postgres profiles
  - GitHub Actions for tests and build; environment‑specific configs
  - _Requirements: 5.1, 5.2, 5.3_