# Duplicate Detection and Diversity Controls

This document describes the duplicate detection and diversity control features implemented in the AI Content Curator.

## Overview

The system implements comprehensive duplicate detection and diversity controls to improve the quality and variety of curated content. This includes:

1. **URL Canonicalization** - Normalizes URLs to reduce false duplicates
2. **Near-Duplicate Detection** - Uses embeddings, simhash, or basic text comparison
3. **Domain Diversity Controls** - Limits articles from the same domain

## URL Canonicalization

### Features

The URL canonicalization process normalizes URLs to reduce duplicates caused by:

- **Tracking Parameters**: Removes UTM, Google Analytics, Facebook, and other tracking parameters
- **Case Normalization**: Converts scheme and host to lowercase
- **www Prefix**: Removes www. prefix from domains
- **Mobile Variants**: Normalizes m. and mobile. subdomains
- **AMP URLs**: Normalizes Accelerated Mobile Pages URLs
- **Trailing Slashes**: Removes trailing slashes (except root)
- **Fragment Removal**: Strips URL fragments (#section)
- **Parameter Sorting**: Sorts remaining query parameters for consistency

### Example

```python
from curator.core.utils import canonicalize_url

# Input URL with tracking parameters
url = "https://WWW.Example.com/article?utm_source=twitter&gclid=123&id=456"

# Canonicalized URL
canonical = canonicalize_url(url)
# Result: "https://example.com/article?id=456"
```

### Removed Parameters

The following tracking and session parameters are automatically removed:

- **UTM Parameters**: `utm_source`, `utm_medium`, `utm_campaign`, `utm_term`, `utm_content`
- **Google Analytics**: `gclid`, `gclsrc`, `dclid`, `_ga`, `_gid`
- **Facebook**: `fbclid`, `fb_action_ids`, `fb_action_types`, `fb_ref`, `fb_source`
- **Other Tracking**: `mc_eid`, `mc_cid`, `ref`, `referrer`
- **Session/Cache**: `v`, `version`, `cache`, `timestamp`, `t`, `_`
- **Social Media**: `share`, `shared`, `via`, `source`
- **Newsletter/Email**: `newsletter`, `email`, `subscriber_id`
- **Affiliate**: `affiliate`, `aff`, `partner`, `promo`

## Near-Duplicate Detection

The system uses a three-tier approach for detecting near-duplicate content:

### 1. Embeddings (Preferred)

Uses sentence-transformers to generate semantic embeddings and compute cosine similarity.

- **Model**: Configurable via `EMBEDDINGS_MODEL_NAME` (default: "all-MiniLM-L6-v2")
- **Threshold**: Configurable via `DUP_SIM_THRESHOLD` (default: 0.97)
- **Accuracy**: Highest - detects semantic similarity even with different wording

### 2. Simhash (Fallback)

Uses simhash algorithm for fast approximate duplicate detection.

- **Library**: `simhash` package
- **Threshold**: Converts similarity threshold to Hamming distance
- **Accuracy**: Medium - good for detecting near-exact duplicates

### 3. Basic Text Comparison (Last Resort)

Falls back to exact title and canonicalized URL comparison.

- **Method**: Exact string matching (case-insensitive for titles)
- **Accuracy**: Low - only catches exact duplicates

## Domain Diversity Controls

### Domain Cap

Limits the number of articles from the same domain in results.

- **Configuration**: `DOMAIN_CAP` environment variable (default: 2)
- **Behavior**: Keeps highest-scoring articles from each domain
- **Domain Extraction**: Normalizes domains (removes www, converts to lowercase)

### Example

```python
from curator.services._analyzer import batch_analyze

# Configure domain cap
import os
os.environ['DOMAIN_CAP'] = '3'  # Max 3 articles per domain

# Analyze articles with diversity controls
results = batch_analyze(urls, apply_diversity=True)
```

## Configuration

### Environment Variables

- `DIVERSIFY_RESULTS`: Enable/disable diversity filtering (default: "1")
- `DOMAIN_CAP`: Maximum articles per domain (default: 2)
- `DUP_SIM_THRESHOLD`: Similarity threshold for duplicates (default: 0.97)
- `EMBEDDINGS_MODEL_NAME`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `USE_EMBEDDINGS_RELEVANCE`: Enable embeddings for relevance scoring

### Example Configuration

```bash
# Enable diversity controls
DIVERSIFY_RESULTS=1

# Allow up to 3 articles per domain
DOMAIN_CAP=3

# Set high similarity threshold (0.95 = 95% similar)
DUP_SIM_THRESHOLD=0.95

# Use specific embeddings model
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
```

## Integration

### Batch Analysis

Diversity controls are automatically applied during batch analysis:

```python
from curator.services.analyzer import batch_analyze

# Automatic diversity filtering (if DIVERSIFY_RESULTS=1)
results = batch_analyze(urls, query="technology")

# Explicit control
results = batch_analyze(urls, query="technology", apply_diversity=True)
```

### News Source Integration

URL canonicalization is applied when fetching articles:

```python
from curator.services.news_source import NewsSource

news_source = NewsSource()
urls = news_source.get_article_urls("technology")
# URLs are automatically canonicalized and deduplicated
```

## Performance Considerations

### Embeddings

- **Memory**: Models require ~100MB RAM
- **Speed**: ~10-50ms per article depending on content length
- **Accuracy**: Highest quality duplicate detection

### Simhash

- **Memory**: Minimal overhead
- **Speed**: Very fast (~1ms per article)
- **Accuracy**: Good for near-exact duplicates

### Basic Comparison

- **Memory**: Minimal overhead
- **Speed**: Fastest (~0.1ms per article)
- **Accuracy**: Only exact matches

## Error Handling

The system implements graceful degradation:

1. If embeddings fail → fallback to simhash
2. If simhash fails → fallback to basic comparison
3. If diversity filtering fails → return unfiltered results
4. All errors are logged but don't stop processing

## Testing

Comprehensive tests are provided in `tests/test_duplicate_detection.py`:

- URL canonicalization edge cases
- Domain extraction scenarios
- Diversity filtering with various configurations
- Error handling and fallback behavior
- Performance with different similarity thresholds

Run tests with:

```bash
python -m pytest tests/test_duplicate_detection.py -v
```

## Requirements

The following packages are required:

```
sentence-transformers  # For embeddings-based duplicate detection
simhash               # For simhash-based duplicate detection
```

These are automatically included in `requirements.txt`.