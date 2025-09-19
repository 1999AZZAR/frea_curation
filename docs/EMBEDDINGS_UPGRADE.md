# Embeddings Relevance Scoring Upgrade

This document describes the upgrade from TF-IDF to semantic embeddings for relevance scoring in the AI Content Curator.

## Overview

The relevance scoring system has been enhanced to support semantic similarity using SentenceTransformers embeddings, while maintaining backward compatibility with TF-IDF scoring.

## Features

### 1. Semantic Similarity with SentenceTransformers

- Uses the `all-MiniLM-L6-v2` model by default for fast, accurate embeddings
- Computes cosine similarity between article content and query embeddings
- Provides better semantic understanding compared to keyword-based TF-IDF

### 2. Configurable Fallback System

- Environment variable `USE_EMBEDDINGS_RELEVANCE` controls which method to use
- Automatic fallback to TF-IDF if embeddings model is unavailable
- Graceful error handling for model loading failures

### 3. Lazy Model Initialization

- Models are loaded only when first needed
- Global caching prevents repeated model loading
- Minimal memory footprint when embeddings are disabled

## Configuration

### Environment Variables

```bash
# Enable/disable embeddings (default: false)
USE_EMBEDDINGS_RELEVANCE=true|false|1|0|yes|no|on|off

# Configure model name (default: all-MiniLM-L6-v2)
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
```

### Example Configuration

```bash
# Enable embeddings with default model
export USE_EMBEDDINGS_RELEVANCE=true

# Use custom model
export USE_EMBEDDINGS_RELEVANCE=true
export EMBEDDINGS_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
```

## API Usage

### Direct Function Calls

```python
from curator.services.analyzer import (
    compute_embeddings_relevance_score,
    compute_tfidf_relevance_score,
    compute_relevance_score
)

# Use embeddings directly
embeddings_score = compute_embeddings_relevance_score(article, query)

# Use TF-IDF directly  
tfidf_score = compute_tfidf_relevance_score(article, query)

# Use configured method (embeddings or TF-IDF based on environment)
relevance_score = compute_relevance_score(article, query)
```

### Automatic Integration

The existing `analyze_article()` and `batch_analyze()` functions automatically use the configured relevance scoring method:

```python
# This will use embeddings if USE_EMBEDDINGS_RELEVANCE=true
scorecard = analyze_article(url, query="machine learning")
```

## Performance Considerations

### Model Loading

- First call loads the model (may take 1-2 seconds)
- Subsequent calls use cached model (fast)
- Model loading happens lazily when first needed

### Memory Usage

- `all-MiniLM-L6-v2`: ~90MB RAM
- Larger models like `all-mpnet-base-v2`: ~400MB RAM
- No memory usage when embeddings disabled

### Speed Comparison

- **TF-IDF**: ~1-5ms per article
- **Embeddings**: ~10-50ms per article (after model loading)
- **Batch processing**: Embeddings become more efficient with larger batches

## Quality Improvements

### Semantic Understanding

Embeddings provide better semantic matching:

```python
# Query: "artificial intelligence"
# Article 1: "Machine learning algorithms and neural networks..."
# Article 2: "Stock market analysis and trading strategies..."

# TF-IDF: May score both low due to lack of exact keyword matches
# Embeddings: Will score Article 1 much higher due to semantic similarity
```

### Better Ranking

- Captures synonyms and related concepts
- Understands context and meaning
- Reduces false positives from keyword stuffing
- Improves relevance for complex queries

## Testing

### Unit Tests

Comprehensive test suite covers:

- Mock embeddings with controlled similarity scores
- Environment variable configuration
- Fallback behavior when models unavailable
- Error handling for encoding failures
- Ranking improvement verification

### Running Tests

```bash
# Run all analyzer tests
python -m pytest tests/test_analyzer.py -v

# Run only embeddings tests
python -m pytest tests/test_analyzer.py::TestEmbeddingsRankingImprovement -v

# Run specific embeddings test
python -m pytest tests/test_analyzer.py::TestScoringFunctions::test_embeddings_relevance_score_with_mock_model -v
```

## Deployment

### Requirements

Add to `requirements.txt`:
```
sentence-transformers
```

### Installation

```bash
pip install sentence-transformers
```

### Docker Considerations

For containerized deployments, consider:

```dockerfile
# Pre-download models during build to avoid runtime downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Migration Guide

### Existing Deployments

1. **Install Dependencies**:
   ```bash
   pip install sentence-transformers
   ```

2. **Enable Embeddings** (optional):
   ```bash
   export USE_EMBEDDINGS_RELEVANCE=true
   ```

3. **Test Configuration**:
   ```bash
   # Verify fallback works
   python -c "from curator.services.analyzer import compute_relevance_score; print('OK')"
   ```

### Backward Compatibility

- All existing APIs remain unchanged
- Default behavior uses TF-IDF (no breaking changes)
- ScoreCard field names remain the same (`tfidf_relevance_score`)
- JSON API responses unchanged

## Troubleshooting

### Common Issues

1. **Model Download Failures**:
   - Check internet connectivity
   - Verify model name is correct
   - Check disk space for model cache

2. **Memory Issues**:
   - Use smaller models like `all-MiniLM-L6-v2`
   - Disable embeddings: `USE_EMBEDDINGS_RELEVANCE=false`
   - Monitor memory usage in production

3. **Performance Issues**:
   - Model loading is one-time cost
   - Consider batch processing for multiple articles
   - Use TF-IDF for latency-critical applications

### Debug Commands

```bash
# Test embeddings availability
python -c "from curator.core.nlp import get_sentence_transformer; print(get_sentence_transformer() is not None)"

# Test configuration
python -c "import os; print('Embeddings enabled:', os.environ.get('USE_EMBEDDINGS_RELEVANCE', 'false'))"
```

## Future Enhancements

- Support for multilingual models
- Custom fine-tuned models for domain-specific content
- Caching of computed embeddings
- Batch embedding computation optimization
- Integration with vector databases for large-scale similarity search