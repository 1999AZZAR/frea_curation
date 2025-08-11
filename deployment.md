# Deployment Guide

This document summarizes prerequisites, configuration, and recommended steps to run and deploy AI Content Curator in development and production.

## 1) Prerequisites

- Python 3.10+
- pip / venv
- Node.js (optional, only for building Tailwind CSS locally)
- NewsAPI.org API key (required for topic curation)
- System packages (Linux) recommended for newspaper3k and lxml:
  - Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y build-essential libxml2-dev libxslt1-dev libjpeg-dev zlib1g-dev

Notes:
- spaCy NER and NLTK VADER are optional at runtime. If not installed, the app gracefully degrades (no entities, neutral sentiment baseline).
- Sentence-transformers (embeddings) are optional; enabled via env flag.

## 2) Environment variables

Required
- NEWS_API_KEY: your NewsAPI.org key

Security
- SECRET_KEY: Flask secret, set in production

Scoring weights (must sum to ~1.0)
- READABILITY_WEIGHT (default 0.2)
- NER_DENSITY_WEIGHT (default 0.2)
- SENTIMENT_WEIGHT (default 0.15)
- TFIDF_RELEVANCE_WEIGHT (default 0.25)
- RECENCY_WEIGHT (default 0.2)

Scoring params
- MIN_WORD_COUNT (default 300)
- MAX_ARTICLES_PER_TOPIC (default 20)

Relevance (optional embeddings)
- USE_EMBEDDINGS_RELEVANCE = 1|true|yes|on to enable
- EMBEDDINGS_MODEL_NAME (default all-MiniLM-L6-v2)

Diversity controls
- DIVERSIFY_RESULTS = 1|true|yes|on (default on)
- DOMAIN_CAP (default 2)
- DUP_SIM_THRESHOLD (default 0.97)

Flask
- FLASK_ENV = development|production (affects debug logging)

## 3) Local setup

Create venv and install dependencies
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional NLP resources
```
# spaCy English model for NER
python -m spacy download en_core_web_sm
# NLTK VADER lexicon
python -c "import nltk; nltk.download('vader_lexicon')"
```

Build Tailwind CSS (optional if already compiled)
```
npm install
npm run build:css
```

Run dev server
```
python app.py
# http://localhost:5000
```

## 4) Production run (Gunicorn)

Basic
```
# assuming venv activated and requirements installed
export FLASK_ENV=production
export SECRET_KEY=change-me
export NEWS_API_KEY=your-key
# optional: export USE_EMBEDDINGS_RELEVANCE=1

# run gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
```

Reverse proxy (Nginx) minimal example
```
server {
  listen 80;
  server_name your.domain;

  location /static/ {
    alias /srv/app/static/;
    access_log off;
    expires 7d;
  }

  location / {
    proxy_pass http://127.0.0.1:5000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```

## 5) Docker (example)

Dockerfile
```
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FLASK_ENV=production
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Build and run
```
docker build -t ai-content-curator .
docker run -e NEWS_API_KEY=your-key -e SECRET_KEY=change-me -p 5000:5000 ai-content-curator
```

## 6) Health and troubleshooting

- Basic health check: GET / should return the UI
- Common issues:
  - Newspaper3k failures: Verify network egress and domain accessibility; consider increasing timeouts; some sites block scraping
  - Missing NLP resources: install spaCy model and/or NLTK VADER lexicon for best quality
  - Embeddings model download: first run may download the model; ensure disk/network availability
  - Invalid weights: ensure weights sum to ~1.0 (auto-adjust applies only if not all weights overridden)

## 7) Operational tips

- Configure logs (stdout/stderr) and use process supervision (systemd, Docker, etc.)
- Set MIN_WORD_COUNT and MAX_ARTICLES_PER_TOPIC for workload control
- Tune DOMAIN_CAP and DUP_SIM_THRESHOLD to balance diversity and similarity suppression
- Consider adding caching (e.g., Redis) and background workers (Celery) for heavy topics (see tasks roadmap)

## 8) Security

- Do not commit secrets; use environment variables or secret managers
- Set a strong SECRET_KEY in production
- Terminate TLS at your proxy/load balancer

## 9) CI/CD (outline)

- Lint and test on each commit
- Build Docker image; push to registry
- Deploy with environment-specific configuration and secrets

---
This guide covers the essentials for local and production deployments. See README.md for project overview and usage; consult the tasks roadmap for planned enhancements (caching, persistence, background jobs).
