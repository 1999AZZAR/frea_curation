# AI Content Curator

An AI-powered content curation web application built with Flask. It fetches articles, analyzes them across multiple quality dimensions, and ranks results with a configurable composite score. The UI is optimized for a clean, professional (Medium-like) reading experience.

## Features
- Manual article analysis (single URL) with detailed scorecard
- Topic-based curation (fetch from NewsAPI, parse, analyze, rank)
- Scoring engine components: readability, NER density, sentiment (neutrality), TF‑IDF relevance, recency
- Configurable weights and minimum word count
- Modern UI: light theme, responsive layout, loading/error states, sorting/filtering/search/pagination

## Project Structure

```
ai-content-curator/
├── app.py                 # Flask app+routes (JSON or server-rendered views)
├── config.py              # Scoring configuration loading
├── requirements.txt       # Python dependencies (unpinned)
├── README.md              # Project documentation
├── curator/               # Application package
│   ├── core/              # Models, validation, nlp helpers
│   └── services/          # Analyzer, parser (newspaper3k), news source (NewsAPI)
├── templates/             # Jinja2 templates (base, index, results, curation_results, errors)
├── static/                # Compiled assets
│   ├── css/
│   └── js/
├── assets/                # Tailwind input CSS (source)
├── package.json           # Tailwind/PostCSS build scripts
├── tailwind.config.js     # Tailwind configuration
└── postcss.config.js      # PostCSS configuration
```

## Prerequisites
- Python 3.10+
- Node.js (optional, for building Tailwind CSS locally)
- Environment variables:
  - `NEWS_API_KEY` (required for topic curation via NewsAPI)
  - Optional scoring settings (see `config.py`)

## Setup
1) Create and activate a virtual environment, install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Optional: install NLP resources
```bash
# spaCy English model (if you want NER enabled locally)
python -m spacy download en_core_web_sm

# NLTK VADER lexicon (for sentiment); skip if running in restricted envs
python -c "import nltk; nltk.download('vader_lexicon')"
```

3) Build the UI (Tailwind, optional at runtime if already built)
```bash
npm install
npm run build:css
```

## Running
```bash
source .venv/bin/activate
python app.py
```
App runs at `http://localhost:5000`.

## API Endpoints
- POST `/analyze`
  - Body: `{ "url": string, "query"?: string }`
  - Returns: scorecard JSON (if `Content-Type: application/json`), otherwise renders `results.html`.

- POST `/curate-topic`
  - Body: `{ "topic": string, "max_articles"?: number }`
  - Returns: ranked list JSON (if JSON request), otherwise renders `curation_results.html`.

## Using the UI
- Analyze: Enter an article URL (+ optional query), click Analyze → view breakdown and overall score.
- Curate: Enter a topic (+ optional max), click Curate → filter/sort/search/paginate ranked cards.

## Testing
```bash
source .venv/bin/activate
pytest -q
```

## Troubleshooting
- Newspaper3k parsing
  - Some sources may block scraping; retries and user-agent rotation are enabled.
  - If you encounter parsing issues, ensure network access and consider raising timeouts.
- NLP resource availability
  - The app degrades gracefully if spaCy model / VADER lexicon are unavailable (NER disabled, neutral sentiment),
    but installing them improves scoring quality (see Setup step 2).
- Tailwind CSS
  - If Node is unavailable, the app can still run using the last compiled CSS in `static/css/tailwind.css`.

## Roadmap (next)
- Embedding-based relevance scoring (SentenceTransformers) with TF‑IDF fallback
- Duplicate detection and domain diversity caps
- Topic-aware recency calibration
- Parser resilience (readability-lxml fallback)
- Source reputation and topic coverage metrics
- Caching, background jobs, persistence, feedback loop, observability

