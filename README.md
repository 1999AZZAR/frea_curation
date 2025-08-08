# AI Content Curator

An AI-powered content curation web application built with Flask that automatically sources, analyzes, and ranks high-quality online articles using a multifactorial scoring algorithm.

## Project Structure

```
ai-content-curator/
├── app.py                 # Main Flask application entry point
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.template         # Environment variables template
├── README.md             # Project documentation
├── templates/            # Jinja2 templates
│   └── .gitkeep
├── static/               # Static assets
│   ├── css/
│   │   └── .gitkeep
│   └── js/
│       └── .gitkeep
└── .kiro/               # Kiro specifications
    └── specs/
        └── ai-content-curator/
            ├── requirements.md
            ├── design.md
            └── tasks.md
```

## Setup Instructions

1. **Clone the repository and navigate to the project directory**

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.template .env
   # Edit .env file with your actual configuration values
   ```

5. **Run the development server:**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## Environment Variables

Copy `.env.template` to `.env` and configure the following variables:

- `SECRET_KEY`: Flask secret key for session management
- `NEWS_API_KEY`: Your NewsAPI.org API key
- `MAX_ARTICLES_PER_TOPIC`: Maximum articles to fetch per topic (default: 20)
- `MIN_WORD_COUNT`: Minimum word count for article quality (default: 300)
- Scoring weights for the composite algorithm (should sum to 1.0)

## Development Status

This project is currently in development. The basic Flask application structure has been set up with:

- ✅ Project directory structure
- ✅ Core dependencies configuration
- ✅ Basic Flask application with routing stubs
- ✅ Environment configuration management
- ⏳ Article parsing and analysis (pending)
- ⏳ NewsAPI integration (pending)
- ⏳ NLP processing components (pending)
- ⏳ User interface templates (pending)

## Next Steps

Refer to `.kiro/specs/ai-content-curator/tasks.md` for the complete implementation plan.