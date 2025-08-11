# AI Content Curator

An AI-powered content curation web application built with Flask that automatically sources, analyzes, and ranks high-quality online articles using a multifactorial scoring algorithm.

## Project Structure

```
ai-content-curator/
├── app.py                 # Main Flask application entry point
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── curator/              # Application package (organized modules)
│   ├── core/             # Models, validation, nlp
│   ├── services/         # Analyzer, parser, news source
│   └── web/              # Web layer (future use)
├── templates/            # Jinja2 templates
├── static/               # Static assets
│   ├── css/
│   └── js/
├── assets/               # Tailwind input CSS
├── package.json          # Tailwind build scripts
├── tailwind.config.js    # Tailwind configuration
└── postcss.config.js     # PostCSS configuration
```

## Setup Instructions

1. Python environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. UI build tooling
   ```bash
   # If Node is available
   npm install
   # Build once
   npm run build:css
   # Or during development
   npm run dev:css
   ```

3. Run the development server
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`.

## Notes
- The UI now uses a compiled Tailwind CSS (`static/css/tailwind.css`) for a clean, professional look inspired by Medium.
- If Node is not available, you can temporarily switch back to the Tailwind CDN by re-adding it in `templates/base.html`.
