# Project Structure

This document outlines the reorganized structure of the AI Content Curator project.

## Directory Organization

```
CURATION/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Main project documentation
├── models.py             # Legacy models (compatibility)
├── .env                  # Environment variables
├── .env.template         # Environment template
├── .gitignore           # Git ignore rules
├── package.json         # Node.js dependencies
├── package-lock.json    # Node.js lock file
├── postcss.config.js    # PostCSS configuration
├── tailwind.config.js   # Tailwind CSS configuration
│
├── assets/              # Static assets and examples
│   ├── tailwind.css
│   └── styles_example.css
│
├── config/              # Configuration files
│   └── config.py        # Application configuration
│
├── curator/             # Main package
│   ├── core/            # Core functionality
│   ├── services/        # Service layer
│   ├── data/            # Data utilities
│   └── web/             # Web-specific code
│
├── data/                # Database and data files
│   └── curator.db       # SQLite database
│
├── docs/                # Documentation
│   ├── BACKGROUND_JOBS.md
│   ├── CACHING.md
│   ├── DUPLICATE_DETECTION.md
│   ├── EMBEDDINGS_UPGRADE.md
│   ├── deployment.md
│   └── STRUCTURE.md     # This file
│
├── examples/            # Example code and usage
│
├── scripts/             # Utility scripts
│   ├── celery_worker.py
│   ├── debug_coverage.py
│   ├── manage_db.py
│   ├── start_workers.sh
│   └── verify_parser_resilience.py
│
├── static/              # Web static files
│   ├── css/
│   └── js/
│
├── templates/           # HTML templates
│
└── tests/               # Test files
    ├── test_*.py        # Unit tests
    └── README_INTEGRATION_TESTS.md
```

## Key Changes Made

### 1. Root Level Cleanup
- Moved scattered test files to `tests/` directory
- Moved utility scripts to `scripts/` directory
- Moved database files to `data/` directory
- Organized documentation in `docs/` directory

### 2. New Directory Structure
- **config/**: Configuration files
- **data/**: Database and data files
- **docs/**: All documentation files
- **scripts/**: Utility and management scripts

### 3. Updated Paths
- Database path updated to `data/curator.db`
- Script imports updated to work from new locations
- Configuration files properly organized

## Benefits

1. **Professional Structure**: Clear separation of concerns
2. **Better Organization**: Related files grouped together
3. **Cleaner Root**: Less clutter at the project root
4. **Maintainability**: Easier to find and manage files
5. **Standards Compliance**: Follows Python project conventions

## Usage Notes

- Main application still runs from `app.py` at root level
- Scripts in `scripts/` directory have updated import paths
- Database location updated in configuration
- Documentation centralized in `docs/` directory
