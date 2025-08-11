"""
AI Content Curator Flask Application
Main application entry point with routing and configuration
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_app():
    """Application factory pattern for Flask app creation"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['DEBUG'] = os.environ.get('FLASK_ENV') == 'development'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Basic routes
    @app.route('/')
    def index():
        """Homepage with input forms for manual analysis and topic curation"""
        return render_template('index.html')
    
    @app.route('/analyze', methods=['POST'])
    def analyze_article_route():
        """Manual article analysis endpoint"""
        from curator.services.analyzer import analyze_article as analyze_fn
        from config import load_scoring_config
        from curator.core.validation import validate_url

        data = request.get_json(silent=True) or request.form
        url = (data.get('url') if data else None) or ''
        query = (data.get('query') if data else None) or ''

        is_valid, error = validate_url(url)
        if not is_valid:
            # If request is JSON, return JSON error; else render 400
            if request.is_json:
                return jsonify({'error': error or 'Invalid URL'}), 400
            return render_template('errors/400.html', message=error or 'Invalid URL'), 400

        try:
            config = load_scoring_config()
        except Exception as e:
            if request.is_json:
                return jsonify({'error': f'Configuration error: {str(e)}'}), 500
            return render_template('errors/500.html', message=f'Configuration error: {str(e)}'), 500

        try:
            # Optional NLP components
            from curator.core.nlp import get_spacy_model, get_vader_analyzer
            nlp_model = get_spacy_model()
            vader = get_vader_analyzer()

            scorecard = analyze_fn(url=url, query=query, config=config, nlp=nlp_model, vader_analyzer=vader)
            result = {
                'overall_score': scorecard.overall_score,
                'readability_score': scorecard.readability_score,
                'ner_density_score': scorecard.ner_density_score,
                'sentiment_score': scorecard.sentiment_score,
                'tfidf_relevance_score': scorecard.tfidf_relevance_score,
                'recency_score': scorecard.recency_score,
                'article': {
                    'url': scorecard.article.url,
                    'title': scorecard.article.title,
                    'author': scorecard.article.author,
                    'summary': scorecard.article.summary,
                }
            }
            if request.is_json:
                return jsonify(result)
            # Render server view
            return render_template('results.html', card=scorecard)
        except Exception as e:
            if request.is_json:
                return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
            return render_template('errors/500.html', message=f'Analysis failed: {str(e)}'), 500
    
    @app.route('/curate-topic', methods=['POST'])
    def curate_topic():
        """Topic-based curation endpoint"""
        from curator.services.analyzer import batch_analyze
        from config import load_scoring_config
        from curator.core.validation import validate_topic_keywords
        from curator.services.news_source import NewsSource

        data = request.get_json(silent=True) or request.form
        topic = (data.get('topic') if data else None) or ''
        max_articles = data.get('max_articles')
        try:
            max_articles = int(max_articles) if max_articles is not None else None
        except Exception:
            if request.is_json:
                return jsonify({'error': 'max_articles must be an integer'}), 400
            return render_template('errors/400.html', message='max_articles must be an integer'), 400

        is_valid, error = validate_topic_keywords(topic)
        if not is_valid:
            if request.is_json:
                return jsonify({'error': error or 'Invalid topic'}), 400
            return render_template('errors/400.html', message=error or 'Invalid topic'), 400

        try:
            config = load_scoring_config()
        except Exception as e:
            if request.is_json:
                return jsonify({'error': f'Configuration error: {str(e)}'}), 500
            return render_template('errors/500.html', message=f'Configuration error: {str(e)}'), 500

        try:
            # Fetch URLs
            api_key = os.environ.get('NEWS_API_KEY', 'test-api-key')
            source = NewsSource(api_key=api_key)
            urls = source.get_article_urls(topic, max_articles=max_articles or config.max_articles_per_topic)

            # Analyze
            from curator.core.nlp import get_spacy_model, get_vader_analyzer
            nlp_model = get_spacy_model()
            vader = get_vader_analyzer()
            results = batch_analyze(urls, query=topic, config=config, nlp=nlp_model, vader_analyzer=vader)
            # Sort by overall_score descending
            results.sort(key=lambda r: r.overall_score, reverse=True)

            if request.is_json:
                payload = [
                    {
                        'overall_score': r.overall_score,
                        'readability_score': r.readability_score,
                        'ner_density_score': r.ner_density_score,
                        'sentiment_score': r.sentiment_score,
                        'tfidf_relevance_score': r.tfidf_relevance_score,
                        'recency_score': r.recency_score,
                        'article': {
                            'url': r.article.url,
                            'title': r.article.title,
                            'author': r.article.author,
                            'summary': r.article.summary,
                        }
                    }
                    for r in results
                ]
                return jsonify({'count': len(payload), 'results': payload})
            # Render server-side list
            return render_template('curation_results.html', topic=topic, results=results)
        except Exception as e:
            if request.is_json:
                return jsonify({'error': f'Curation failed: {str(e)}'}), 500
            return render_template('errors/500.html', message=f'Curation failed: {str(e)}'), 500
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors"""
        return render_template('errors/400.html', 
                             message="Invalid input provided"), 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors"""
        return render_template('errors/404.html', 
                             message="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server errors"""
        return render_template('errors/500.html', 
                             message="An unexpected error occurred"), 500
    
    return app

# Create the Flask application instance
app = create_app()

if __name__ == '__main__':
    # Development server configuration
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=debug_mode
    )