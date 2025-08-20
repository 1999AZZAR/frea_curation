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

    @app.route('/compare', methods=['GET'])
    def compare_page():
        """Similarity comparison page."""
        return render_template('compare.html')

    @app.route('/compare', methods=['POST'])
    def compare_api():
        """Compute similarity between two URLs or raw texts."""
        from curator.services.parser import parse_article
        from curator.core.validation import validate_url
        from curator.services._analyzer import compute_tfidf_relevance_score, compute_embeddings_relevance_score
        from curator.core.models import Article
        import os

        data = request.get_json(silent=True) or request.form
        a_url = (data.get('a_url') or '').strip()
        b_url = (data.get('b_url') or '').strip()
        a_text = (data.get('a_text') or '').strip()
        b_text = (data.get('b_text') or '').strip()
        use_embeddings = str(data.get('use_embeddings', '')).strip().lower() in {'1','true','yes','on'}

        # Prepare articles
        try:
            if a_url:
                ok, err = validate_url(a_url)
                if not ok:
                    return jsonify({'error': f'Invalid A URL: {err}'}), 400
                a_article = parse_article(a_url)
            else:
                a_article = Article(url='about:blank', title='Input A', content=a_text)

            if b_url:
                ok, err = validate_url(b_url)
                if not ok:
                    return jsonify({'error': f'Invalid B URL: {err}'}), 400
                b_article = parse_article(b_url)
            else:
                b_article = Article(url='about:blank', title='Input B', content=b_text)
        except Exception as e:
            return jsonify({'error': f'Parsing failed: {str(e)}'}), 500

        # Compute similarities both ways using A content vs B title/content proxy query
        def sim_scores(a: Article, b: Article):
            query = (b.title or '').strip() or (b.content[:200] if b.content else '')
            tfidf = compute_tfidf_relevance_score(a, query)
            embed = 0.0
            if use_embeddings:
                embed = compute_embeddings_relevance_score(a, query)
            return tfidf, embed

        tfidf_ab, embed_ab = sim_scores(a_article, b_article)
        tfidf_ba, embed_ba = sim_scores(b_article, a_article)
        response = {
            'a': {'title': a_article.title, 'url': a_article.url},
            'b': {'title': b_article.title, 'url': b_article.url},
            'tfidf': {'a_to_b': tfidf_ab, 'b_to_a': tfidf_ba, 'avg': round((tfidf_ab + tfidf_ba)/2.0, 2)},
            'embeddings': {'a_to_b': embed_ab, 'b_to_a': embed_ba, 'avg': round((embed_ab + embed_ba)/2.0, 2)} if use_embeddings else None,
        }
        return jsonify(response)
    
    @app.route('/analyze', methods=['POST'])
    def analyze_article_route():
        """Manual article analysis endpoint"""
        from curator.services.analyzer import analyze_article as analyze_fn
        from curator.services.analyzer import (
            compute_embeddings_relevance_score,
            compute_tfidf_relevance_score,
        )
        # Prefer package config to keep modules under curator/
        try:
            from curator.core.config import load_scoring_config
        except Exception:
            from config import load_scoring_config
        from curator.core.validation import validate_url
        from urllib.parse import urlparse
        from curator.services.parser import get_article_word_count

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

            # Optional per-request override for embeddings usage
            use_embeddings_param = str((request.get_json(silent=True) or {}).get('use_embeddings', '')).strip().lower()
            original_env = os.environ.get('USE_EMBEDDINGS_RELEVANCE')
            if use_embeddings_param in {'1', 'true', 'yes', 'on'}:
                os.environ['USE_EMBEDDINGS_RELEVANCE'] = 'true'
            elif use_embeddings_param in {'0', 'false', 'no', 'off'}:
                os.environ['USE_EMBEDDINGS_RELEVANCE'] = 'false'

            scorecard = analyze_fn(url=url, query=query, config=config, nlp=nlp_model, vader_analyzer=vader)

            # Restore environment after call
            if original_env is None:
                os.environ.pop('USE_EMBEDDINGS_RELEVANCE', None)
            else:
                os.environ['USE_EMBEDDINGS_RELEVANCE'] = original_env

            # Derived stats (robust to partial objects)
            try:
                wc = get_article_word_count(scorecard.article)
            except Exception:
                wc = None
            try:
                ent_count = len(getattr(scorecard.article, 'entities', []) or [])
            except Exception:
                ent_count = 0
            try:
                netloc = urlparse(scorecard.article.url).netloc.lower()
                domain = netloc[4:] if netloc.startswith('www.') else netloc
            except Exception:
                domain = ''
            effective_query = (query or '').strip() or (scorecard.article.title or '').strip()
            # Determine relevance method used
            method = 'tfidf'
            try:
                emb_score = compute_embeddings_relevance_score(scorecard.article, effective_query)
                if os.environ.get('USE_EMBEDDINGS_RELEVANCE', '').strip().lower() in {'1','true','yes','on'} and emb_score > 0:
                    method = 'embeddings'
                else:
                    # Compare which score matches card more closely
                    tf_score = compute_tfidf_relevance_score(scorecard.article, effective_query)
                    method = 'embeddings' if abs(emb_score - scorecard.tfidf_relevance_score) < abs(tf_score - scorecard.tfidf_relevance_score) and emb_score > 0 else 'tfidf'
            except Exception:
                method = 'tfidf'

            result = {
                'overall_score': scorecard.overall_score,
                'readability_score': scorecard.readability_score,
                'ner_density_score': scorecard.ner_density_score,
                'sentiment_score': scorecard.sentiment_score,
                'tfidf_relevance_score': scorecard.tfidf_relevance_score,
                'recency_score': scorecard.recency_score,
                'article': {
                    'url': getattr(scorecard.article, 'url', ''),
                    'title': getattr(scorecard.article, 'title', ''),
                    'author': getattr(scorecard.article, 'author', ''),
                    'summary': getattr(scorecard.article, 'summary', ''),
                    'publish_date': scorecard.article.publish_date.isoformat() if getattr(scorecard.article, 'publish_date', None) else None,
                    'entities': [{'text': e.text, 'label': e.label} for e in (getattr(scorecard.article, 'entities', []) or [])],
                }
            }
            result['stats'] = {
                'word_count': wc,
                'entity_count': ent_count,
                'domain': domain,
                'relevance_method': method,
            }
            if request.is_json:
                return jsonify(result)
            # Render server view
            return render_template('results.html', card=scorecard, stats=result['stats'])
        except Exception as e:
            if request.is_json:
                return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
            return render_template('errors/500.html', message=f'Analysis failed: {str(e)}'), 500
    
    @app.route('/curate-topic', methods=['POST'])
    def curate_topic():
        """Topic-based curation endpoint"""
        from curator.services.analyzer import batch_analyze
        try:
            from curator.core.config import load_scoring_config
        except Exception:
            from config import load_scoring_config
        from curator.core.validation import validate_topic_keywords
        from curator.services.news_source import NewsSource
        from urllib.parse import urlparse
        from curator.services.parser import get_article_word_count

        data = request.get_json(silent=True) or request.form
        topic = (data.get('topic') if data else None) or ''
        max_articles = data.get('max_articles')
        apply_diversity = data.get('apply_diversity')
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
            results = batch_analyze(urls, query=topic, config=config, nlp=nlp_model, vader_analyzer=vader, apply_diversity=(str(apply_diversity).lower() in {'1','true','yes','on'} if apply_diversity is not None else None))
            # Sort by overall_score descending
            results.sort(key=lambda r: r.overall_score, reverse=True)
            
            if request.is_json:
                payload = []
                for r in results:
                    try:
                        netloc = urlparse(getattr(r.article, 'url', '')).netloc.lower()
                        domain = netloc[4:] if netloc.startswith('www.') else netloc
                    except Exception:
                        domain = ''
                    try:
                        wc = get_article_word_count(r.article)
                    except Exception:
                        wc = None
                    payload.append({
                        'overall_score': getattr(r, 'overall_score', 0),
                        'readability_score': getattr(r, 'readability_score', 0),
                        'ner_density_score': getattr(r, 'ner_density_score', 0),
                        'sentiment_score': getattr(r, 'sentiment_score', 0),
                        'tfidf_relevance_score': getattr(r, 'tfidf_relevance_score', 0),
                        'recency_score': getattr(r, 'recency_score', 0),
                        'domain': domain,
                        'word_count': wc,
                        'article': {
                            'url': getattr(r.article, 'url', ''),
                            'title': getattr(r.article, 'title', ''),
                            'author': getattr(r.article, 'author', ''),
                            'summary': getattr(r.article, 'summary', ''),
                            'publish_date': (getattr(r.article, 'publish_date', None).isoformat() if getattr(r.article, 'publish_date', None) else None),
                        }
                    })
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