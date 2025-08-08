"""
AI Content Curator Flask Application
Main application entry point with routing and configuration
"""

import os
from flask import Flask, render_template, request, jsonify
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
    def analyze_article():
        """Manual article analysis endpoint"""
        # TODO: Implement article analysis logic
        return jsonify({'message': 'Article analysis endpoint - to be implemented'})
    
    @app.route('/curate-topic', methods=['POST'])
    def curate_topic():
        """Topic-based curation endpoint"""
        # TODO: Implement topic curation logic
        return jsonify({'message': 'Topic curation endpoint - to be implemented'})
    
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