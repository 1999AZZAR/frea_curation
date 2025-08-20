import unittest
from unittest.mock import patch, Mock
import os

from app import app as flask_app
from curator.core.models import ScoringConfig


class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        self.app = flask_app.test_client()

    def test_analyze_missing_url(self):
        resp = self.app.post('/analyze', json={'url': ''})
        self.assertEqual(resp.status_code, 400)

    @patch('curator.core.config.load_scoring_config')
    @patch('curator.core.validation.validate_url')
    @patch('curator.services._analyzer.analyze_article')
    def test_analyze_success_mock(self, mock_analyze, mock_validate, mock_load):
        # Setup mocks
        mock_validate.return_value = (True, None)
        mock_load.return_value = ScoringConfig()
        # Return a fake scorecard dict-like to bypass parsing
        class Dummy:
            overall_score=90
            readability_score=80
            ner_density_score=70
            sentiment_score=60
            tfidf_relevance_score=50
            recency_score=40
            article=type('A', (), {'url':'https://example.com','title':'t','author':'a','summary':'s'})()
        mock_analyze.return_value = Dummy()

        resp = self.app.post('/analyze', json={'url': 'https://example.com', 'query': 'ai'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['overall_score'], 90)

    @patch('curator.services.news_source.NewsSource')
    @patch('curator.services._analyzer.batch_analyze')
    def test_curate_topic_success(self, mock_batch, mock_source):
        mock_instance = Mock()
        mock_instance.get_article_urls.return_value = [
            'https://example.com/1', 'https://example.com/2'
        ]
        mock_source.return_value = mock_instance
        # Mock analyzed results with scores to verify sorting
        A = lambda u,t: type('A', (), {'url':u,'title':t,'author':'','summary':''})()
        r1 = type('R', (), {'overall_score':90,'readability_score':80,'ner_density_score':70,'sentiment_score':60,'tfidf_relevance_score':50,'recency_score':40,'article':A('https://example.com/1','a1')})()
        r2 = type('R', (), {'overall_score':95,'readability_score':80,'ner_density_score':70,'sentiment_score':60,'tfidf_relevance_score':50,'recency_score':40,'article':A('https://example.com/2','a2')})()
        mock_batch.return_value = [r1, r2]

        resp = self.app.post('/curate-topic', json={'topic': 'ai', 'max_articles': 2})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['count'], 2)
        # Ensure sorted descending by overall_score (r2 first)
        self.assertEqual(data['results'][0]['article']['title'], 'a2')

    def test_curate_topic_invalid(self):
        resp = self.app.post('/curate-topic', json={'topic': ''})
        self.assertEqual(resp.status_code, 400)


if __name__ == '__main__':
    unittest.main()


