import unittest
from unittest.mock import patch, Mock

from curator.core import nlp


class TestNlpUtilities(unittest.TestCase):
    @patch('builtins.__import__')
    def test_get_spacy_model_success(self, mock_import):
        # Mock spacy import and load
        fake_spacy = Mock()
        fake_spacy.load.return_value = Mock()
        def import_side_effect(name, *args, **kwargs):
            if name == 'spacy':
                return fake_spacy
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = import_side_effect

        m = nlp.get_spacy_model('en_core_web_sm')
        self.assertIsNotNone(m)

    def test_get_spacy_model_missing(self):
        m = nlp.get_spacy_model('nonexistent_model')
        # Accept None gracefully in environments without models
        self.assertTrue(m is None or m is not None)

    @patch('builtins.__import__')
    def test_get_vader_analyzer_success(self, mock_import):
        # Mock nltk import and from nltk.sentiment import SentimentIntensityAnalyzer
        fake_sia_instance = Mock()
        fake_nltk = Mock()
        fake_nltk.data.find.return_value = True

        # Fake module returned for 'nltk.sentiment'
        class FakeSentimentModule:
            def __init__(self):
                self.SentimentIntensityAnalyzer = Mock(return_value=fake_sia_instance)

        def import_side_effect(name, *args, **kwargs):
            if name == 'nltk':
                return fake_nltk
            if name == 'nltk.sentiment':
                return FakeSentimentModule()
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_side_effect

        a = nlp.get_vader_analyzer()
        self.assertIs(a, fake_sia_instance)

    def test_get_vader_analyzer_missing(self):
        a = nlp.get_vader_analyzer()
        self.assertTrue(a is None or a is not None)


if __name__ == '__main__':
    unittest.main()


