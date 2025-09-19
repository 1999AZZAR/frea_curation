"""
Unit tests for summarization functionality.
"""

import pytest
from curator.core.summarization import (
    is_summary_low_quality,
    clean_text,
    split_into_sentences,
    calculate_sentence_scores,
    extractive_summarize,
    simple_abstractive_summarize,
    generate_summary,
    enhance_article_summary
)
from curator.core.models import Article


class TestSummaryQualityCheck:
    """Test summary quality assessment."""
    
    def test_empty_summary_is_low_quality(self):
        content = "This is some article content with multiple sentences."
        assert is_summary_low_quality("", content)
        assert is_summary_low_quality(None, content)
        assert is_summary_low_quality("   ", content)
    
    def test_too_short_summary_is_low_quality(self):
        content = "This is some article content with multiple sentences."
        short_summary = "Short"
        assert is_summary_low_quality(short_summary, content, min_length=50)
    
    def test_too_long_summary_is_low_quality(self):
        content = "Short content."
        long_summary = "This is a very long summary that is almost as long as the content itself."
        assert is_summary_low_quality(long_summary, content, max_ratio=0.5)
    
    def test_good_quality_summary(self):
        content = "This is a longer article with multiple sentences. It discusses various topics and provides detailed information about the subject matter."
        good_summary = "This article discusses various topics and provides detailed information."
        assert not is_summary_low_quality(good_summary, content)
    
    def test_low_quality_patterns(self):
        content = "Some article content here."
        assert is_summary_low_quality("Click here to read more", content)
        assert is_summary_low_quality("Subscribe to our newsletter", content)
        assert is_summary_low_quality("Advertisement content", content)
        assert is_summary_low_quality("...", content)


class TestTextCleaning:
    """Test text cleaning and preprocessing."""
    
    def test_clean_text_removes_extra_whitespace(self):
        text = "  This   has    extra   spaces  "
        cleaned = clean_text(text)
        assert cleaned == "This has extra spaces"
    
    def test_clean_text_removes_brackets(self):
        text = "This is text [with brackets] and more text."
        cleaned = clean_text(text)
        assert "[with brackets]" not in cleaned
    
    def test_clean_text_handles_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_split_into_sentences(self):
        text = "First sentence. Second sentence! Third sentence? Fourth fragment"
        sentences = split_into_sentences(text)
        assert len(sentences) >= 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
    
    def test_split_filters_short_fragments(self):
        text = "Good sentence. A. Another good sentence."
        sentences = split_into_sentences(text)
        # Should filter out "A" as it's too short
        assert all(len(s) > 10 for s in sentences)


class TestSentenceScoring:
    """Test sentence importance scoring."""
    
    def test_calculate_sentence_scores(self):
        content = "Technology is important. Artificial intelligence is the future. AI will change everything."
        sentences = split_into_sentences(content)
        scored = calculate_sentence_scores(sentences, content)
        
        assert len(scored) == len(sentences)
        assert all(isinstance(score, float) for _, score in scored)
        assert all(score >= 0 for _, score in scored)
    
    def test_empty_sentences_return_empty_scores(self):
        scored = calculate_sentence_scores([], "some content")
        assert scored == []


class TestExtractiveSummarization:
    """Test extractive summarization."""
    
    def test_extractive_summarize_basic(self):
        content = """
        Technology is rapidly advancing in many fields. Artificial intelligence represents 
        one of the most significant developments. Machine learning algorithms are becoming 
        more sophisticated every year. These advances will transform how we work and live.
        The implications are far-reaching and profound.
        """
        summary = extractive_summarize(content, max_sentences=2, target_length=150)
        
        assert len(summary) > 0
        assert len(summary) <= 150
        assert "." in summary  # Should be proper sentences
    
    def test_extractive_summarize_short_content(self):
        content = "This is very short content."
        summary = extractive_summarize(content, target_length=200)
        assert summary == content.strip()
    
    def test_extractive_summarize_empty_content(self):
        summary = extractive_summarize("", target_length=200)
        assert summary == ""
        
        summary = extractive_summarize(None, target_length=200)
        assert summary == ""


class TestAbstractiveSummarization:
    """Test simple abstractive summarization."""
    
    def test_simple_abstractive_summarize(self):
        content = """
        Apple Inc. announced new iPhone models today. The Technology company revealed 
        several innovative features. CEO Tim Cook presented the devices at a special event.
        The new phones include advanced camera systems and improved processors.
        """
        summary = simple_abstractive_summarize(content, target_length=150)
        
        assert len(summary) > 0
        assert len(summary) <= 150
        # Should mention key entities
        assert any(term in summary for term in ["Apple", "iPhone", "Technology"])
    
    def test_abstractive_handles_empty_content(self):
        summary = simple_abstractive_summarize("", target_length=200)
        assert summary == ""


class TestSummaryGeneration:
    """Test main summary generation function."""
    
    def test_generate_summary_keeps_good_existing(self):
        content = "This is a long article about technology and innovation in the modern world."
        existing = "This article discusses technology and innovation."
        
        result = generate_summary(content, existing, method="extractive")
        assert result == existing  # Should keep good existing summary
    
    def test_generate_summary_replaces_bad_existing(self):
        content = """
        This is a comprehensive article about artificial intelligence and machine learning.
        The field has seen rapid advancement in recent years. Deep learning algorithms
        have revolutionized many applications. The future looks very promising.
        """
        bad_existing = "..."  # Low quality
        
        result = generate_summary(content, bad_existing, method="extractive")
        assert result != bad_existing
        assert len(result) > 20
    
    def test_generate_summary_extractive_method(self):
        content = """
        Climate change is a pressing global issue. Scientists worldwide are studying
        its effects on ecosystems. Renewable energy sources offer potential solutions.
        Government policies play a crucial role in addressing this challenge.
        """
        
        result = generate_summary(content, method="extractive", target_length=100)
        assert len(result) > 0
        assert len(result) <= 100
    
    def test_generate_summary_abstractive_method(self):
        content = """
        Microsoft Corporation released quarterly earnings today. The Software giant
        reported strong revenue growth. Cloud services drove much of the increase.
        CEO Satya Nadella expressed optimism about future prospects.
        """
        
        result = generate_summary(content, method="abstractive", target_length=100)
        assert len(result) > 0
        assert len(result) <= 100
    
    def test_generate_summary_fallback_behavior(self):
        # Test with empty content
        result = generate_summary("", "fallback summary")
        assert result == "fallback summary"
        
        # Test with no content or existing summary
        result = generate_summary("", "")
        assert result == ""


class TestArticleEnhancement:
    """Test article summary enhancement."""
    
    def test_enhance_article_summary(self):
        article = Article(
            url="https://example.com/test",
            title="Test Article",
            content="This is a test article with some content about technology and innovation.",
            summary="..."  # Low quality summary
        )
        
        enhance_article_summary(article, method="extractive", target_length=100)
        
        # Summary should be enhanced
        assert article.summary != "..."
        assert len(article.summary) > 10
    
    def test_enhance_article_summary_keeps_good_summary(self):
        good_summary = "This article discusses technology and innovation in detail."
        article = Article(
            url="https://example.com/test",
            title="Test Article", 
            content="This is a comprehensive article about technology and innovation in the modern world.",
            summary=good_summary
        )
        
        enhance_article_summary(article, method="extractive")
        
        # Should keep the good summary
        assert article.summary == good_summary
    
    def test_enhance_article_summary_handles_missing_attributes(self):
        # Test with object missing required attributes
        class MockArticle:
            pass
        
        mock_article = MockArticle()
        
        # Should not raise exception
        enhance_article_summary(mock_article)
    
    def test_enhance_article_summary_handles_errors_gracefully(self):
        article = Article(
            url="https://example.com/test",
            title="Test Article",
            content=None,  # This might cause issues
            summary="original"
        )
        
        # Should not raise exception and preserve original summary
        enhance_article_summary(article)
        assert hasattr(article, 'summary')


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_newspaper_summary_enhancement(self):
        """Test enhancing a typical newspaper3k extracted summary."""
        # Simulate newspaper3k output with poor summary
        content = """
        The Federal Reserve announced a new interest rate policy today, marking a significant
        shift in monetary policy. The decision comes after months of economic analysis and
        consultation with financial experts. Market analysts expect this change to impact
        various sectors of the economy. The new policy aims to balance inflation control
        with economic growth objectives.
        """
        
        newspaper_summary = "The Federal Reserve announced..."  # Incomplete
        
        enhanced = generate_summary(content, newspaper_summary, method="extractive", target_length=150)
        
        assert len(enhanced) > len(newspaper_summary)
        assert "Federal Reserve" in enhanced
        assert len(enhanced) <= 150
    
    def test_missing_summary_generation(self):
        """Test generating summary when newspaper3k provides none."""
        content = """
        Breakthrough research in quantum computing has been published by scientists at MIT.
        The new algorithm demonstrates significant improvements in processing speed.
        This development could revolutionize cryptography and data security.
        Industry experts are calling it a major milestone in the field.
        """
        
        summary = generate_summary(content, existing_summary="", method="extractive")
        
        assert len(summary) > 50
        assert any(term in summary.lower() for term in ["quantum", "research", "mit", "algorithm"])