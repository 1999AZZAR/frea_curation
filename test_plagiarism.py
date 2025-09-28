#!/usr/bin/env python3
"""
Test script for plagiarism detection functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from curator.services.plagiarism_detector import PlagiarismDetector

def test_plagiarism_detection():
    """Test the plagiarism detection functionality"""
    
    # Sample content to test
    test_content = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. The term "artificial intelligence" 
    is often used to describe machines that mimic "cognitive" functions that humans 
    associate with the human mind, such as "learning" and "problem solving".
    
    As machines become increasingly capable, tasks considered to require "intelligence" 
    are often removed from the definition of AI, a phenomenon known as the AI effect. 
    A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, 
    optical character recognition is frequently excluded from things considered to be AI, 
    having become a routine technology.
    """
    
    print("🧪 Testing Plagiarism Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = PlagiarismDetector()
    
    # Test options
    options = {
        'deep_search': True,
        'citation_check': False,
        'paraphrase_check': True
    }
    
    print(f"📝 Testing content: {len(test_content.split())} words")
    print(f"🔍 Options: {options}")
    print()
    
    try:
        # Perform detection
        print("⏳ Running plagiarism detection...")
        result = detector.detect_plagiarism(test_content, options)
        
        print("✅ Detection completed!")
        print()
        print("📊 RESULTS:")
        print("-" * 30)
        print(f"Originality Score: {result.originality_score}%")
        print(f"Total Words: {result.total_words}")
        print(f"Sources Analyzed: {result.analyzed_sources}")
        print(f"Detection Method: {result.detection_method}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print()
        
        if result.similarity_sources:
            print("🔍 SIMILARITY SOURCES:")
            print("-" * 30)
            for i, source in enumerate(result.similarity_sources, 1):
                print(f"{i}. {source.title}")
                print(f"   URL: {source.url}")
                print(f"   Similarity: {source.similarity_percentage}%")
                print(f"   Type: {source.source_type}")
                print(f"   Snippet: {source.matched_text[:100]}...")
                print()
        else:
            print("🎉 No similarity sources found!")
            print()
        
        if result.recommendations:
            print("💡 RECOMMENDATIONS:")
            print("-" * 30)
            for i, rec in enumerate(result.recommendations, 1):
                print(f"{i}. {rec}")
        
        print()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plagiarism_detection()
