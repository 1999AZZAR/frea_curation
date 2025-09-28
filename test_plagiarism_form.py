#!/usr/bin/env python3
"""
Test script to verify plagiarism form submission works correctly
"""

import requests
import json

def test_plagiarism_form():
    """Test the plagiarism detection form submission"""
    
    # Test data
    test_content = """
    This is a test article about artificial intelligence and machine learning.
    Artificial intelligence is transforming the way we work and live.
    Machine learning algorithms can process vast amounts of data to find patterns.
    Deep learning is a subset of machine learning that uses neural networks.
    Natural language processing helps computers understand human language.
    Computer vision enables machines to interpret visual information.
    AI applications are being used in healthcare, finance, and transportation.
    The future of AI holds great promise for solving complex problems.
    """
    
    # Test URL
    test_url = "https://example.com/test-article"
    
    print("🧪 Testing Plagiarism Form Submission")
    print("=" * 50)
    
    # Test with content
    print("📝 Testing with content text...")
    form_data = {
        'content': test_content,
        'deep_search': 'on',
        'citation_check': 'off',
        'paraphrase_check': 'on'
    }
    
    try:
        response = requests.post('http://localhost:5000/plagiarism', data=form_data)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Form submission successful!")
            # Check if response contains results
            if 'originality_score' in response.text:
                print("✅ Response contains plagiarism analysis results")
            else:
                print("⚠️  Response doesn't contain expected results")
        else:
            print(f"❌ Form submission failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure Flask app is running.")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test with URL
    print("🔗 Testing with URL...")
    form_data = {
        'url': test_url,
        'deep_search': 'on',
        'citation_check': 'off',
        'paraphrase_check': 'on'
    }
    
    try:
        response = requests.post('http://localhost:5000/plagiarism', data=form_data)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ URL form submission successful!")
        else:
            print(f"❌ URL form submission failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure Flask app is running.")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    print("🎯 Test completed!")

if __name__ == "__main__":
    test_plagiarism_form()
