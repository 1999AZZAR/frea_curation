#!/usr/bin/env python3
"""Simple template validation test"""

import os
import sys
from jinja2 import Environment, FileSystemLoader

def test_templates():
    """Test that all templates can be loaded and compiled"""
    try:
        env = Environment(loader=FileSystemLoader('templates'))
        templates = ['base.html', 'index.html', 'results.html', 'curation_results.html', 'compare.html']
        
        for template_name in templates:
            try:
                template = env.get_template(template_name)
                print(f'✓ {template_name} - OK')
            except Exception as e:
                print(f'✗ {template_name} - ERROR: {e}')
                return False
        
        # Test error templates
        error_templates = ['errors/400.html', 'errors/404.html', 'errors/500.html']
        for template_name in error_templates:
            try:
                template = env.get_template(template_name)
                print(f'✓ {template_name} - OK')
            except Exception as e:
                print(f'✗ {template_name} - ERROR: {e}')
                return False
        
        print("\n✅ All templates validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Template validation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_templates()
    sys.exit(0 if success else 1)