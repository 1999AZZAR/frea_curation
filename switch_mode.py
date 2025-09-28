#!/usr/bin/env python3
"""
Script to switch between development and production CSS modes
"""

import os
import shutil
import sys

def switch_to_production():
    """Switch to production mode using local CSS files"""
    print("🔄 Switching to PRODUCTION mode...")
    
    # Backup current base.html
    if os.path.exists('templates/base.html'):
        shutil.copy('templates/base.html', 'templates/base_dev.html')
        print("✅ Backed up base.html to base_dev.html")
    
    # Use production template
    if os.path.exists('templates/base_production.html'):
        shutil.copy('templates/base_production.html', 'templates/base.html')
        print("✅ Switched to production base template")
    
    print("🚀 Production mode activated!")
    print("   - Using local Tailwind CSS")
    print("   - Using local FontAwesome CSS")
    print("   - No CDN dependencies")

def switch_to_development():
    """Switch to development mode using CDN"""
    print("🔄 Switching to DEVELOPMENT mode...")
    
    # Restore development base.html
    if os.path.exists('templates/base_dev.html'):
        shutil.copy('templates/base_dev.html', 'templates/base.html')
        print("✅ Restored development base template")
    
    print("🛠️  Development mode activated!")
    print("   - Using Tailwind CDN")
    print("   - Using FontAwesome CDN")
    print("   - Faster development iteration")

def show_status():
    """Show current mode status"""
    if os.path.exists('templates/base_dev.html'):
        print("📊 Current mode: PRODUCTION")
        print("   - Local CSS files")
        print("   - Optimized for production")
    else:
        print("📊 Current mode: DEVELOPMENT")
        print("   - CDN resources")
        print("   - Faster development")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python switch_mode.py [production|development|status]")
        print()
        print("Commands:")
        print("  production   - Switch to production mode (local CSS)")
        print("  development  - Switch to development mode (CDN)")
        print("  status       - Show current mode")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "production":
        switch_to_production()
    elif command == "development":
        switch_to_development()
    elif command == "status":
        show_status()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: production, development, or status")
        sys.exit(1)
