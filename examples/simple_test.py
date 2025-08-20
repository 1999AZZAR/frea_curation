from curator.services._news_source import NewsSource

# Test URL canonicalization
url = "https://WWW.Example.com/article?utm_source=test&id=123"
canonical = NewsSource._canonicalize_url(url)
print(f"Original: {url}")
print(f"Canonical: {canonical}")
print(f"Expected: https://example.com/article?id=123")
print(f"Match: {canonical == 'https://example.com/article?id=123'}")