"""
Utility functions for content curation.
"""

from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


def canonicalize_url(url: str) -> str:
    """Normalize URL to reduce duplicates from tracking params, fragments, and case.

    - Lowercase scheme and host
    - Strip fragment and common tracking parameters
    - Remove www prefix and normalize path
    - Sort remaining query parameters for stability
    - Handle mobile/AMP variants
    """
    try:
        parsed = urlparse(url.strip())
        
        # Normalize scheme
        scheme = (parsed.scheme or 'https').lower()
        
        # Normalize netloc (host)
        netloc = (parsed.netloc or '').lower()
        
        # Remove www prefix
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        
        # Handle mobile and AMP variants
        if netloc.startswith('m.'):
            netloc = netloc[2:]
        elif netloc.startswith('mobile.'):
            netloc = netloc[7:]
        elif netloc.endswith('.amp.html') or '/amp/' in parsed.path:
            # Normalize AMP URLs by removing AMP-specific parts
            pass  # Keep netloc as-is but will handle path below
        
        # Normalize path
        path = parsed.path or '/'
        
        # Remove trailing slash unless it's the root
        if len(path) > 1 and path.endswith('/'):
            path = path[:-1]
        
        # Remove AMP-specific path components
        if path.endswith('/amp'):
            path = path[:-4] or '/'
        elif '/amp/' in path:
            path = path.replace('/amp/', '/')
        
        # Filter query params - remove tracking and session parameters
        tracking_params = {
            # UTM parameters
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            # Google Analytics
            'gclid', 'gclsrc', 'dclid',
            # Facebook
            'fbclid', 'fb_action_ids', 'fb_action_types', 'fb_ref', 'fb_source',
            # Other tracking
            'mc_eid', 'mc_cid', '_ga', '_gid', 'ref', 'referrer',
            # Session/cache busting
            'v', 'version', 'cache', 'timestamp', 't', '_',
            # Social media
            'share', 'shared', 'via', 'source',
            # Newsletter/email
            'newsletter', 'email', 'subscriber_id',
            # Affiliate tracking
            'affiliate', 'aff', 'partner', 'promo'
        }
        
        params = []
        for k, v in parse_qsl(parsed.query, keep_blank_values=True):
            lk = k.lower()
            # Skip tracking parameters
            if (lk in tracking_params or 
                lk.startswith('utm_') or 
                lk.startswith('ga_') or
                lk.startswith('fb_') or
                lk.startswith('mc_')):
                continue
            params.append((k, v))
        
        # Sort parameters for consistent ordering
        params.sort()
        
        # Reconstruct canonical URL
        canonical = urlunparse((
            scheme,
            netloc,
            path,
            '',  # params (unused)
            urlencode(params, doseq=True),
            ''   # fragment (always removed)
        ))
        
        return canonical
        
    except Exception:
        # Return original URL if canonicalization fails
        return url


def extract_domain(url: str) -> str:
    """Extract domain from URL, removing www prefix and lowercasing."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith('www.') else netloc
    except Exception:
        return ""