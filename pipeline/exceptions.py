class AuthorizationError(Exception):
    """Corresponds to 403 response (likely no API key in request)"""

class RateLimitError(Exception):
    """Corresponds to 429 response (likely due to rate limit exceeded)"""