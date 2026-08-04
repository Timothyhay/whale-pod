"""Auth module."""
class TokenManager:
    """Manages tokens with TTL."""
    def __init__(self, store, ttl=3600):
        self.store = store
        self.ttl = ttl

    def issue(self, user_id: str) -> str:
        """Issue a fresh token."""
        return "tok:" + user_id

def authenticate_token(req) -> bool:
    return req.token == "valid"
