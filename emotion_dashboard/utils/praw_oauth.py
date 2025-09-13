import praw
import os

def get_oauth_reddit():
    """
    Returns a PRAW Reddit instance configured for OAuth login.
    Requires environment variables:
    - OAUTH_CLIENT_ID
    - OAUTH_CLIENT_SECRET
    - REDIRECT_URI (must match your Reddit app settings)
    """
    return praw.Reddit(
        client_id=os.getenv("OAUTH_CLIENT_ID"),
        client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
        redirect_uri=os.getenv("REDIRECT_URI", "http://localhost:8501/"),
        user_agent="VolatilityApp OAuth",
    )
