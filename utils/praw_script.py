import praw
import os

def get_script_reddit():
    """
    Returns a PRAW Reddit instance using script authentication.
    Requires environment variables:
    - SCRIPT_CLIENT_ID
    - SCRIPT_CLIENT_SECRET
    - REDDIT_USERNAME
    - REDDIT_PASSWORD
    """
    return praw.Reddit(
        client_id=os.getenv("SCRIPT_CLIENT_ID"),
        client_secret=os.getenv("SCRIPT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent="VolatilityApp Script",
    )
