import os
import praw
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_reddit():
    return praw.Reddit(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        user_agent=os.getenv("USER_AGENT", "volatility_detector")
    )

def get_reddit_comments(subreddit, limit=50):
    reddit = get_reddit()
    comments = []
    for comment in reddit.subreddit(subreddit).comments(limit=limit):
        comments.append(comment.body)
    return comments
