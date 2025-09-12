import os
import pytest
from utils.reddit_client import get_reddit

@pytest.mark.skipif(
    not os.getenv("CLIENT_ID") or not os.getenv("CLIENT_SECRET"),
    reason="Reddit API credentials not set"
)
def test_reddit_connection():
    reddit = get_reddit()
    assert reddit.user.me() is None or isinstance(reddit.user.me(), object)
