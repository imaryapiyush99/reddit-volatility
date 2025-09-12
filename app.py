import streamlit as st
from utils.reddit_client import get_reddit_comments
from utils.sentiment import analyze_text
from utils.volatility import VolatilityTracker

st.set_page_config(page_title="Emotional Volatility Monitor", layout="wide")
st.title("🧠 Real-time Emotional Volatility Detection (Reddit)")

subreddit = st.sidebar.text_input("Subreddit", "mentalhealth")
window = st.sidebar.slider("Volatility Window", 5, 50, 20)
threshold = st.sidebar.slider("Volatility Threshold", 0.1, 1.0, 0.3)

tracker = VolatilityTracker(window)

if st.button("▶️ Start Stream"):
    comments = get_reddit_comments(subreddit, limit=50)
    results = []
    for text in comments:
        score = analyze_text(text)
        volatility = tracker.update(score)
        results.append({"comment": text, "sentiment": score, "volatility": volatility})

    st.write(results)  # Replace with chart & table later
