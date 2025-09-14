import streamlit as st
import pandas as pd
import plotly.express as px
import praw
import os

from utils.praw_oauth import get_oauth_reddit
from utils.metrics import calculate_comprehensive_metrics, display_metrics
from utils.sentiment import analyze_sentiment

st.title("👤 My Reddit Volatility")

# --- OAuth ---
reddit_oauth = get_oauth_reddit()
auth_url = reddit_oauth.auth.url(["identity", "history"], "state123", "permanent")
st.markdown(f"[🔗 Connect Reddit Account]({auth_url})")

# --- Handle OAuth redirect ---
code = st.query_params.get("code", [None])[0]
if code and "refresh_token" not in st.session_state:
    try:
        st.session_state.refresh_token = reddit_oauth.auth.authorize(code)
        st.success("✅ Authenticated successfully!")
    except Exception as e:
        st.error(f"Auth failed: {e}")

# --- Authenticate Reddit user ---
reddit_user = None
if "refresh_token" in st.session_state:
    reddit_user = praw.Reddit(
        client_id=os.getenv("OAUTH_CLIENT_ID"),
        client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
        refresh_token=st.session_state.refresh_token,
        user_agent="VolatilityApp by u/imaryapiyush99",
    )
    st.success(f"Logged in as: {reddit_user.user.me().name}")

# --- Initialize session storage ---
if "df_user" not in st.session_state:
    st.session_state["df_user"] = pd.DataFrame()

# --- Fetch data ---
if reddit_user and st.button("Fetch My Data"):
    comments = [
        {"time": pd.to_datetime(c.created_utc, unit="s"), "text": c.body}
        for c in reddit_user.user.me().comments.new(limit=50)
    ]
    df_user = pd.DataFrame(comments)

    if not df_user.empty:
        # Sentiment + volatility
        df_user = analyze_sentiment(df_user)
        df_user["volatility"] = df_user["sentiment"].rolling(5).std().fillna(0)

        # Save in session_state ✅
        st.session_state["df_user"] = df_user

# --- Always reuse stored data ---
df_user = st.session_state["df_user"]

if df_user.empty:
    st.info("🔑 Please fetch your Reddit data to see personal volatility.")
else:
    # Show metrics and plots
    display_metrics(df_user)

    st.plotly_chart(
        px.line(df_user, x="time", y="sentiment_score", title="📈 Sentiment Timeline", markers=True)
          .update_yaxes(range=[-1, 1]),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(df_user, x="time", y="volatility", title="🌪 Volatility Timeline", markers=True),
        use_container_width=True,
    )
    st.plotly_chart(
            px.line(df_user, x="time", y="sentiment_score", color="type",
                    title="📈 Sentiment Timeline (Posts vs Comments)", markers=True)
              .update_yaxes(range=[-1, 1]),
            use_container_width=True
    )
    st.plotly_chart(
            px.line(df_user, x="time", y="volatility", color="type",
                    title="🌪 Volatility Timeline (Posts vs Comments)", markers=True),
            use_container_width=True
    )
