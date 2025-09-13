# app.py
import sys, os
import streamlit as st
import praw
import pandas as pd
import numpy as np
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from urllib.parse import urlparse, parse_qs

# Add project root so imports from utils work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ------------------ Imports from utils ------------------
from utils.metrics import calculate_comprehensive_metrics, display_metrics
from utils.praw_oauth import get_oauth_reddit
from utils.praw_script import get_script_reddit

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Reddit Emotional Volatility", layout="wide")
st.title("🧠 Emotional Volatility Dashboard")

# ------------------ Sentiment Setup ------------------
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()


# ------------------ Helper Functions ------------------
def analyze_sentiment(df):
    sentiments = []
    for text in df["text"].fillna(""):
        vader_score = sia.polarity_scores(text)["compound"]
        blob_score = TextBlob(text).sentiment.polarity
        combined = np.mean([vader_score, blob_score])
        sentiments.append(combined)
    df["sentiment"] = sentiments
    return df


# ------------------ Reddit API Setup ------------------
reddit_oauth = get_oauth_reddit()  # OAuth for user login
reddit_bot = get_script_reddit()   # Script for background community scraping

# ------------------ OAuth Login ------------------
st.title("🔗 Connect Reddit Account")

auth_url = reddit_oauth.auth.url(["identity", "history"], "state123", "permanent")
st.markdown(f"[🔗 Connect Reddit Account]({auth_url})")

query_params = st.query_params
code = query_params.get("code", [None])[0]

if code and "refresh_token" not in st.session_state:
    try:
        refresh_token = reddit_oauth.auth.authorize(code)
        st.session_state.refresh_token = refresh_token
        st.success("✅ Authenticated successfully!")
    except Exception as e:
        st.error(f"Auth failed: {e}")

# Use refresh token if available
reddit_user = None
if "refresh_token" in st.session_state:
    reddit_user = praw.Reddit(
        client_id=os.getenv("OAUTH_CLIENT_ID"),
        client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
        refresh_token=st.session_state.refresh_token,
        redirect_uri="http://localhost:8501/",   # MUST match Reddit app settings
        user_agent="VolatilityApp by u/imaryapiyush99",
    )
    me = reddit_user.user.me()
    st.success(f"Logged in as: {me.name}")


# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["👤 My Volatility", "🌍 Community Volatility", "📊 Comparison"])
df_user, df_comm = pd.DataFrame(), pd.DataFrame()

# ---------------------- USER TAB ----------------------
with tab1:
    st.subheader("👤 My Reddit Volatility")

    if reddit_user:
        comments = [
            {"time": pd.to_datetime(c.created_utc, unit="s"), "text": c.body}
            for c in reddit_user.user.me().comments.new(limit=50)
        ]
        df_user = pd.DataFrame(comments)

        if not df_user.empty:
            df_user = analyze_sentiment(df_user)
            df_user["volatility"] = df_user["sentiment"].rolling(5).std().fillna(0)

            display_metrics(df_user)

            fig_sent = px.line(df_user, x="time", y="sentiment", markers=True, title="📈 Sentiment Timeline")
            fig_sent.update_yaxes(range=[-1, 1])
            st.plotly_chart(fig_sent, use_container_width=True)

            fig_vol = px.line(df_user, x="time", y="volatility", markers=True, title="🌪 Volatility Timeline")
            st.plotly_chart(fig_vol, use_container_width=True)

            counts = {
                "Positive": (df_user["sentiment"] > 0.1).sum(),
                "Neutral": (df_user["sentiment"].between(-0.1, 0.1)).sum(),
                "Negative": (df_user["sentiment"] < -0.1).sum(),
            }
            dist_df = pd.DataFrame({"Emotion": counts.keys(), "Count": counts.values()})
            fig_dist = px.pie(
                dist_df,
                names="Emotion",
                values="Count",
                color="Emotion",
                color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("🔑 Please connect your Reddit account to see personal volatility.")


# ---------------------- COMMUNITY TAB ----------------------
with tab2:
    st.subheader("🌍 Community Emotional Volatility")

    subreddit_input = st.text_input("Enter subreddit(s), comma separated:", "depression, mentalhealth")

    if st.button("Fetch Community Data"):
        subreddit_list = [s.strip() for s in subreddit_input.split(",") if s.strip()]
        all_posts = []

        for sub in subreddit_list:
            try:
                for p in reddit_bot.subreddit(sub).new(limit=50):
                    all_posts.append(
                        {
                            "time": pd.to_datetime(p.created_utc, unit="s"),
                            "text": (p.title or "") + " " + (p.selftext or ""),
                            "author": str(p.author),
                            "subreddit": sub,
                        }
                    )
            except Exception as e:
                st.warning(f"⚠️ Could not fetch from r/{sub}: {e}")

        df_comm = pd.DataFrame(all_posts)

        if not df_comm.empty:
            df_comm = analyze_sentiment(df_comm)
            df_comm["volatility"] = df_comm.groupby("subreddit")["sentiment"].transform(
                lambda x: x.rolling(5).std().fillna(0)
            )

            display_metrics(df_comm)

            fig_sent_comm = px.line(df_comm, x="time", y="sentiment", color="subreddit", title="📈 Community Sentiment")
            fig_sent_comm.update_yaxes(range=[-1, 1])
            st.plotly_chart(fig_sent_comm, use_container_width=True)

            fig_vol_comm = px.line(df_comm, x="time", y="volatility", color="subreddit", title="🌪 Community Volatility")
            st.plotly_chart(fig_vol_comm, use_container_width=True)

            emotion_counts = []
            for sub in subreddit_list:
                df_sub = df_comm[df_comm["subreddit"] == sub]
                emotion_counts.append(
                    {
                        "Subreddit": sub,
                        "Positive": (df_sub["sentiment"] > 0.1).sum(),
                        "Neutral": (df_sub["sentiment"].between(-0.1, 0.1)).sum(),
                        "Negative": (df_sub["sentiment"] < -0.1).sum(),
                    }
                )

            dist_df = pd.DataFrame(emotion_counts).melt(id_vars="Subreddit", var_name="Emotion", value_name="Count")
            fig_dist = px.bar(
                dist_df, x="Subreddit", y="Count", color="Emotion",
                barmode="group", title="🔎 Emotion Distribution by Subreddit"
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ---------------------- COMPARISON TAB ----------------------
with tab3:
    st.subheader("📊 Comparison")
    if not df_user.empty and not df_comm.empty:
        df_user_plot = df_user.copy()
        df_user_plot["source"] = "Me"
        df_comm_plot = df_comm.copy()
        df_comm_plot["source"] = df_comm_plot["subreddit"]

        df_all = pd.concat([df_user_plot, df_comm_plot])

        fig_compare = px.line(df_all, x="time", y="sentiment", color="source", title="My Sentiment vs Community")
        fig_compare.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info("Fetch both personal and community data to compare.")
