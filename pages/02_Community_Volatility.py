import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.praw_script import get_script_reddit
from utils.metrics import calculate_comprehensive_metrics, display_metrics
from utils.sentiment import analyze_sentiment, SentimentEnsemble


st.title("🌍 Community Emotional Volatility")

reddit_bot = get_script_reddit()

subreddit_input = st.text_input("Enter subreddit(s), comma separated:", "depression, mentalhealth, bpd")

# Initialize in session_state if not already present
if "df_comm" not in st.session_state:
    st.session_state["df_comm"] = pd.DataFrame()

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

    if df_comm.empty:
        st.warning("No posts fetched for the provided subreddits.")
    else:
        df_comm = df_comm.copy()

        try:
            analyzer = SentimentEnsemble()
            df_comm = analyze_sentiment(df_comm, analyzer)
        except Exception as e:
            st.error(f"Failed to compute sentiment: {e}")
            st.write(df_comm.head())
            df_comm = pd.DataFrame()

        if not df_comm.empty and "sentiment_score" in df_comm.columns:
            df_comm["time"] = pd.to_datetime(df_comm["time"])
            df_comm = df_comm.sort_values(["subreddit", "time"]).reset_index(drop=True)

            try:
                df_comm["volatility"] = df_comm.groupby("subreddit")["sentiment_score"].transform(
                    lambda x: x.rolling(5).std().fillna(0)
                )
            except Exception as e:
                st.error(f"Failed computing grouped volatility: {e}")
                df_comm["volatility"] = 0.0

            st.session_state["df_comm"] = df_comm

# Always reuse stored data
df_comm = st.session_state["df_comm"]

if df_comm.empty:
    st.info("⚠️ No community data available. Please fetch it first.")
else:
    try:
        comm_metrics = calculate_comprehensive_metrics(df_comm)
    except Exception:
        comm_metrics = {}

    display_metrics(df_comm)

    # 📈 Sentiment timeline
    st.plotly_chart(
        px.line(
            df_comm, x="time", y="sentiment_score", color="subreddit",
            title="📈 Community Sentiment"
        ).update_yaxes(range=[-1, 1]),
        use_container_width=True,
    )

    # 🌪 Volatility timeline
    st.plotly_chart(
        px.line(
            df_comm, x="time", y="volatility", color="subreddit",
            title="🌪 Community Volatility"
        ),
        use_container_width=True,
    )

    dist_df = pd.DataFrame(emotion_counts).melt(id_vars="Subreddit", var_name="Emotion", value_name="Count")
    fig_dist = px.bar(
        dist_df, x="Subreddit", y="Count", color="Emotion",
        barmode="group", title="🔎 Emotion Distribution by Subreddit"
        )
    st.plotly_chart(fig_dist, use_container_width=True)

    # 🔎 Emotion distribution
    rows = []
    for sub in df_comm["subreddit"].unique():
        df_sub = df_comm[df_comm["subreddit"] == sub]
        rows.append({
            "Subreddit": sub,
            "Positive": int((df_sub["sentiment_score"] > 0.1).sum()),
            "Neutral": int(df_sub["sentiment_score"].between(-0.1, 0.1).sum()),
            "Negative": int((df_sub["sentiment_score"] < -0.1).sum()),
        })

    if rows:
        dist_df = pd.DataFrame(rows).melt(
            id_vars="Subreddit", var_name="Emotion", value_name="Count"
        )
        st.plotly_chart(
            px.bar(
                dist_df, x="Subreddit", y="Count", color="Emotion",
                barmode="group", title="🔎 Emotion Distribution by Subreddit"
            ),
            use_container_width=True,
        )
    else:
        st.info("No distribution data to show.")
