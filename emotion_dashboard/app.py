import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import praw
import nltk


nltk.download('vader_lexicon', quiet=True)

from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from dotenv import load_dotenv
from utils.sentiment import analyze_sentiment

load_dotenv()



def ensure_dataframe(obj):
    """
    Make sure we have a pandas.DataFrame.
    - If obj is a DataFrame, return copy
    - If obj is a GroupBy, return the underlying df (obj.obj)
    - If obj is a list/dict of records, convert to DataFrame
    - Else return empty DataFrame
    """
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    # handle GroupBy (pandas internals may vary; import inside to be safe)
    try:
        from pandas.core.groupby.generic import DataFrameGroupBy
        if isinstance(obj, DataFrameGroupBy):
            return obj.obj.copy()
    except Exception:
        pass

    if isinstance(obj, list):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()
    if isinstance(obj, dict):
        try:
            return pd.DataFrame([obj])
        except Exception:
            return pd.DataFrame()
    # fallback: try to coerce
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()

from utils.metrics import calculate_comprehensive_metrics, display_metrics
from utils.praw_oauth import get_oauth_reddit
from utils.praw_script import get_script_reddit

# ------------------ Local sentiment ensemble (kept inside app.py) ------------------
# We add this here so we don't modify utils/ files as requested.




class SentimentEnsemble:
    """Lightweight ensemble using VADER + TextBlob (safe to run in-app).
    Returns a single float sentiment in range ~[-1, 1].
    """
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # Use VADER twice (vader + nltk_vader) kept for parity with your earlier design
        self.nltk_analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return 0.0

        try:
            vader_scores = self.vader.polarity_scores(text)
            vader_compound = vader_scores.get('compound', 0.0)
        except Exception:
            vader_compound = 0.0

        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
        except Exception:
            textblob_polarity = 0.0

        try:
            nltk_scores = self.nltk_analyzer.polarity_scores(text)
            nltk_compound = nltk_scores.get('compound', 0.0)
        except Exception:
            nltk_compound = 0.0

        # Weighted ensemble
        ensemble_score = (
            0.4 * vader_compound +
            0.3 * textblob_polarity +
            0.3 * nltk_compound
        )

        # Clamp to [-1, 1]
        ensemble_score = max(min(ensemble_score, 1.0), -1.0)
        return ensemble_score


# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Reddit Emotional Volatility", layout="wide")
st.title("🧠 Emotional Volatility Dashboard")

# ------------------ Reddit API Setup ------------------
reddit_oauth = get_oauth_reddit()  # for user login
reddit_bot = get_script_reddit()   # for subreddit scraping

# ------------------ OAuth Login ------------------
st.subheader("🔗 Connect Reddit Account")

auth_url = reddit_oauth.auth.url(["identity", "history"], "state123", "permanent")
st.markdown(f"[👉 Connect your Reddit Account]({auth_url})")

# Grab query params (when Reddit redirects back with ?code=xxx)
params = st.query_params
code = params.get("code")

reddit_user = None
if code and "refresh_token" not in st.session_state:
    try:
        refresh_token = reddit_oauth.auth.authorize(code)
        st.session_state.refresh_token = refresh_token
        st.success("✅ Authenticated successfully!")
    except Exception as e:
        st.error(f"Auth failed: {e}")

# Use refresh token if already stored
if "refresh_token" in st.session_state:
    try:
        reddit_user = praw.Reddit(
            client_id=os.getenv("OAUTH_CLIENT_ID"),
            client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
            refresh_token=st.session_state.refresh_token,
            user_agent="VolatilityApp by u/imaryapiyush99",
        )
        me = reddit_user.user.me()
        st.success(f"👤 Logged in as: {me.name}")
    except Exception as e:
        st.error(f"⚠️ Could not use refresh token: {e}")


# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["👤 My Volatility", "🌍 Community Volatility", "📊 Comparison"])

# Keep DataFrames in session_state so tabs persist across reruns
if 'df_user' not in st.session_state:
    st.session_state.df_user = pd.DataFrame()
if 'df_comm' not in st.session_state:
    st.session_state.df_comm = pd.DataFrame()

# ---------------------- USER TAB ----------------------
    # ---- USER TAB (replace existing Fetch My Data handling) ----
if reddit_user:
    if st.button("📥 Fetch My Data"):
        comments = [
            {"time": pd.to_datetime(c.created_utc, unit="s"), "text": c.body}
            for c in reddit_user.user.me().comments.new(limit=100)
        ]
        df_user = pd.DataFrame(comments)
        # coerce to DataFrame in case a stale/GroupBy/other object sneaked in
        df_user = ensure_dataframe(df_user)


        if df_user.empty:
            st.warning("No comments found for the user.")
        else:
            df_user['text'] = df_user['text'].astype(str)

            # compute sentiment if missing (defensive)
            if 'sentiment' not in df_user.columns:
                try:
                    analyzer = SentimentEnsemble()
                    df_user['sentiment'] = df_user['text'].apply(analyzer.analyze_text)
                except Exception as e:
                    st.error(f"Failed to compute sentiment: {e}")
                    st.write(df_user.head())
                    # abort early
                    df_user = pd.DataFrame()

            if df_user.empty:
                st.warning("No community data fetched. Try different subreddit(s).")

            # compute volatility column (after sentiment exists)
            try:
                df_user = df_user.sort_values('time')
                df_user["volatility"] = df_user["sentiment"].rolling(5).std().fillna(0)
            except Exception as e:
                st.error(f"Failed computing volatility: {e}")
                st.write("DEBUG df_user columns:", df_user.columns.tolist())
                st.write(df_user.head())
                

            # Save as records (safer for session_state)
            st.session_state.df_user_records = df_user.to_dict('records')

            # metrics dict
            user_metrics = calculate_comprehensive_metrics(df_user)

            st.markdown("### My Data")
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
            fig_dist = px.pie(dist_df, names="Emotion", values="Count", title="🧾 Emotion Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)

    if st.button("🔁 Reset stored data (clear session)"):
        st.session_state.clear()
        st.experimental_rerun()



# ---------------------- COMMUNITY TAB ----------------------
with tab2:
    st.subheader("🌍 Community Emotional Volatility")

    subreddit_input = st.text_input("Enter subreddit(s), comma separated:", "depression, mentalhealth, bpd")

  
# ---- COMMUNITY TAB (replace existing Fetch Community Data handling) ----
if st.button("📥 Fetch Community Data"):
    subreddit_list = [s.strip() for s in subreddit_input.split(",") if s.strip()]
    all_posts = []

    for sub in subreddit_list:
        try:
            for p in reddit_bot.subreddit(sub).new(limit=100):
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
    df_comm = ensure_dataframe(df_comm)


    if df_comm.empty:
        st.warning("No posts fetched for the provided subreddits.")
    else:
        df_comm['text'] = df_comm['text'].astype(str)

        # compute sentiment if missing (defensive)
        if 'sentiment' not in df_comm.columns:
            try:
                analyzer = SentimentEnsemble()
                df_comm['sentiment'] = df_comm['text'].apply(analyzer.analyze_text)
            except Exception as e:
                st.error(f"Failed to compute sentiment: {e}")
                st.write(df_comm.head())
                df_comm = pd.DataFrame()

        if df_comm.empty:
            st.warning("No community data fetched. Try different subreddit(s).")

        # compute grouped volatility
        try:
            df_comm = df_comm.sort_values('time')
            df_comm["volatility"] = df_comm.groupby("subreddit")["sentiment"].transform(
                lambda x: x.rolling(5).std().fillna(0)
            )
        except Exception as e:
            st.error(f"Failed computing grouped volatility: {e}")
            st.write("DEBUG df_comm columns:", df_comm.columns.tolist())
            st.write(df_comm.head())
           

        # Save as records (safer for session_state)
        st.session_state.df_comm_records = df_comm.to_dict('records')

        comm_metrics = calculate_comprehensive_metrics(df_comm)

        st.markdown("### Community Data")
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
        fig_dist = px.bar(dist_df, x="Subreddit", y="Count", color="Emotion", barmode="group", title="🔎 Emotion Distribution by Subreddit")
        st.plotly_chart(fig_dist, use_container_width=True)

               


# ---------------------- COMPARISON TAB ----------------------
with tab3:
    st.subheader("📊 Comparison")
    df_user = st.session_state.get('df_user', pd.DataFrame())
    df_comm = st.session_state.get('df_comm', pd.DataFrame())

    if not df_user.empty and not df_comm.empty:
        df_user_plot = df_user.copy()
        df_user_plot["source"] = "Me"
        df_comm_plot = df_comm.copy()
        df_comm_plot["source"] = df_comm_plot["subreddit"]

        df_all = pd.concat([df_user_plot, df_comm_plot])

        fig_compare = px.line(df_all, x="time", y="sentiment", color="source", title="📊 My Sentiment vs Community")
        fig_compare.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info("Fetch both personal and community data to compare.")
