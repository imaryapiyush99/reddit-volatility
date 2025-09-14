import streamlit as st
import plotly.express as px
import pandas as pd
import praw, os, nltk
from dotenv import load_dotenv
from utils.metrics import calculate_comprehensive_metrics, display_metrics
from utils.praw_oauth import get_oauth_reddit
from utils.praw_script import get_script_reddit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.sentiment import SentimentEnsemble, analyze_sentiment

# ------------------ Setup ------------------
nltk.download('vader_lexicon', quiet=True)
load_dotenv()
st.set_page_config(page_title="Reddit Emotional Volatility", layout="wide")


# ------------------ Custom Styling ------------------
st.markdown("""
<style>
    .big-title {
        font-size: 36px !important;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 20px !important;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 25px;
    }
    .footer {
        text-align: center;
        color: #95A5A6;
        font-size: 14px;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠 Reddit Emotional Volatility Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Track, Compare & Visualize Emotional Patterns Across Reddit</div>', unsafe_allow_html=True)

_analyzer = SentimentEnsemble()

# ------------------ Reddit Setup ------------------
reddit_oauth = get_oauth_reddit()
reddit_bot = get_script_reddit()
reddit_user = None

auth_url = reddit_oauth.auth.url(["identity", "history"], "state123", "permanent")
st.markdown(f"### 🔑 [Connect Reddit Account]({auth_url})")

params = st.query_params
code = params.get("code")

if code and "refresh_token" not in st.session_state:
    try:
        st.session_state.refresh_token = reddit_oauth.auth.authorize(code)
        st.success("✅ Authenticated successfully!")
    except Exception as e:
        st.error(f"Auth failed: {e}")

if "refresh_token" in st.session_state:
    try:
        reddit_user = praw.Reddit(
            client_id=os.getenv("OAUTH_CLIENT_ID"),
            client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
            refresh_token=st.session_state.refresh_token,
            user_agent="VolatilityApp by u/imaryapiyush99",
        )
        st.success(f"👤 Logged in as: {reddit_user.user.me().name}")
    except Exception as e:
        st.error(f"⚠️ Could not use refresh token: {e}")

# ------------------ Helper: Fetch User Activity ------------------
def fetch_user_activity(user, limit=200):
    """Fetch both comments and posts from a Reddit user."""
    activities = []

    # Comments
    try:
        for c in user.comments.new(limit=limit):
            activities.append({
                "time": pd.to_datetime(c.created_utc, unit="s"),
                "text": c.body,
                "type": "comment",
                "subreddit": str(c.subreddit)
            })
    except Exception as e:
        st.warning(f"⚠️ Could not fetch comments: {e}")

    # Posts
    try:
        for p in user.submissions.new(limit=limit):
            activities.append({
                "time": pd.to_datetime(p.created_utc, unit="s"),
                "text": f"{p.title} {p.selftext or ''}",
                "type": "post",
                "subreddit": str(p.subreddit)
            })
    except Exception as e:
        st.warning(f"⚠️ Could not fetch posts: {e}")

    df = pd.DataFrame(activities)
    if df.empty:
        return df

    # Sentiment analysis
    analyzer = SentimentEnsemble()
    df = analyze_sentiment(df, analyzer)
    df["volatility"] = df.groupby("type")["sentiment_score"].transform(lambda x: x.rolling(5).std().fillna(0))

    return df

# ------------------ Tabs ------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["👤 My Volatility", "🌍 Community Volatility", "⚖️ Comparison", "📊 Accuracy Benchmark"]
)

if "df_user" not in st.session_state:
    st.session_state.df_user = pd.DataFrame()
if "df_comm" not in st.session_state:
    st.session_state.df_comm = pd.DataFrame()

# ------------------ USER TAB ------------------
with tab1:
    st.subheader("👤 Your Emotional Volatility")
    if reddit_user and st.button("📥 Fetch My Data"):
        try:
            df_user = fetch_user_activity(reddit_user.user.me())
            if not df_user.empty:
                st.session_state.df_user = df_user
        except Exception as e:
            st.error(f"⚠️ Failed to fetch your activity: {e}")

    df_user = st.session_state.df_user
    if not df_user.empty:
        df_user = analyze_sentiment(df_user, _analyzer)
    
        st.session_state["df_user"] = df_user
        display_metrics(df_user)
        
        fig_sent = px.line(df_user, x="time", y="sentiment_score", markers=True, title="📈 Sentiment Timeline")
        fig_sent.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig_sent, use_container_width=True)

        fig_vol = px.line(df_user, x="time", y="volatility", markers=True, title="🌪 Volatility Timeline")
        st.plotly_chart(fig_vol, use_container_width=True)


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
    else:
        st.info("🔑 Please connect your Reddit account to see personal volatility.")    

# ------------------ COMMUNITY TAB ------------------
with tab2:
    st.subheader("🌍 Community Emotional Volatility")
    subs = st.text_input("Enter subreddit(s):", "depression, mentalhealth, bpd")

    if st.button("📥 Fetch Community Data"):
        all_posts = []
        for sub in [s.strip() for s in subs.split(",") if s.strip()]:
            try:
                for p in reddit_bot.subreddit(sub).new(limit=100):
                    all_posts.append({
                        "time": pd.to_datetime(p.created_utc, unit="s"),
                        "text": f"{p.title} {p.selftext}",
                        "subreddit": sub
                    })
            except Exception as e:
                st.warning(f"⚠️ r/{sub}: {e}")

        df_comm = pd.DataFrame(all_posts)
        if not df_comm.empty:
            analyzer = SentimentEnsemble()
            df_comm = analyze_sentiment(df_comm, _analyzer)
            df_comm["volatility"] = df_comm.groupby("subreddit")["sentiment_score"].transform(
                lambda x: x.rolling(5).std().fillna(0)
            )
            st.session_state[df_comm] = df_comm

    df_comm = st.session_state.df_comm
    if not df_comm.empty:
        display_metrics(df_comm)
        st.plotly_chart(
            px.line(df_comm, x="time", y="sentiment_score", color="subreddit",
                    title="📈 Community Sentiment", markers=True).update_yaxes(range=[-1, 1]),
            use_container_width=True
        )
        st.plotly_chart(
            px.line(df_comm, x="time", y="volatility", color="subreddit",
                    title="🌪 Community Volatility", markers=True),
            use_container_width=True
        )

# ------------------ COMPARISON TAB ------------------
with tab3:
    st.subheader("⚖️ You vs Community")
    df_user, df_comm = st.session_state.df_user, st.session_state.df_comm

    if df_user.empty or df_comm.empty:
        st.warning("⚠️ Please fetch both datasets first.")
    else:
        df_user["source"] = df_user["type"].str.capitalize()
        df_comm["source"] = df_comm["subreddit"]
        combined = pd.concat([df_user, df_comm])

        st.plotly_chart(
            px.line(combined, x="time", y="sentiment_score", color="source",
                    title="📈 Sentiment Comparison").update_yaxes(range=[-1, 1]),
            use_container_width=True
        )
        st.plotly_chart(
            px.line(combined, x="time", y="volatility", color="source",
                    title="🌪 Volatility Comparison"),
            use_container_width=True
        )

    # Emotion distribution
    emotion_counts = []
    if "subreddit" in df_comm.columns:
        for sub in df_comm["subreddit"].unique():
            df_sub = df_comm[df_comm["subreddit"] == sub]
            emotion_counts.append({
                "Subreddit": sub,
                "Positive": (df_sub["sentiment_label"] == "positive").sum(),
                "Neutral": (df_sub["sentiment_label"] == "neutral").sum(),
                "Negative": (df_sub["sentiment_label"] == "negative").sum(),
            })

        dist_df = pd.DataFrame(emotion_counts).melt(
            id_vars="Subreddit", var_name="Emotion", value_name="Count"
        )
        fig_dist = px.bar(
            dist_df, x="Subreddit", y="Count", color="Emotion",
            barmode="group", title="🔎 Emotion Distribution by Subreddit"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.warning("⚠️ No 'subreddit' column found. Please fetch community data again.")

# ------------------ BENCHMARK TAB ------------------
with tab4:
    st.header("📊 Accuracy Benchmark")
    uploaded_file = st.file_uploader("Upload a labeled Reddit sentiment dataset (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Check required columns
            if "text" not in df.columns or "label" not in df.columns:
                st.error("❌ Dataset must contain 'text' and 'label' columns.")
            else:
                df["text"] = df["text"].astype(str).fillna("")
                df["label"] = df["label"].astype(str).str.lower().str.strip()

                mapping = {"positive": 1, "neutral": 0, "negative": -1,
                           "1": 1, "0": 0, "-1": -1}
                df["label_mapped"] = df["label"].map(mapping)
                df = df.dropna(subset=["label_mapped"])

                if df.empty:
                    st.error("❌ No valid labels found after mapping.")
                else:
                    analyzer = SentimentEnsemble()
                    df["sentiment_score"] = df["text"].apply(analyzer.analyze_text)
                    df["sentiment"] = df["sentiment_score"].apply(
                        lambda s: 1 if s > 0.1 else (-1 if s < -0.1 else 0)
                    )

                    acc = accuracy_score(df["label_mapped"], df["sentiment"])
                    st.success(f"✅ Accuracy: {acc:.2%}")

                    unique_classes = sorted(df["label_mapped"].unique())
                    class_names = []
                    for uc in unique_classes:
                        for name, val in mapping.items():
                            if val == uc:
                                class_names.append(name)
                                break

                    report = classification_report(
                        df["label_mapped"], df["sentiment"],
                        labels=unique_classes, target_names=class_names
                    )
                    st.text(report)

                    cm = confusion_matrix(df["label_mapped"], df["sentiment"], labels=unique_classes)
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"True {c}" for c in class_names],
                        columns=[f"Pred {c}" for c in class_names]
                    )

                    st.subheader("📉 Confusion Matrix")
                    fig_cm = px.imshow(
                        cm_df, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Failed to process dataset: {e}")

# ------------------ Footer ------------------
st.markdown(
    """
    <div class="footer">
        ✨ Built with ❤️ by HexaMind | Powered by Reddit API & Streamlit ✨
    </div>
    """,
    unsafe_allow_html=True
)
