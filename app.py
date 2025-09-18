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
        st.session_state.df_user = df_user

        display_metrics(df_user)

        fig_sent = px.line(df_user, x="time", y="sentiment_score", markers=True, title="📈 Sentiment Timeline")
        fig_sent.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig_sent, use_container_width=True)

        fig_vol = px.line(df_user, x="time", y="volatility", markers=True, title="🌪 Volatility Timeline")
        st.plotly_chart(fig_vol, use_container_width=True)

    else:
        st.info("🔑 Please connect your Reddit account to see personal volatility.")      

# ------------------ COMMUNITY TAB ------------------
# ---------- Community tab (replace your existing tab2) ----------
with tab2:
    st.subheader("🌍 Community Emotional Volatility")
    subs = st.text_input("Enter subreddit(s):", "depression, mentalhealth, bpd")

    # choose aggregation interval (one value per subreddit per interval)
    time_bin = st.selectbox(
        "Aggregation interval (one point per subreddit per interval)",
        ["1min", "5min", "15min", "30min", "1H", "3H", "6H", "1D"],
        index=4
    )
    # persist selection so comparison tab can use the same interval
    st.session_state["time_bin"] = time_bin

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
            # ensure datetime and sorting
            df_comm["time"] = pd.to_datetime(df_comm["time"])
            df_comm = df_comm.sort_values(["subreddit", "time"]).reset_index(drop=True)

            # analyze sentiment (use existing analyzer; fallback if missing)
            try:
                analyzer_obj = _analyzer
            except NameError:
                analyzer_obj = SentimentEnsemble()

            df_comm = analyze_sentiment(df_comm, analyzer_obj)

            # aggregate/resample into time bins so there is one value per bin
            df_comm_agg = (
                df_comm
                .groupby(["subreddit", pd.Grouper(key="time", freq=time_bin)])["sentiment_score"]
                .mean()
                .reset_index()
            )

            # sort and compute rolling volatility on the aggregated series
            df_comm_agg = df_comm_agg.sort_values(["subreddit", "time"]).reset_index(drop=True)
            df_comm_agg["volatility"] = df_comm_agg.groupby("subreddit")["sentiment_score"].transform(
                lambda s: s.rolling(window=3, min_periods=1).std().fillna(0)
            )

            # save both raw and aggregated in session_state
            st.session_state.df_comm = df_comm          # raw per-post data
            st.session_state.df_comm_agg = df_comm_agg  # aggregated for plotting

            st.success(f"Fetched {len(df_comm)} posts — aggregated to {len(df_comm_agg)} points.")

    # load from session_state (if present)
    df_comm = st.session_state.get("df_comm", pd.DataFrame())
    df_comm_agg = st.session_state.get("df_comm_agg", pd.DataFrame())

    if not df_comm.empty:
        # show raw metrics if you want (keeps existing behavior)
        display_metrics(df_comm)

        # Plot aggregated sentiment (one series per subreddit)
        if not df_comm_agg.empty:
            fig_sent = px.line(
                df_comm_agg,
                x="time",
                y="sentiment_score",
                color="subreddit",
                title="📈 Community Sentiment (aggregated)",
                markers=True
            ).update_yaxes(range=[-1, 1])
            st.plotly_chart(fig_sent, use_container_width=True)

            fig_vol = px.line(
                df_comm_agg,
                x="time",
                y="volatility",
                color="subreddit",
                title="🌪 Community Volatility (aggregated)",
                markers=True
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("No aggregated data yet — press 'Fetch Community Data' to compute aggregation.")
    else:
        st.info("No community data. Click 'Fetch Community Data' to load posts.")


# ---------- Comparison tab (replace your existing tab3) ----------
with tab3:
    st.subheader("⚖️ You vs Community")
    df_user = st.session_state.get("df_user", pd.DataFrame())
    df_comm = st.session_state.get("df_comm", pd.DataFrame())          # raw per-post
    df_comm_agg = st.session_state.get("df_comm_agg", pd.DataFrame())  # aggregated
    time_bin = st.session_state.get("time_bin", "1min")

    if df_user.empty or df_comm.empty:
        st.warning("⚠️ Please fetch both datasets first.")
    else:
        # make sure user times are datetime & sorted
        df_user["time"] = pd.to_datetime(df_user["time"])
        df_user = df_user.sort_values("time").reset_index(drop=True)

        # aggregate user to same time_bin so comparison is apples-to-apples
        df_user_agg = (
            df_user
            .groupby(pd.Grouper(key="time", freq=time_bin))["sentiment_score"]
            .mean()
            .reset_index()
        )
        df_user_agg["source"] = df_user.get("type", "User").iloc[0].capitalize() if "type" in df_user.columns else "User"

        # aggregated community should already exist; if not, compute a quick aggregation now
        if df_comm_agg.empty:
            df_comm_agg = (
                df_comm
                .groupby(["subreddit", pd.Grouper(key="time", freq=time_bin)])["sentiment_score"]
                .mean()
                .reset_index()
            )
            df_comm_agg["volatility"] = df_comm_agg.groupby("subreddit")["sentiment_score"].transform(
                lambda s: s.rolling(window=3, min_periods=1).std().fillna(0)
            )

        # prepare combined DataFrame for sentiment comparison
        # for plotting, we unify column names: 'source' (user) and 'subreddit' (community)
        df_comm_agg_plot = df_comm_agg.rename(columns={"subreddit": "source"})
        df_sent_comb = pd.concat([
            df_user_agg.rename(columns={"sentiment_score": "sentiment_score", "time": "time"}).assign(source=df_user_agg["source"]),
            df_comm_agg_plot[["time", "sentiment_score", "source"]]
        ], ignore_index=True, sort=False)

        st.plotly_chart(
            px.line(df_sent_comb, x="time", y="sentiment_score", color="source",
                    title="📈 Sentiment Comparison (aggregated)").update_yaxes(range=[-1, 1]),
            use_container_width=True
        )

        # combined volatility plot (user volatility could be computed similarly if desired)
        # For now show community volatility per subreddit (source)
        df_comm_vol_plot = df_comm_agg.rename(columns={"subreddit": "source"})
        st.plotly_chart(
            px.line(df_comm_vol_plot, x="time", y="volatility", color="source",
                    title="🌪 Volatility Comparison (community aggregated)"),
            use_container_width=True
        )

        # Emotion distribution (keep original behavior using raw df_comm)
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
