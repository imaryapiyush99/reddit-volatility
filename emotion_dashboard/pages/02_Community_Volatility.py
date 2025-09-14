import streamlit as st
import pandas as pd
import plotly.express as px


from utils.praw_script import get_script_reddit
from utils.metrics import calculate_comprehensive_metrics, display_metrics
from utils.sentiment import analyze_sentiment


# Add project root (/app) to sys.path so root-level utils is importable



st.title("🌍 Community Emotional Volatility")

reddit_bot = get_script_reddit()

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

    if df_comm.empty:
        st.warning("No posts fetched for the provided subreddits.")
    else:
        # Work on a copy to avoid SettingWithCopy issues
        df_comm = df_comm.copy()

        # Defensive sentiment calculation
        try:
            df_comm = analyze_sentiment(df_comm)
        except Exception as e:
            st.error(f"Failed to compute sentiment: {e}")
            st.write(df_comm.head())
            df_comm = pd.DataFrame()  # abort further processing


        # Verify sentiment column exists
        if df_comm.empty or "sentiment" not in df_comm.columns:
            st.error("No sentiment scores available for fetched posts (analyze_sentiment failed or returned no 'sentiment' column).")
        else:
            # Ensure time is datetime and sort by subreddit+time BEFORE rolling
            df_comm["time"] = pd.to_datetime(df_comm["time"])
            df_comm = df_comm.sort_values(["subreddit", "time"]).reset_index(drop=True)

            try:
                df_comm["volatility"] = df_comm.groupby("subreddit")["sentiment"].transform(
                    lambda x: x.rolling(5).std().fillna(0)
                )
            except Exception as e:
                st.error(f"Failed computing grouped volatility: {e}")
                st.write("DEBUG df_comm columns:", df_comm.columns.tolist())
                st.write(df_comm.head())
                df_comm["volatility"] = 0.0

            # Optional: compute metrics dict if you need it
            try:
                comm_metrics = calculate_comprehensive_metrics(df_comm)
            except Exception:
                comm_metrics = {}

            # Display metrics and charts
            display_metrics(df_comm)

            st.plotly_chart(
                px.line(df_comm, x="time", y="sentiment", color="subreddit", title="📈 Community Sentiment")
                  .update_yaxes(range=[-1, 1]),
                use_container_width=True,
            )
            st.plotly_chart(
                px.line(df_comm, x="time", y="volatility", color="subreddit", title="🌪 Community Volatility"),
                use_container_width=True,
            )

            # Build distribution safely
            rows = []
            for sub in subreddit_list:
                df_sub = df_comm[df_comm["subreddit"] == sub]
                rows.append({
                    "Subreddit": sub,
                    "Positive": int((df_sub["sentiment"] > 0.1).sum()),
                    "Neutral": int(df_sub["sentiment"].between(-0.1, 0.1).sum()),
                    "Negative": int((df_sub["sentiment"] < -0.1).sum()),
                })

            if rows:
                dist_df = pd.DataFrame(rows).melt(id_vars="Subreddit", var_name="Emotion", value_name="Count")
                st.plotly_chart(
                    px.bar(dist_df, x="Subreddit", y="Count", color="Emotion", barmode="group", title="🔎 Emotion Distribution by Subreddit"),
                    use_container_width=True,
                )
            else:
                st.info("No distribution data to show.")
