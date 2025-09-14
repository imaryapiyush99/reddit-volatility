import streamlit as st
import plotly.express as px
import pandas as pd

st.title("📊 Community Daily Sentiment Trend")

# --- Check community data ---
df_comm = st.session_state.get("df_comm", pd.DataFrame())

if df_comm.empty:
    st.warning("⚠️ No community data available. Please fetch it from the main dashboard first.")
elif not {"time", "sentiment", "subreddit"}.issubset(df_comm.columns):
    st.error("❌ The dataset must have 'time', 'sentiment', and 'subreddit' columns. Please refetch data.")
    st.stop()
else:
    # Ensure datetime
    df_comm = df_comm.copy()
    df_comm["time"] = pd.to_datetime(df_comm["time"], errors="coerce")
    df_comm = df_comm.dropna(subset=["time"])

    if df_comm.empty:
        st.warning("⚠️ No valid datetime values found in community data.")
    else:
        # --- Per subreddit daily averages ---
        trend = (
            df_comm.groupby([df_comm["time"].dt.date, "subreddit"])["sentiment"]
            .mean()
            .reset_index()
        )
        trend.rename(columns={"time": "date", "sentiment": "avg_sentiment"}, inplace=True)

        # --- Overall daily average across all subreddits ---
        overall = (
            df_comm.groupby(df_comm["time"].dt.date)["sentiment"]
            .mean()
            .reset_index()
        )
        overall["subreddit"] = "All Communities"
        overall.rename(columns={"time": "date", "sentiment": "avg_sentiment"}, inplace=True)

        # Merge subreddit + overall
        trend_all = pd.concat([trend, overall], ignore_index=True)

        # --- Plot ---
        fig_trend = px.line(
            trend_all,
            x="date",
            y="avg_sentiment",
            color="subreddit",
            markers=True,
            title="📊 Daily Sentiment Trend (Communities + Overall)"
        )
        fig_trend.update_yaxes(range=[-1, 1])

        # Make overall line thicker + dashed
        fig_trend.for_each_trace(
            lambda t: t.update(line=dict(width=4, dash="dash", color="black"))
            if t.name == "All Communities" else ()
        )

        st.plotly_chart(fig_trend, use_container_width=True)
