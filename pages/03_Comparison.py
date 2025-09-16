import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Comparison: My Sentiment vs Community")

# --- Check data availability ---
df_user = st.session_state.get("df_user", pd.DataFrame())
df_comm = st.session_state.get("df_comm", pd.DataFrame())

if df_user.empty:
    st.warning("⚠️ No user data available. Please fetch your Reddit comments first.")
elif df_comm.empty:
    st.warning("⚠️ No community data available. Please fetch subreddit posts first.")
elif "subreddit" not in df_comm.columns:
    st.error("❌ Community dataset has no 'subreddit' column. Please refetch data.")
    st.stop()
else:
    # --- Sentiment Comparison ---
    df_user_plot = df_user.copy()
    df_user_plot["source"] = "Me"

    df_comm_plot = df_comm.copy()
    df_comm_plot["source"] = df_comm_plot["subreddit"]

    df_all = pd.concat([df_user_plot, df_comm_plot], ignore_index=True)

    fig_sent = px.line(
        df_all,
        x="time",
        y="sentiment_score",
        color="source",
        title="📈 Sentiment: Me vs Community"
    )
    fig_sent.update_yaxes(range=[-1, 1])
    st.plotly_chart(fig_sent, use_container_width=True)

    # --- Volatility Comparison ---
    if "volatility" in df_user.columns and "volatility" in df_comm.columns:
        df_user_vol = df_user.copy()
        df_user_vol["source"] = "Me"

        df_comm_vol = df_comm.copy()
        df_comm_vol["source"] = df_comm_vol["subreddit"]

        df_all_vol = pd.concat([df_user_vol, df_comm_vol], ignore_index=True)

        # Chart 1: Me vs Each Subreddit
        fig_vol_each = px.line(
            df_all_vol,
            x="time",
            y="volatility",
            color="source",
            title="🌪 Volatility: Me vs Each Subreddit",
            labels={"volatility": "Volatility", "source": "Source"}
        )
        st.plotly_chart(fig_vol_each, use_container_width=True)

        # Chart 2: Me vs Community Average
        df_comm_avg = df_comm.groupby(df_comm["time"].dt.date)["volatility"].mean().reset_index()
        df_comm_avg["source"] = "Community Avg"

        df_user_avg = df_user.copy()
        df_user_avg = df_user_avg.groupby(df_user_avg["time"].dt.date)["volatility"].mean().reset_index()
        df_user_avg["source"] = "Me"

        df_avg = pd.concat([df_user_avg, df_comm_avg], ignore_index=True)

        fig_vol_avg = px.line(
            df_avg,
            x="time",
            y="volatility",
            color="source",
            title="🌍 Volatility: Me vs Community Average"
        )
        st.plotly_chart(fig_vol_avg, use_container_width=True)

    # 🔎 Emotion Distribution: Subreddit vs My Emotion
    rows = []

# Subreddit distributions
    for sub in df_comm["subreddit"].unique():
        df_sub = df_comm[df_comm["subreddit"] == sub]
        rows.append({
            "Source": sub,  # <- each subreddit is a source
            "Positive": int((df_sub["sentiment_score"] > 0.1).sum()),
            "Neutral": int(df_sub["sentiment_score"].between(-0.1, 0.1).sum()),
            "Negative": int((df_sub["sentiment_score"] < -0.1).sum()),
        })

# --- Add My Emotion distribution ---
    if not df_user.empty and "sentiment_score" in df_user.columns:
        rows.append({
            "Source": "My Emotion",
            "Positive": int((df_user["sentiment_score"] > 0.1).sum()),
            "Neutral": int(df_user["sentiment_score"].between(-0.1, 0.1).sum()),
            "Negative": int((df_user["sentiment_score"] < -0.1).sum()),
        })

# --- Build chart ---
    if rows:
        dist_df = pd.DataFrame(rows).melt(
            id_vars="Source", var_name="Emotion", value_name="Count"
        )
        st.plotly_chart(
            px.bar(
                dist_df, x="Source", y="Count", color="Emotion",
                barmode="group", title="🔎 Emotion Distribution: Subreddits vs My Emotion"
            ),
            use_container_width=True,
        )

    else:
        st.info("ℹ️ Volatility data not available yet. Fetch sentiment first.")





