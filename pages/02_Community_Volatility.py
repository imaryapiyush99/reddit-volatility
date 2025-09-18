import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🌍 Community Emotional Volatility")

# --- Load community data from session state ---
df_comm = st.session_state.get("df_comm", pd.DataFrame())

# --- Check data availability ---
if df_comm.empty:
    st.warning("⚠️ No community data available. Please fetch subreddit posts first.")
elif "subreddit" not in df_comm.columns:
    st.error("❌ Community dataset has no 'subreddit' column. Please refetch data.")
    st.stop()
else:
    # Ensure datetime and sort
    df_comm["time"] = pd.to_datetime(df_comm["time"])
    df_comm = df_comm.sort_values("time")

    # Compute volatility if missing
    if "volatility" not in df_comm.columns:
        df_comm["volatility"] = df_comm.groupby("subreddit")["sentiment_score"].transform(
            lambda x: x.rolling(5).std().fillna(0)
        )

    # --- Sentiment over time ---
    fig_sent = px.line(
        df_comm,
        x="time",
        y="sentiment_score",
        color="subreddit",
        title="📈 Community Sentiment Over Time",
        markers=True
    ).update_yaxes(range=[-1, 1])
    st.plotly_chart(fig_sent, use_container_width=True)

    # --- Volatility over time ---
    fig_vol = px.line(
        df_comm,
        x="time",
        y="volatility",
        color="subreddit",
        title="🌪 Community Volatility Over Time",
        markers=True
    )
    st.plotly_chart(fig_vol, use_container_width=True)


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
