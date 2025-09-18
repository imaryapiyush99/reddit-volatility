import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# download VADER lexicon once
nltk.download("vader_lexicon", quiet=True)


class SentimentEnsemble:
    """Ensemble sentiment analyzer using VADER + TextBlob + VADER(NLTK) with configurable weights."""

    def __init__(self, w_vader: float = 0.05, w_blob: float = 0.9, w_nltk: float = 0.05):
        self.vader = SentimentIntensityAnalyzer()
        self.nltk_analyzer = SentimentIntensityAnalyzer()
        self.w_vader = w_vader
        self.w_blob = w_blob
        self.w_nltk = w_nltk

    def analyze_text(self, text: str) -> float:
        """Return ensemble sentiment score in [-1, 1]."""
        if not isinstance(text, str) or not text.strip():
            return 0.0

        # VADER
        vader_score = self.vader.polarity_scores(text).get("compound", 0.0)

        # TextBlob
        blob_score = TextBlob(text).sentiment.polarity

        # NLTK (reusing VADER)
        nltk_score = self.nltk_analyzer.polarity_scores(text).get("compound", 0.0)

        # Weighted ensemble
        total_w = self.w_vader + self.w_blob + self.w_nltk
        if total_w == 0:
            return 0.0

        score = (
            vader_score * self.w_vader +
            blob_score * self.w_blob +
            nltk_score * self.w_nltk
        ) / total_w
        return round(float(score), 3)

    def score_to_label(self, score: float) -> str:
        """Convert numeric score → sentiment label."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"


# ---------- Helper function for DataFrames ----------
import pandas as pd
def analyze_sentiment(df, analyzer):
    # Continuous score
    df["sentiment_score"] = df["text"].apply(analyzer.analyze_text)

    # Discrete sentiment (-1, 0, 1)
    df["sentiment"] = df["sentiment_score"].apply(
        lambda s: 1 if s > 0.1 else (-1 if s < -0.1 else 0)
    )

    # Labels for readability
    df["sentiment_label"] = df["sentiment"].map({
        1: "positive",
        0: "neutral",
        -1: "negative"
    })

    return df

# def analyze_sentiment(df: pd.DataFrame, analyzer: SentimentEnsemble) -> pd.DataFrame:
#     if df is None or df.empty or "text" not in df.columns:
#         return df

#     df = df.copy()
#     df["text"] = df["text"].astype(str)

#     # float score
#     df["sentiment_score"] = df["text"].apply(analyzer.analyze_text)

#     # string label
#     df["sentiment_label"] = df["sentiment_score"].apply(analyzer.score_to_label)

#     return df
