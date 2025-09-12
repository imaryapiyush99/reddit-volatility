from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

def analyze_text(text):
    vader_score = sia.polarity_scores(text)["compound"]
    tb_score = TextBlob(text).sentiment.polarity
    return (vader_score + tb_score) / 2
