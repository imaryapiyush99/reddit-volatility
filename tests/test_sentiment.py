from utils.sentiment import analyze_text

def test_positive_sentiment():
    text = "I am very happy and excited!"
    score = analyze_text(text)
    assert score > 0, "Positive text should have positive score"

def test_negative_sentiment():
    text = "This is terrible and depressing."
    score = analyze_text(text)
    assert score < 0, "Negative text should have negative score"

def test_neutral_sentiment():
    text = "The sky is blue."
    score = analyze_text(text)
    assert -0.2 < score < 0.2, "Neutral text should be near zero"
