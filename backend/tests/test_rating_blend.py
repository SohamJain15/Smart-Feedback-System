"""The star rating nudges sentiment/priority, but the text model stays the main signal."""
import numpy as np

from app.ml_service import blend_with_rating

SENTIMENT = np.array(["negative", "neutral", "positive"])


def test_no_rating_leaves_probabilities_unchanged():
    probabilities = np.array([0.3, 0.3, 0.4])
    assert np.allclose(blend_with_rating("sentiment", SENTIMENT, probabilities, None), probabilities)


def test_topic_ignores_rating():
    probabilities = np.array([0.5, 0.5])
    assert np.allclose(blend_with_rating("topic", np.array(["hostel", "library"]), probabilities, 1), probabilities)


def test_low_rating_tips_an_uncertain_prediction_to_negative():
    uncertain = np.array([0.35, 0.23, 0.42])  # model slightly prefers positive
    blended = blend_with_rating("sentiment", SENTIMENT, uncertain, 1)
    assert SENTIMENT[blended.argmax()] == "negative"
    assert np.isclose(blended.sum(), 1.0)


def test_confident_text_still_wins_over_a_contradicting_rating():
    confident_positive = np.array([0.05, 0.10, 0.85])
    blended = blend_with_rating("sentiment", SENTIMENT, confident_positive, 1)
    assert SENTIMENT[blended.argmax()] == "positive"
