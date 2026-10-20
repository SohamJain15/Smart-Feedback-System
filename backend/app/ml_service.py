from pathlib import Path
import os
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "training_data.csv"
MODELS = Path(os.environ.get("SMART_FEEDBACK_MODELS", ROOT / "models"))
TARGETS = ["sentiment", "topic", "priority"]

# The star rating is a weak signal from the user about how they felt. The text model stays the main
# signal; the rating only shifts its probabilities, so clear text still wins over a contradicting rating.
RATING_PRIORS = {
    "sentiment": {1: {"negative": .75, "neutral": .15, "positive": .10},
                  2: {"negative": .60, "neutral": .30, "positive": .10},
                  3: {"negative": .25, "neutral": .50, "positive": .25},
                  4: {"negative": .10, "neutral": .25, "positive": .65},
                  5: {"negative": .05, "neutral": .15, "positive": .80}},
    "priority": {1: {"critical": .20, "high": .40, "medium": .30, "low": .10},
                 2: {"critical": .10, "high": .30, "medium": .45, "low": .15},
                 3: {"critical": .05, "high": .15, "medium": .45, "low": .35},
                 4: {"critical": .02, "high": .08, "medium": .25, "low": .65},
                 5: {"critical": .02, "high": .03, "medium": .15, "low": .80}},
}
RATING_WEIGHT = {"sentiment": 0.4, "priority": 0.3}


def blend_with_rating(target, classes, probabilities, rating):
    """Weighted average of the model's probabilities and a rating-based prior (unchanged without a rating)."""
    priors = RATING_PRIORS.get(target, {}).get(rating)
    if not priors:
        return probabilities
    prior = np.array([priors.get(str(label), 0.0) for label in classes])
    if prior.sum() == 0:
        return probabilities
    weight = RATING_WEIGHT[target]
    return (1 - weight) * np.asarray(probabilities) + weight * prior / prior.sum()


class MLService:
    def __init__(self):
        self.models = {}
        self.version = None

    def train(self, frame: pd.DataFrame):
        required = {"text", "sentiment", "topic", "priority"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError("Training data is missing required labelled columns")
        if frame["text"].isna().any() or any(frame[target].nunique() < 2 for target in TARGETS):
            raise ValueError("Training data needs non-empty text and at least two labels per classifier")
        trained, metrics = {}, {}
        for target in TARGETS:
            # Topic receives a category context token as well as feedback text.
            # It is still classified by the trained model, not assigned in an if/else rule.
            features = frame["text"] if target != "topic" else (("category_" + frame["topic"] + " ") * 8 + frame["text"])
            x_train, x_test, y_train, y_test = train_test_split(
                features, frame[target], test_size=0.25, random_state=42, stratify=frame[target]
            )
            model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                              ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            p, r, f, _ = precision_recall_fscore_support(y_test, predicted, average="weighted", zero_division=0)
            metrics[target] = {"accuracy": float(accuracy_score(y_test, predicted)), "precision": float(p), "recall": float(r), "f1_score": float(f)}
            trained[target] = model
        return trained, metrics

    def save(self, version, trained):
        target = MODELS / version
        target.mkdir(parents=True, exist_ok=True)
        for name, model in trained.items(): joblib.dump(model, target / f"{name}.joblib")

    def load(self, version):
        folder = MODELS / version
        self.models = {name: joblib.load(folder / f"{name}.joblib") for name in TARGETS}
        self.version = version

    def predict(self, text, category=None, rating=None):
        if not self.models: raise RuntimeError("No active models are available")
        result = {}
        for target, model in self.models.items():
            features = text if target != "topic" else (f"category_{(category or '').lower()} " * 8 + text)
            probabilities = blend_with_rating(target, model.classes_, model.predict_proba([features])[0], rating)
            index = int(probabilities.argmax())
            result[target] = {"label": str(model.classes_[index]), "confidence": float(probabilities[index])}
        return result

    def label_options(self):
        if not self.models:
            raise RuntimeError("No active models are available")
        return {target: [str(value) for value in self.models[target].classes_] for target in TARGETS}

ml = MLService()

def base_dataset(): return pd.read_csv(DATASET)
