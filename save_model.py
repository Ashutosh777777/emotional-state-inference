"""
Run this AFTER pipeline.py to save model bundle for the Flask API.
Usage: python save_model.py
"""
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# Re-use functions from pipeline
from pipeline import (
    load_data, engineer_features, get_text_vectorizer,
    train_state_classifier, train_intensity_model, ORDINAL_MAPS, CATEGORICAL_COLS
)

print("Loading data...")
train, test = load_data()

le_state     = LabelEncoder()
y_state      = le_state.fit_transform(train["emotional_state"])
y_intensity  = train["intensity"].values.astype(int)

print("Engineering features...")
meta_train, encoders = engineer_features(train)
X_text_train, tfidf  = get_text_vectorizer(train["journal_text"], fit=True)
X_train = np.hstack([X_text_train, meta_train.values])

print("Training models...")
clf_state     = train_state_classifier(X_train, y_state)
clf_intensity = train_intensity_model(X_train, y_intensity)

bundle = {
    "clf_state":     clf_state,
    "clf_intensity": clf_intensity,
    "le_state":      le_state,
    "tfidf":         tfidf,
    "encoders":      encoders,
    "meta_cols":     list(meta_train.columns),
}

with open("model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("✅ model_bundle.pkl saved!")
