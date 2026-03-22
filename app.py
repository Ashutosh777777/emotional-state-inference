"""
ArvyaX Prediction API — Flask
Run: python app.py
POST /predict with JSON body → returns full emotional intelligence response
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

# Delay heavy imports until needed
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Globals (loaded at startup) ──────────────────────────────────────────────
MODEL = None

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    try:
        with open("model_bundle.pkl", "rb") as f:
            MODEL = pickle.load(f)
        print("[API] Model bundle loaded.")
    except FileNotFoundError:
        print("[API] model_bundle.pkl not found — run: python save_model.py")
        sys.exit(1)
    return MODEL


# ── Helpers (mirrored from pipeline.py) ──────────────────────────────────────
ORDINAL_MAPS = {
    "time_of_day":       {"morning": 0, "afternoon": 1, "evening": 2, "night": 3},
    "previous_day_mood": {"negative": 0, "mixed": 1, "neutral": 2, "positive": 3},
    "reflection_quality":{"low": 0, "medium": 1, "high": 2},
}
CATEGORICAL_COLS = ["ambience_type", "face_emotion_hint"]

MESSAGES = {
    "box_breathing":  "You seem tense right now. Try box breathing: inhale 4, hold 4, exhale 4, hold 4.",
    "journaling":     "Your thoughts seem scattered. Writing them down can bring clarity.",
    "grounding":      "Feeling overwhelmed? Name 5 things you can see, 4 you can touch.",
    "deep_work":      "You're in a great mental space — lock in on your most important task.",
    "yoga":           "Some gentle movement will recharge you beautifully.",
    "sound_therapy":  "Soothing sounds can lift a heavy mood. Try calming music or binaural beats.",
    "light_planning": "Perfect time to sketch out your next few priorities.",
    "rest":           "Your body is asking for recovery. Rest is maintenance, not laziness.",
    "movement":       "Channel that restless energy — a brisk walk will reset your nervous system.",
    "pause":          "The best action right now is a mindful pause. Breathe and observe.",
}


def preprocess_single(data: dict, bundle: dict):
    """Convert raw request dict to feature vector."""
    df = pd.DataFrame([data])
    encoders = bundle["encoders"]

    # Numerical fill
    num_cols = ["sleep_hours", "energy_level", "stress_level", "duration_min"]
    for c in num_cols:
        if c not in df.columns or pd.isna(df[c].iloc[0]):
            df[c] = encoders["medians"].get(c, 5)

    # Ordinal
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(1)
        else:
            df[col] = 1

    # OHE
    for col in CATEGORICAL_COLS:
        for cls in encoders["ohe"].get(col, []):
            df[f"{col}_{cls}"] = (df.get(col, "") == cls).astype(int)
        df.drop(columns=[col], inplace=True, errors="ignore")

    # Derived
    df["stress_energy_gap"] = df["stress_level"] - df["energy_level"]
    df["sleep_deficit"]     = np.clip(8.0 - df["sleep_hours"], 0, 10)
    df["low_energy_flag"]   = (df["energy_level"] <= 3).astype(int)
    df["high_stress_flag"]  = (df["stress_level"] >= 7).astype(int)
    df["night_flag"]        = (df["time_of_day"] == 3).astype(int)

    # Text
    journal = data.get("journal_text", "")
    X_text  = bundle["tfidf"].transform([journal]).toarray()

    # Align meta columns
    meta_cols = bundle["meta_cols"]
    for c in meta_cols:
        if c not in df.columns:
            df[c] = 0
    meta_vec = df[meta_cols].values

    return np.hstack([X_text, meta_vec])


def decide(state, intensity, stress, energy, time_enc):
    t = int(time_enc)
    if state == "anxious":          what = "box_breathing"
    elif state == "overwhelmed":    what = "grounding" if intensity >= 4 else "pause"
    elif state == "restless":       what = "movement" if t <= 1 else "journaling"
    elif state in ["focused","energized"]: what = "deep_work"
    elif state == "tired":          what = "rest"
    elif state == "sad":            what = "sound_therapy" if t == 3 else "journaling"
    elif state == "calm":           what = "light_planning"
    elif state == "content":        what = "light_planning" if t <= 1 else "rest"
    elif state == "hopeful":        what = "light_planning"
    else:                           what = "pause"

    urgent = state in ["anxious","overwhelmed"] and intensity >= 4
    if urgent:                      when = "now"
    elif state in ["restless","anxious"] and intensity >= 3: when = "within_15_min"
    elif t in [2,3] and state in ["tired","sad","calm"]:     when = "tonight"
    elif t == 3:                    when = "tomorrow_morning"
    elif state in ["focused","energized"]: when = "now"
    else:                           when = "later_today"

    return what, when


@app.route("/predict", methods=["POST"])
def predict():
    bundle = load_model()
    data   = request.json or {}

    try:
        X = preprocess_single(data, bundle)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 400

    proba      = bundle["clf_state"].predict_proba(X)[0]
    pred_idx   = int(np.argmax(proba))
    pred_state = bundle["le_state"].inverse_transform([pred_idx])[0]
    confidence = round(float(proba.max()), 4)
    top2_gap   = sorted(proba)[-1] - sorted(proba)[-2]
    text_len   = len(str(data.get("journal_text", "")))
    uncertain  = int(confidence < 0.45 or top2_gap < 0.10 or text_len < 10)

    pred_int   = int(bundle["clf_intensity"].predict(X)[0])
    stress     = float(data.get("stress_level", 5))
    energy     = float(data.get("energy_level", 5))
    tod        = ORDINAL_MAPS["time_of_day"].get(data.get("time_of_day","morning"), 0)

    what, when = decide(pred_state, pred_int, stress, energy, tod)
    msg_base   = MESSAGES.get(what, "Take a moment for yourself.")
    prefix     = "Things feel intense. " if pred_int >= 4 else ("Gentle space. " if pred_int <= 2 else "")
    message    = prefix + msg_base

    return jsonify({
        "predicted_state":     pred_state,
        "predicted_intensity": pred_int,
        "confidence":          confidence,
        "uncertain_flag":      uncertain,
        "what_to_do":          what,
        "when_to_do":          when,
        "supportive_message":  message,
        "top_probabilities":   {
            bundle["le_state"].classes_[i]: round(float(proba[i]), 3)
            for i in np.argsort(proba)[::-1][:3]
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


if __name__ == "__main__":
    load_model()
    print("[API] Starting on http://localhost:5000")
    app.run(debug=False, port=5000)
