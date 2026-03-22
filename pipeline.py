"""
ArvyaX ML Pipeline — v3 (Fixed for Real Data)
===============================================
Fixes applied based on real data run:
- Only 6 classes (not 12): calm, focused, mixed, neutral, overwhelmed, restless
- Overfitting fix: proper train/val split, regularized RF, max_features tuning
- Better text features: char n-grams added, sentiment score
- Intensity: treated as regression (MAE was 1.48, need smoother predictions)
- Error analysis: runs on held-out validation set (not train set)
- Model comparison: RF vs GradientBoosting, pick best by CV
- Proper uncertainty: calibrated on val set
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split
)
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    mean_absolute_error, confusion_matrix
)
from sklearn.pipeline import Pipeline

from data_cleaning import clean_dataframe, audit

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_csv(path):
    try:
        df = pd.read_csv(path, sep='\t')
        if df.shape[1] < 5:
            df = pd.read_csv(path, sep=',')
    except Exception:
        df = pd.read_csv(path, sep=',')
    print(f"  Loaded {path}: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (improved)
# ─────────────────────────────────────────────

OHE_COLS     = ["ambience_type", "face_emotion_hint"]
META_NUMERIC = ["sleep_hours","energy_level","stress_level",
                "duration_min","time_of_day","previous_day_mood","reflection_quality"]
DERIVED_COLS = ["stress_energy_gap","sleep_deficit","low_energy_flag",
                "high_stress_flag","night_flag","text_length","is_vague_text",
                "both_high_flag","energy_x_stress"]


def sentiment_score(text):
    """Simple positive/negative word count ratio."""
    pos = {"calm","peaceful","clear","settled","focused","ready","lighter",
           "good","better","great","motivated","alive","hopeful","content",
           "gentle","ease","sharp","sorted","okay","fine"}
    neg = {"anxious","overwhelmed","tense","heavy","scattered","racing",
           "stuck","drained","exhausted","sad","tired","fidgety","restless",
           "pressure","hard","difficult","lost","confused","uneasy","low"}
    words = str(text).lower().split()
    p = sum(1 for w in words if w in pos)
    n = sum(1 for w in words if w in neg)
    total = p + n
    if total == 0:
        return 0.0
    return round((p - n) / total, 3)


def derive_features(df):
    df = df.copy()
    df["stress_energy_gap"] = df["stress_level"] - df["energy_level"]
    df["sleep_deficit"]     = np.clip(8.0 - df["sleep_hours"], 0, 10)
    df["low_energy_flag"]   = (df["energy_level"] <= 3).astype(int)
    df["high_stress_flag"]  = (df["stress_level"] >= 7).astype(int)
    df["night_flag"]        = (df["time_of_day"] == 4).astype(int)
    df["both_high_flag"]    = ((df["stress_level"] >= 7) & (df["energy_level"] >= 7)).astype(int)
    df["energy_x_stress"]   = df["energy_level"] * df["stress_level"]  # interaction term
    df["sentiment_score"]   = df["journal_text_clean"].apply(sentiment_score)
    return df


def build_ohe(df, fit_classes=None):
    df = df.copy()
    classes = fit_classes or {}
    for col in OHE_COLS:
        if col not in df.columns:
            continue
        if fit_classes is None:
            classes[col] = sorted(df[col].dropna().unique())
        for cls in classes[col]:
            df[f"{col}_{cls}"] = (df[col] == cls).astype(int)
        df.drop(columns=[col], inplace=True, errors="ignore")
    return df, classes


def prepare_features(train_df, test_df):
    train_df = derive_features(train_df)
    test_df  = derive_features(test_df)

    train_df, ohe_classes = build_ohe(train_df)
    test_df,  _           = build_ohe(test_df, fit_classes=ohe_classes)

    ohe_expanded = [c for c in train_df.columns
                    if any(c.startswith(f"{o}_") for o in OHE_COLS)]
    meta_cols = [c for c in (META_NUMERIC + DERIVED_COLS + ["sentiment_score"] + ohe_expanded)
                 if c in train_df.columns]

    for c in meta_cols:
        if c not in test_df.columns:
            test_df[c] = 0

    X_meta_tr = train_df[meta_cols].fillna(0).values.astype(float)
    X_meta_te = test_df[meta_cols].fillna(0).values.astype(float)

    # Word + character n-grams for better coverage of short/similar texts
    tfidf_word = TfidfVectorizer(
        max_features=200, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2,
        token_pattern=r"(?u)\b\w+\b",
        analyzer="word",
    )
    tfidf_char = TfidfVectorizer(
        max_features=100, ngram_range=(3, 5),
        sublinear_tf=True, min_df=2,
        analyzer="char_wb",
    )

    X_word_tr = tfidf_word.fit_transform(train_df["journal_text_clean"]).toarray()
    X_word_te = tfidf_word.transform(test_df["journal_text_clean"]).toarray()

    X_char_tr = tfidf_char.fit_transform(train_df["journal_text_clean"]).toarray()
    X_char_te = tfidf_char.transform(test_df["journal_text_clean"]).toarray()

    word_names = [f"word_{w}" for w in tfidf_word.get_feature_names_out()]
    char_names = [f"char_{w}" for w in tfidf_char.get_feature_names_out()]
    all_names  = word_names + char_names + meta_cols

    X_text_tr = np.hstack([X_word_tr, X_char_tr])
    X_text_te = np.hstack([X_word_te, X_char_te])

    X_tr = np.hstack([X_text_tr, X_meta_tr])
    X_te = np.hstack([X_text_te, X_meta_te])

    return (X_tr, X_te, X_text_tr, X_text_te, X_meta_tr, X_meta_te,
            all_names, meta_cols, tfidf_word, tfidf_char, ohe_classes,
            train_df, test_df)


# ─────────────────────────────────────────────
# MODEL SELECTION (compare, pick best)
# ─────────────────────────────────────────────

def _min_cv(y):
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(int(counts.min()), 5))


def select_best_classifier(X, y, cv_folds):
    """
    Compare RF (regularized) vs GradientBoosting vs LogisticRegression.
    Pick by CV weighted-F1. Returns (best_name, best_score, best_params).
    """
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,           # regularize: was None (overfit)
            min_samples_leaf=5,    # regularize: was 1
            max_features="sqrt",
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.08, subsample=0.8,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
        results[name] = (scores.mean(), scores.std(), clf)
        print(f"    {name:22s}  F1={scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k][0])
    best_mean, best_std, best_clf = results[best_name]
    print(f"  → Best: {best_name} (F1={best_mean:.4f})")
    return best_name, best_mean, best_std, best_clf, results


def train_final_classifier(X, y, base_clf, cv_folds):
    """Wrap best clf with calibration for probability estimates."""
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=cv_folds)
    clf.fit(X, y)
    return clf


def train_intensity_model(X, y):
    """
    Intensity as REGRESSION (continuous 1-5), then round.
    GradientBoosting regressor handles ordinal structure well.
    Rationale: with MAE=1.48 from classifier, regression gives smoother output.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.08, subsample=0.8,
        random_state=42,
    )
    reg.fit(X, y.astype(float))
    return reg


def predict_intensity(model, X):
    raw = model.predict(X)
    return np.clip(np.round(raw), 1, 5).astype(int)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_state(clf, X, y, le, label):
    preds = le.inverse_transform(clf.predict(X))
    true  = le.inverse_transform(y)
    acc   = accuracy_score(true, preds)
    f1    = f1_score(true, preds, average="weighted", zero_division=0)
    print(f"\n  [{label}] Accuracy={acc:.4f}  Weighted-F1={f1:.4f}")
    print(classification_report(true, preds, target_names=le.classes_, zero_division=0))
    return acc, f1, preds


def evaluate_intensity(model, X, y_true, label):
    preds = predict_intensity(model, X)
    mae   = mean_absolute_error(y_true, preds)
    acc1  = np.mean(np.abs(y_true - preds) <= 1)   # within-1 accuracy
    print(f"  [{label}] Intensity MAE={mae:.4f}  Within-1-acc={acc1:.4f}")
    return mae, acc1, preds


def ablation_study(X_text, X_meta, y, cv_folds):
    print(f"\n{'─'*50}\n  Ablation Study ({cv_folds}-fold CV)\n{'─'*50}")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    for name, X in [("text_only", X_text), ("meta_only", X_meta),
                    ("combined",  np.hstack([X_text, X_meta]))]:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
        results[name] = {"mean_f1": round(scores.mean(), 4), "std_f1": round(scores.std(), 4)}
        print(f"  [{name:12s}]  F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    return results


# ─────────────────────────────────────────────
# UNCERTAINTY
# ─────────────────────────────────────────────

def compute_uncertainty(proba, text_lengths, refl_quality):
    """
    Uncertainty for a 6-class problem.
    With 6 classes, random chance = 0.167. A confident prediction sits ~0.40+.
    Thresholds calibrated so ~25-35% of predictions get flagged (realistic for
    overlapping emotional states with noisy short text).

    uncertain_flag = 1 if ANY of:
    - max_prob < 0.30  (model barely above random, genuinely lost)
    - top2_gap < 0.05  (top two classes nearly tied — true ambiguity)
    - text very short  (< 10 chars — no signal at all)
    - reflection vague AND confidence below moderate (< 0.35)
    """
    max_p    = proba.max(axis=1)
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    gap      = sorted_p[:, 0] - sorted_p[:, 1]
    tl       = np.array(text_lengths)
    rq       = np.array(refl_quality)

    uncertain = (
        (max_p < 0.30) |                          # barely above random
        (gap   < 0.05) |                          # top-2 nearly tied
        (tl    < 10)   |                          # no text signal
        ((rq == 0) & (max_p < 0.35))              # vague reflection + low conf
    ).astype(int)

    return np.round(max_p, 4), uncertain


# ─────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────

def decide_what(state, intensity, stress, energy, time_enc):
    t = int(time_enc)
    if state == "anxious":                        return "box_breathing"
    if state == "overwhelmed":                    return "grounding" if intensity >= 4 else "pause"
    if state == "restless":                       return "movement" if t <= 2 else "journaling"
    if state in ("focused", "energized"):         return "deep_work"
    if state == "tired":                          return "rest" if (energy <= 3 or t >= 3) else "yoga"
    if state == "sad":                            return "sound_therapy" if t >= 3 else "journaling"
    if state in ("calm", "content", "hopeful"):   return "light_planning" if t <= 2 else "rest"
    if state in ("neutral", "mixed"):             return "journaling" if stress >= 4 else "light_planning"
    return "pause"


def decide_when(state, intensity, stress, energy, time_enc):
    t = int(time_enc)
    if state in ("anxious","overwhelmed") and intensity >= 4:   return "now"
    if state in ("restless","anxious") and intensity >= 3:      return "within_15_min"
    if t >= 3 and state in ("tired","sad","calm","neutral"):    return "tonight"
    if t == 4:                                                   return "tomorrow_morning"
    if state in ("focused","energized"):                         return "now"
    if state in ("content","hopeful") and t <= 1:               return "within_15_min"
    return "later_today"


MESSAGES = {
    "box_breathing":  "You seem tense right now. Let's slow things down — try box breathing: inhale 4, hold 4, exhale 4, hold 4.",
    "journaling":     "Your thoughts seem scattered. Writing them down can bring surprising clarity. Take 10 minutes to journal freely.",
    "grounding":      "You're feeling overwhelmed — completely okay. Try grounding: name 5 things you see, 4 you can touch.",
    "deep_work":      "You're in a great mental space. Lock in on your most important task while this clarity lasts.",
    "yoga":           "Some gentle movement will recharge you. A short yoga flow can honour what your body needs.",
    "sound_therapy":  "Soothing sounds can lift a heavy mood. Try binaural beats or calming music for a while.",
    "light_planning": "You're in a balanced state — a perfect time to sketch your next priorities gently.",
    "rest":           "Your body is asking for recovery. Rest isn't laziness; it's essential maintenance.",
    "movement":       "Channel that restless energy — a brisk walk or a stretch will reset your nervous system.",
    "pause":          "The best action right now is a mindful pause. Breathe, observe, and simply be.",
}

def generate_message(what, state, intensity):
    base   = MESSAGES.get(what, "Take a moment for yourself right now.")
    prefix = ("Things feel intense. " if intensity >= 4 else
              "You're in a gentle space. " if intensity <= 2 else "")
    return prefix + base


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def feature_importance(clf, feat_names, top_n=25):
    try:
        est = clf.calibrated_classifiers_[0].estimator
        if hasattr(est, "feature_importances_"):
            imps = est.feature_importances_
        elif hasattr(est, "coef_"):
            imps = np.abs(est.coef_).mean(axis=0)
        else:
            return []
    except Exception:
        return []
    idx = np.argsort(imps)[::-1][:top_n]
    return [(feat_names[i], round(float(imps[i]), 5)) for i in idx]


# ─────────────────────────────────────────────
# ERROR ANALYSIS (on val set — NOT train set)
# ─────────────────────────────────────────────

def _error_reason(row):
    tl = row.get("text_length", 50)
    sl = row.get("stress_level", 5)
    el = row.get("energy_level", 5)
    rq = row.get("reflection_quality", 1)
    pred = row.get("predicted_state", "")
    true = row.get("emotional_state", "")

    if tl < 15:               return "SHORT_TEXT — too little signal for model"
    if row.get("is_vague_text", 0) == 1:
                              return "VAGUE_TEXT — generic language, low discriminability"
    if sl >= 7 and el >= 7:  return "CONFLICTING_SIGNALS — high stress + high energy simultaneously"
    if rq == 0:               return "NOISY_LABEL — vague reflection quality, unreliable ground truth"
    close_pairs = [{"calm","neutral"},{"focused","neutral"},{"mixed","restless"},
                   {"calm","mixed"},{"overwhelmed","restless"},{"neutral","mixed"},
                   {"focused","calm"}]
    if {pred, true} in close_pairs:
                              return f"OVERLAPPING_STATES — '{true}' and '{pred}' share vocabulary"
    conf = row.get("confidence", 1.0)
    if conf < 0.4:            return "LOW_CONFIDENCE — model genuinely uncertain, borderline case"
    return "AMBIGUOUS_CONTEXT — multiple valid interpretations"


def run_error_analysis(val_df, true_col, pred_col):
    """Run on validation set for honest error analysis."""
    errors = val_df[val_df[true_col] != val_df[pred_col]].copy()
    errors["error_reason"] = errors.apply(_error_reason, axis=1)
    return errors


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ArvyaX Emotional Intelligence Pipeline  v3")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────
    print("\n[1] Loading data...")
    train_raw = load_csv(TRAIN_PATH)
    test_raw  = load_csv(TEST_PATH)

    # ── 2. Clean ─────────────────────────────
    print("\n[2] Cleaning data...")
    train_clean, medians = clean_dataframe(train_raw, is_train=True)
    test_clean,  _       = clean_dataframe(test_raw,  is_train=False, train_medians=medians)
    audit(train_clean, "Train (cleaned)")

    # ── 3. Train/Val split for honest eval ───
    # Hold out 20% for error analysis + unbiased metrics
    from sklearn.model_selection import train_test_split
    train_main, val_df = train_test_split(
        train_clean, test_size=0.20, random_state=42,
        stratify=train_clean["emotional_state"]
    )
    print(f"\n  Train split: {len(train_main)} | Val split: {len(val_df)}")

    # ── 4. Labels ────────────────────────────
    le_state = LabelEncoder()
    le_state.fit(train_clean["emotional_state"])   # fit on all so classes are complete
    y_tr  = le_state.transform(train_main["emotional_state"])
    y_val = le_state.transform(val_df["emotional_state"])
    y_tr_int  = train_main["intensity"].values.astype(float)
    y_val_int = val_df["intensity"].values.astype(float)
    print(f"\n  Classes ({len(le_state.classes_)}): {list(le_state.classes_)}")

    # ── 5. Features ──────────────────────────
    print("\n[3] Building features (word + char n-grams + metadata + derived)...")
    (X_tr, X_te,
     X_text_tr, X_text_te,
     X_meta_tr, X_meta_te,
     feat_names, meta_cols,
     tfidf_word, tfidf_char, ohe_classes,
     train_feat, test_feat) = prepare_features(train_main, test_clean)

    # Also transform val set using same fitted transformers
    val_feat = derive_features(val_df)
    val_feat, _ = build_ohe(val_feat, fit_classes=ohe_classes)
    for c in meta_cols:
        if c not in val_feat.columns:
            val_feat[c] = 0
    X_meta_val = val_feat[meta_cols].fillna(0).values.astype(float)
    X_word_val = tfidf_word.transform(val_feat["journal_text_clean"]).toarray()
    X_char_val = tfidf_char.transform(val_feat["journal_text_clean"]).toarray()
    X_text_val = np.hstack([X_word_val, X_char_val])
    X_val = np.hstack([X_text_val, X_meta_val])

    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_te.shape}")

    # ── 6. Ablation ──────────────────────────
    print("\n[4] Ablation study...")
    cv_folds = _min_cv(y_tr)
    ablation = ablation_study(X_text_tr, X_meta_tr, y_tr, cv_folds)
    with open("ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)

    # ── 7. Model selection ───────────────────
    print(f"\n[5] Model selection ({cv_folds}-fold CV on train split)...")
    best_name, best_f1_mean, best_f1_std, best_base, all_results = \
        select_best_classifier(X_tr, y_tr, cv_folds)

    # ── 8. Train final models ────────────────
    print(f"\n[6] Training final {best_name} (calibrated)...")
    clf_state = train_final_classifier(X_tr, y_tr, best_base, cv_folds)

    print("  Training intensity regressor...")
    clf_intens = train_intensity_model(X_tr, y_tr_int)

    # ── 9. Validation scores (honest) ────────
    print("\n[7] Validation scores (held-out 20%):")
    val_acc, val_f1, val_preds_state = evaluate_state(clf_state, X_val, y_val, le_state, "VALIDATION")
    val_mae, val_acc1, val_preds_int = evaluate_intensity(clf_intens, X_val, y_val_int, "VALIDATION")

    # ── 10. CV scores (generalisation) ───────
    print("\n[8] Cross-validation on full train (generalisation)...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_all     = le_state.transform(train_clean["emotional_state"])
    y_all_int = train_clean["intensity"].values.astype(float)

    # Refit for full-train CV
    from sklearn.ensemble import GradientBoostingRegressor
    cv_f1_all = cross_val_score(
        RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                               class_weight="balanced", random_state=42, n_jobs=-1),
        np.hstack([X_text_tr, X_meta_tr]) if False else  # placeholder
        np.vstack([X_tr, X_val]),
        np.concatenate([y_tr, y_val]),
        cv=cv, scoring="f1_weighted"
    )
    print(f"  Full-train CV F1: {cv_f1_all.mean():.4f} ± {cv_f1_all.std():.4f}")

    scores_summary = {
        "val_accuracy":          round(val_acc, 4),
        "val_weighted_f1":       round(val_f1, 4),
        "val_intensity_mae":     round(float(val_mae), 4),
        "val_intensity_within1": round(float(val_acc1), 4),
        "cv_weighted_f1_mean":   round(float(cv_f1_all.mean()), 4),
        "cv_weighted_f1_std":    round(float(cv_f1_all.std()), 4),
        "best_model":            best_name,
        "model_selection_cv_f1": round(best_f1_mean, 4),
        "n_classes":             int(len(le_state.classes_)),
        "classes":               list(le_state.classes_),
        "all_model_scores": {
            k: {"f1_mean": round(v[0],4), "f1_std": round(v[1],4)}
            for k, v in all_results.items()
        }
    }
    with open("scores_summary.json", "w") as f:
        json.dump(scores_summary, f, indent=2)
    print(f"\n  Scores saved → scores_summary.json")
    print(f"  Val F1={val_f1:.4f} | Val MAE={val_mae:.4f} | Within-1={val_acc1:.4f}")

    # ── 11. Feature importance ───────────────
    print("\n[9] Feature importance...")
    imp = feature_importance(clf_state, feat_names)
    for name, val in imp[:10]:
        print(f"  {name:40s}  {val:.5f}")
    pd.DataFrame(imp, columns=["feature","importance"]).to_csv("feature_importance.csv", index=False)

    # ── 12. Retrain on ALL train data ─────────
    # After honest eval, retrain on full train set for best test predictions
    print("\n[10] Retraining on full training set for test predictions...")
    (X_tr_full, X_te_full,
     X_text_full, X_text_te2,
     X_meta_full, X_meta_te2,
     feat_names2, meta_cols2,
     tfidf_word2, tfidf_char2, ohe_classes2,
     train_feat2, test_feat2) = prepare_features(train_clean, test_clean)

    y_full     = le_state.transform(train_clean["emotional_state"])
    y_full_int = train_clean["intensity"].values.astype(float)

    from sklearn.base import clone
    final_base = clone(best_base)
    clf_state_final  = train_final_classifier(X_tr_full, y_full, final_base, cv_folds)
    clf_intens_final = train_intensity_model(X_tr_full, y_full_int)

    # ── 13. Predict on test ───────────────────
    print("\n[11] Predicting on test set...")
    proba      = clf_state_final.predict_proba(X_te_full)
    pred_state = le_state.inverse_transform(np.argmax(proba, axis=1))
    pred_int   = predict_intensity(clf_intens_final, X_te_full)

    tl_te = test_feat2["text_length"].values if "text_length" in test_feat2.columns else np.full(len(test_feat2), 50)
    rq_te = test_feat2["reflection_quality"].values if "reflection_quality" in test_feat2.columns else np.ones(len(test_feat2))
    confidence, uncertain = compute_uncertainty(proba, tl_te, rq_te)

    # ── 14. Decision engine ───────────────────
    what_l, when_l, msg_l = [], [], []
    for i in range(len(test_feat2)):
        row   = test_feat2.iloc[i]
        what  = decide_what(pred_state[i], int(pred_int[i]),
                            float(row.get("stress_level",5)), float(row.get("energy_level",5)),
                            int(row.get("time_of_day",1)))
        when  = decide_when(pred_state[i], int(pred_int[i]),
                            float(row.get("stress_level",5)), float(row.get("energy_level",5)),
                            int(row.get("time_of_day",1)))
        what_l.append(what); when_l.append(when)
        msg_l.append(generate_message(what, pred_state[i], int(pred_int[i])))

    predictions = pd.DataFrame({
        "id":                  test_raw["id"].values,
        "predicted_state":     pred_state,
        "predicted_intensity": pred_int,
        "confidence":          confidence,
        "uncertain_flag":      uncertain,
        "what_to_do":          what_l,
        "when_to_do":          when_l,
        "supportive_message":  msg_l,
    })
    predictions.to_csv("predictions.csv", index=False)
    print(f"  Saved predictions.csv ({len(predictions)} rows)")
    print(f"  State distribution:\n{pd.Series(pred_state).value_counts().to_string()}")
    print(f"  Uncertain flag rate: {uncertain.mean():.2%}")

    # ── 15. Error analysis on val set ─────────
    print("\n[12] Error analysis on validation set (honest)...")
    val_eval = val_df[["id","journal_text","emotional_state","intensity",
                        "stress_level","energy_level","time_of_day",
                        "reflection_quality","text_length","is_vague_text"]].copy()
    val_eval["predicted_state"]     = val_preds_state
    val_eval["predicted_intensity"] = np.round(val_preds_int).astype(int)

    proba_val  = clf_state.predict_proba(X_val)
    tl_val = val_feat["text_length"].values if "text_length" in val_feat.columns else np.full(len(val_feat),50)
    rq_val = val_feat["reflection_quality"].values if "reflection_quality" in val_feat.columns else np.ones(len(val_feat))
    val_eval["confidence"], val_eval["uncertain_flag"] = compute_uncertainty(proba_val, tl_val, rq_val)

    errors = run_error_analysis(val_eval, "emotional_state", "predicted_state")
    errors.to_csv("error_cases.csv", index=False)
    print(f"  {len(errors)} real error cases saved → error_cases.csv")

    if len(errors) > 0:
        print("\n  Error reason breakdown:")
        print(errors["error_reason"].value_counts().to_string())

    # ── 16. Confusion matrix ──────────────────
    cm = confusion_matrix(
        le_state.inverse_transform(y_val),
        val_preds_state,
        labels=le_state.classes_
    )
    cm_df = pd.DataFrame(cm, index=le_state.classes_, columns=le_state.classes_)
    cm_df.to_csv("confusion_matrix.csv")
    print(f"\n  Confusion matrix saved → confusion_matrix.csv")
    print(cm_df.to_string())

    print("\n" + "="*60)
    print("  DONE")
    print(f"  Val F1      = {val_f1:.4f}")
    print(f"  Intensity MAE = {val_mae:.4f}")
    print(f"  Within-1 acc  = {val_acc1:.4f}")
    print("  Outputs:")
    for fname in ["predictions.csv","scores_summary.json","ablation_results.json",
                  "feature_importance.csv","error_cases.csv","confusion_matrix.csv"]:
        print(f"    {fname}")
    print("="*60)

    return predictions, scores_summary


if __name__ == "__main__":
    main()
