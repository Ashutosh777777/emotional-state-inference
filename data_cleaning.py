"""
data_cleaning.py
=================
Handles all real-world data quality issues observed in the ArvyaX dataset:

Issues found in actual data sample:
1. Missing sleep_hours (blank cells)
2. Missing previous_day_mood (blank cells)
3. reflection_quality values: 'clear', 'vague', 'conflicted' (not low/medium/high)
4. time_of_day has 'early_morning' as extra category
5. face_emotion_hint format: 'calm_face', 'tense_face', 'none' (not just emotion names)
6. emotional_state includes 'neutral' and 'mixed'
7. Journal text has noise: trailing 'idk.', 'idk', short vague entries
8. Some duration_min values very small (3 min) — likely valid but note
9. Intensity is integer 1-5
"""

import re
import numpy as np
import pandas as pd


# ── KNOWN VALUE MAPPINGS ──────────────────────────────────────────────────────

REFLECTION_QUALITY_MAP = {
    # Actual values in dataset
    "clear":       2,
    "vague":       0,
    "conflicted":  1,
    # Fallback for synthetic / alternate naming
    "high":        2,
    "medium":      1,
    "low":         0,
}

TIME_OF_DAY_MAP = {
    "early_morning": 0,
    "morning":       1,
    "afternoon":     2,
    "evening":       3,
    "night":         4,
}

PREVIOUS_MOOD_MAP = {
    "negative":    0,
    "mixed":       1,
    "neutral":     2,
    "calm":        2,   # observed in data — treat as neutral
    "focused":     3,   # observed in data — slightly positive
    "positive":    3,
    "overwhelmed": 0,   # observed in data — treat as negative
    "restless":    1,
}

# face_emotion_hint: strip '_face' suffix, map 'none' to neutral
def clean_face_emotion(val):
    if pd.isna(val) or str(val).strip().lower() in ("none", "", "nan"):
        return "neutral"
    val = str(val).lower().replace("_face", "").strip()
    return val  # e.g. calm, tense, tired, happy, neutral, sad


NOISE_PATTERNS = [
    r'\bidk\.?\b',           # idk / idk.
    r'\bi guess\b',
    r'\bwhatever\b',
    r'\buh\b',
    r'\bhmm+\b',
    r'\s+',                  # normalize whitespace (keep last)
]


# ── TEXT CLEANING ─────────────────────────────────────────────────────────────

def clean_text(text):
    if pd.isna(text):
        return "no reflection provided"
    text = str(text).strip().lower()
    # Remove trailing noise tokens
    for pat in NOISE_PATTERNS[:-1]:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    text = re.sub(NOISE_PATTERNS[-1], ' ', text).strip()
    # If text is now very short (< 5 chars), mark explicitly
    if len(text) < 5:
        text = "no clear reflection"
    return text


def text_length(text):
    return len(str(text).strip())


def is_vague_text(text):
    """True if text is too short or contains only noise words."""
    cleaned = clean_text(text)
    return int(len(cleaned.split()) <= 3)


# ── NUMERICAL CLEANING ────────────────────────────────────────────────────────

def impute_numerics(df: pd.DataFrame, medians: dict = None) -> (pd.DataFrame, dict):
    """
    Impute missing numerical values.
    - sleep_hours: median imputation (observed to be missing in real data)
    - energy_level, stress_level: should be present but safe-guard with median
    - duration_min: median imputation
    Returns (cleaned_df, medians_dict) — pass medians from train to test.
    """
    num_cols = ["sleep_hours", "energy_level", "stress_level", "duration_min"]
    df = df.copy()
    computed = {}
    for col in num_cols:
        if col not in df.columns:
            df[col] = 5  # safe default
            computed[col] = 5
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if medians is None:
            med = df[col].median()
        else:
            med = medians.get(col, df[col].median())
        df[col] = df[col].fillna(med)
        computed[col] = med
    return df, computed


# ── CATEGORICAL CLEANING ──────────────────────────────────────────────────────

def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # previous_day_mood — missing observed in real data
    df["previous_day_mood"] = (
        df["previous_day_mood"]
        .fillna("neutral")
        .str.lower()
        .str.strip()
        .map(PREVIOUS_MOOD_MAP)
        .fillna(2)   # unknown → neutral
        .astype(int)
    )

    # reflection_quality — real values are clear/vague/conflicted
    df["reflection_quality"] = (
        df["reflection_quality"]
        .fillna("vague")
        .str.lower()
        .str.strip()
        .map(REFLECTION_QUALITY_MAP)
        .fillna(1)
        .astype(int)
    )

    # time_of_day — real data has early_morning
    df["time_of_day"] = (
        df["time_of_day"]
        .fillna("morning")
        .str.lower()
        .str.strip()
        .map(TIME_OF_DAY_MAP)
        .fillna(1)
        .astype(int)
    )

    # face_emotion_hint — real data uses 'calm_face', 'none', etc.
    df["face_emotion_hint"] = df["face_emotion_hint"].apply(clean_face_emotion)

    # ambience_type — standardize
    df["ambience_type"] = df["ambience_type"].fillna("unknown").str.lower().str.strip()

    return df


# ── LABEL CLEANING ────────────────────────────────────────────────────────────

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Only run on training data (has emotional_state + intensity columns)."""
    df = df.copy()
    if "emotional_state" in df.columns:
        df["emotional_state"] = df["emotional_state"].str.lower().str.strip()
    if "intensity" in df.columns:
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        df["intensity"] = df["intensity"].clip(1, 5).fillna(3).astype(int)
    return df


# ── FULL CLEANING PIPELINE ────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame, is_train: bool = True, train_medians: dict = None):
    """
    Full cleaning pipeline.
    
    Parameters
    ----------
    df           : raw dataframe
    is_train     : if True, also clean label columns + compute medians
    train_medians: dict of medians from training set (pass during test cleaning)
    
    Returns
    -------
    cleaned_df, medians_dict
    """
    df = df.copy()

    # 1. Text
    df["journal_text_clean"] = df["journal_text"].apply(clean_text)
    df["text_length"]        = df["journal_text"].apply(text_length)
    df["is_vague_text"]      = df["journal_text"].apply(is_vague_text)

    # 2. Numerics
    df, medians = impute_numerics(df, medians=train_medians)

    # 3. Categoricals
    df = clean_categoricals(df)

    # 4. Labels (train only)
    if is_train:
        df = clean_labels(df)

    return df, medians


# ── QUICK AUDIT ───────────────────────────────────────────────────────────────

def audit(df: pd.DataFrame, label: str = "Dataset"):
    print(f"\n{'='*50}")
    print(f"  {label} — {len(df)} rows")
    print(f"{'='*50}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    if "emotional_state" in df.columns:
        print(f"\nLabel distribution:\n{df['emotional_state'].value_counts()}")
    if "intensity" in df.columns:
        print(f"\nIntensity distribution:\n{df['intensity'].value_counts().sort_index()}")
    print(f"\ntime_of_day unique: {sorted(df['time_of_day'].unique()) if 'time_of_day' in df.columns else 'N/A'}")
    print(f"reflection_quality unique: {sorted(df['reflection_quality'].unique()) if 'reflection_quality' in df.columns else 'N/A'}")


if __name__ == "__main__":
    # Quick test on inline sample from assignment
    import io
    raw = """id\tjournal_text\tambience_type\tduration_min\tsleep_hours\tenergy_level\tstress_level\ttime_of_day\tprevious_day_mood\tface_emotion_hint\treflection_quality\temotional_state\tintensity
1\tThe ocean ambience helped me stop drifting and concentrate on my next steps. My to-do list feels less chaotic.\tocean\t12\t6.5\t4\t2\tafternoon\tmixed\tcalm_face\tclear\tfocused\t3
3\tThe forest session slowed my thoughts and I feel more settled now.\tforest\t3\t\t2\t1\tnight\toverwhelmed\thappy_face\tclear\tcalm\t3
4\tthe mountain ambience was pleasant, though i can't say it shifted my mood much. idk.\tmountain\t25\t7\t4\t4\tnight\tfocused\tcalm_face\tvague\tneutral\t1
13\tThe forest session made me calmer, but part of me still feels uneasy. Part of me wants rest, part of me wants action.\tforest\t20\t5\t2\t2\tafternoon\tneutral\tnone\tconflicted\tmixed\t2
17\tThe forest session made me calmer, but part of me still feels uneasy. I feel better and not better at the same time.\tforest\t35\t6\t2\t3\tnight\t\tcalm_face\tvague\tmixed\t3"""

    df_raw = pd.read_csv(io.StringIO(raw), sep='\t')
    df_clean, meds = clean_dataframe(df_raw, is_train=True)
    audit(df_clean, "Sample Clean")
    print("\nCleaned journal_text_clean sample:")
    print(df_clean[["id","journal_text_clean","is_vague_text","time_of_day","reflection_quality","face_emotion_hint"]].to_string())
