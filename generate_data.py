"""
generate_data.py
=================
Generates synthetic data matching the ACTUAL ArvyaX schema observed in the dataset sample.

Real schema differences vs original assumptions:
- reflection_quality: 'clear' | 'vague' | 'conflicted'  (not low/medium/high)
- time_of_day: 'early_morning' | 'morning' | 'afternoon' | 'evening' | 'night'
- face_emotion_hint: 'calm_face' | 'tense_face' | 'tired_face' | 'happy_face' | 'neutral_face' | 'none'
- previous_day_mood: 'calm' | 'neutral' | 'mixed' | 'overwhelmed' | 'focused' | 'restless' (from sample)
- emotional_state includes: 'neutral' | 'mixed'
- sleep_hours and previous_day_mood can be blank (NaN)

REPLACE THIS FILE: if you have real data, just put train.csv and test.csv in data/
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

STATES = ["calm","anxious","focused","restless","sad",
          "energized","overwhelmed","content","tired","hopeful","neutral","mixed"]

AMBIENCE = ["ocean","forest","rain","mountain","cafe"]
TIMES    = ["early_morning","morning","afternoon","evening","night"]
FACE     = ["calm_face","tense_face","tired_face","happy_face","neutral_face","none"]
PREV_MOOD= ["calm","neutral","mixed","overwhelmed","focused","restless","positive","negative"]
REFL_Q   = ["clear","vague","conflicted"]

TEMPLATES = {
    "calm":        [
        "The {amb} session slowed my thoughts and I feel more settled now.",
        "I feel lighter after the {amb} sounds, like my mind finally softened.",
        "after the {amb} track i feel peaceful and less pulled in every direction.",
    ],
    "anxious":     [
        "I can't stop thinking even after the {amb} session. Too many things.",
        "Heart was racing throughout. The {amb} helped a bit but anxiety is still there.",
        "Kept second-guessing myself during the {amb} session.",
    ],
    "focused":     [
        "The {amb} ambience helped me stop drifting and concentrate on my next steps.",
        "I came in distracted, but I left the {amb} session with a sharper mind.",
        "I feel mentally clear after the {amb} session and ready to tackle one thing at a time.",
    ],
    "restless":    [
        "even with the {amb} session, my mind kept jumping between tasks.",
        "I couldn't really settle into the {amb} track; I kept thinking of everything at once.",
        "The {amb} sounds were nice, but I still feel unsettled and fidgety.",
    ],
    "sad":         [
        "Felt low even after the {amb} session. Hard to find motivation.",
        "Missing something. The {amb} track didn't lift the heavy feeling much.",
        "Heavy feeling throughout the {amb} session. Just going through the motions.",
    ],
    "energized":   [
        "Felt alive after the {amb} session! Ready to take on the day.",
        "Buzzing with energy. The {amb} track amplified my motivation.",
        "Motivated and upbeat after {amb}. Let's go!",
    ],
    "overwhelmed": [
        "even after the {amb} track, i feel exhausted and emotionally overloaded.",
        "The {amb} session gave me a pause, but the pressure is still sitting hard on me.",
        "Too much on my plate. The {amb} helped briefly but the weight came back.",
    ],
    "content":     [
        "Simple joy after the {amb} session. Things are okay.",
        "Nothing spectacular but feeling good after {amb}.",
        "Quiet happiness after the {amb} track. All is well.",
    ],
    "tired":       [
        "So tired even after {amb}. Could sleep for days.",
        "Running on empty. The {amb} session was calming but I'm still drained.",
        "Exhausted but okay. The {amb} track helped me slow down.",
    ],
    "hopeful":     [
        "Tomorrow looks better after the {amb} session. Feel optimistic.",
        "Things will work out. The {amb} gave me clarity.",
        "Light at the end of the tunnel after the {amb} track.",
    ],
    "neutral":     [
        "the {amb} ambience was pleasant, though i can't say it shifted my mood much. idk.",
        "Nothing strong came up during the {amb} session; I feel fairly normal.",
        "The {amb} was okay I don't feel much different, just a bit more aware.",
    ],
    "mixed":       [
        "The {amb} session made me calmer, but part of me still feels uneasy.",
        "I feel better and not better at the same time after the {amb} session.",
        "I liked the {amb} session, but my mood is still split between calm and tension.",
    ],
}

SHORT_NOISE = ["idk.", "idk", "not sure", "maybe", "ok I guess"]

INTENSITY_BASE = {
    "calm":1,"content":1,"hopeful":2,"neutral":1,
    "focused":3,"tired":2,"restless":3,"mixed":2,
    "sad":3,"anxious":4,"overwhelmed":5,"energized":3
}

def make_row(idx, state, split):
    amb   = np.random.choice(AMBIENCE)
    dur   = int(np.random.randint(5, 61))
    sleep = round(np.random.uniform(3.5, 9.5), 1) if np.random.random() > 0.05 else np.nan
    energy= int(np.random.randint(1, 11))
    stress= int(np.random.randint(1, 11))
    tod   = np.random.choice(TIMES)
    prev  = np.random.choice(PREV_MOOD) if np.random.random() > 0.05 else np.nan
    face  = np.random.choice(FACE)
    refl  = np.random.choice(REFL_Q)

    base_int = INTENSITY_BASE[state]
    intensity = max(1, min(5, base_int + np.random.randint(-1, 2)))

    template = np.random.choice(TEMPLATES[state])
    text = template.format(amb=amb)
    # Occasionally append noise
    if np.random.random() < 0.08:
        text += " " + np.random.choice(SHORT_NOISE)

    row = {
        "id": idx,
        "journal_text": text,
        "ambience_type": amb,
        "duration_min": dur,
        "sleep_hours": sleep,
        "energy_level": energy,
        "stress_level": stress,
        "time_of_day": tod,
        "previous_day_mood": prev,
        "face_emotion_hint": face,
        "reflection_quality": refl,
    }
    if split == "train":
        row["emotional_state"] = state
        row["intensity"]       = intensity
    return row


def generate(n, split, start_id=1):
    rows = []
    for i in range(n):
        state = STATES[i % len(STATES)]
        rows.append(make_row(start_id + i, state, split))
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    train = generate(500, "train", start_id=1)
    test  = generate(150, "test",  start_id=5001)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv",   index=False)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print("Train cols:", list(train.columns))
    print("Test cols: ", list(test.columns))
    # Quick check for schema alignment
    print("\nSample reflection_quality values:", train["reflection_quality"].unique())
    print("Sample time_of_day values:",        train["time_of_day"].unique())
    print("Sample face_emotion_hint values:",  train["face_emotion_hint"].unique())
    print("Missing sleep_hours in train:",     train["sleep_hours"].isna().sum())
