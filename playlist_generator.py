"""
playlist_generator.py
─────────────────────
Loads the trained pkl models and dataset.csv at startup, classifies every
track with the Random Forest (mood label + confidence score), and exposes
a single public function:

    get_playlist_tracks(mood, n=25) -> list[dict]

The returned list is ready to hand straight to the Spotify API.
"""

import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────

# The 8 audio features the models were trained on — order matters for the
# scaler, so keep this list identical to what you used during training.
FEATURES = [
    "valence",
    "energy",
    "danceability",
    "tempo",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "loudness",
]

CLUSTER_NAMES = {
    0: "melancholic",
    1: "happy",
    2: "energized",   # ← was epic
    3: None,
    4: "focused",
    5: "energized",   # ← was epic
    6: None,
    7: "chill",
}

# Map the mood button labels (sent by the frontend) to whatever class labels
# your Random Forest was trained on.  Adjust the right-hand side values to
# match exactly what is stored in your model's classes_ attribute.
MOOD_LABEL_MAP = {
    "happy":      "happy",
    "melancholic":"melancholic",
    "energized":  "energized",
    "chill":      "chill",
    "focused": "focused",
    "epic": "epic"
}

# Paths — update these if your folder layout differs.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "dataset.csv")
KMEANS_PATH= os.path.join(BASE_DIR, "mood_classifier_kmeans.pkl")   # loaded but not
                                                            # used in generation;
                                                            # kept for future use


# ──────────────────────────────────────────────
# 2.  ONE-TIME STARTUP: load models + classify
# ──────────────────────────────────────────────
# Everything in this section runs exactly once when the module is first
# imported.  Flask imports this file at startup, so by the time the first
# request arrives all 114k tracks are already classified and sitting in memory
# as a plain DataFrame — no pkl or CSV reads happen at request time.

print("[playlist_generator] Loading model bundle …")
bundle        = joblib.load(os.path.join(BASE_DIR, "mood_classifier_kmeans.pkl"))
rf_model      = bundle["classifier"]
kmeans_model  = bundle["kmeans"]
scaler        = bundle["scaler"]
mood_merge    = bundle.get("mood_merge", {})
# mood_merge = bundle.get("mood_merge", {
#     "epic_instrumental": "epic",
#     "comedy": None,
# })

# Then in section 2b, replace the scaler block with:
print("[playlist_generator] Reading dataset.csv …")
df_raw = pd.read_csv(DATA_PATH)

# ── 2a. Clean ──────────────────────────────────
# Drop rows that are missing any of the 8 audio features or the columns we
# need for the final playlist payload.
required_cols = FEATURES + ["track_name", "artists", "popularity"]
df = df_raw.dropna(subset=required_cols).copy()

# Popularity is occasionally stored as a float in some Kaggle exports.
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)

# ── 2b. Scale features ─────────────────────────
# The Random Forest was trained on scaled data, so we must apply the same
# transformation here.  We re-fit a fresh scaler on the full dataset because
# the training script didn't export a fitted scaler pkl alongside the model.
#
# ⚠️  If your training script DID save a scaler.pkl, replace the two lines
#     below with:
#         scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
#         X_scaled = scaler.transform(df[FEATURES].values)
# scaler   = StandardScaler()
X_scaled = scaler.transform(df[FEATURES].values)  # transform only, don't re-fit

# ── 2c. Classify ───────────────────────────────
# predict()       → mood label for every track  (shape: N,)
# predict_proba() → probability vector per track (shape: N, n_classes)
# confidence      → the highest probability in that vector — how sure the
#                   model is about its top prediction

# required_cols = FEATURES + ["track_name", "artists", "popularity"]
# df = df_raw.dropna(subset=required_cols).copy()
# df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)

# print("[playlist_generator] Scaling features …")
# X_scaled = scaler.transform(df[FEATURES].values)   # transform only — scaler is pre-fitted

# print("[playlist_generator] Classifying all tracks …")
# predicted_labels  = rf_model.predict(X_scaled)        # raw archetype names e.g. "epic_instrumental"
# predicted_proba   = rf_model.predict_proba(X_scaled)
# confidence_scores = predicted_proba.max(axis=1)

# df["predicted_mood"] = predicted_labels
# df["confidence"]     = confidence_scores

# df["mood"] = df["predicted_mood"].map(
#     lambda m: mood_merge.get(m, m)   # if not in merge map, keep as-is
# )

# df = df[df["mood"].notna()].copy()

# print(f"[playlist_generator] Done. {len(df):,} usable tracks after dropping excluded clusters.")
# print("Mood distribution:\n", df["mood"].value_counts().to_string())



# Map integer cluster IDs → mood names using the cluster_names from the pkl
# cluster_names looks like {0: "melancholic", 1: "happy", 2: "epic", ...}
# Reverse it to a plain dict for .map()
cluster_names = bundle.get("cluster_names", {})
mood_merge    = bundle.get("mood_merge", {})

def resolve_mood(cluster_id):
    raw = cluster_names.get(int(cluster_id))       # e.g. "epic_instrumental"
    if raw is None:
        return None
    return mood_merge.get(raw, raw)  

print("[playlist_generator] Classifying all tracks …")
predicted_labels  = rf_model.predict(X_scaled)        # integers: 0,1,2...
predicted_proba   = rf_model.predict_proba(X_scaled)
confidence_scores = predicted_proba.max(axis=1)

df["predicted_mood"] = predicted_labels
df["confidence"]     = confidence_scores

# Resolve integer cluster IDs → final mood name strings
df["mood"] = df["predicted_mood"].apply(resolve_mood)
print("Raw cluster counts:\n", df["predicted_mood"].value_counts().to_string())

# Drop comedy and any other None-mapped clusters
df = df[df["mood"].notna()].copy()

print(f"[playlist_generator] Done. {len(df):,} usable tracks.")
print("Mood distribution:\n", df["mood"].value_counts().to_string())

print("cluster_names from pkl:", bundle.get("cluster_names"))
print("mood_merge:", mood_merge)
print("Sample raw predictions:", predicted_labels[:10])
print("rf_model.classes_:", rf_model.classes_)

print("\nActual cluster centroids (mean feature values):")
for cid in sorted(df["predicted_mood"].unique()):
    members = df[df["predicted_mood"] == cid]
    avg = members[FEATURES].mean()
    print(f"\nCluster {int(cid)} ({len(members):,} tracks)")
    print(f"  valence:          {avg['valence']:.3f}")
    print(f"  energy:           {avg['energy']:.3f}")
    print(f"  danceability:     {avg['danceability']:.3f}")
    print(f"  tempo:            {avg['tempo']:.1f}")
    print(f"  acousticness:     {avg['acousticness']:.3f}")
    print(f"  speechiness:      {avg['speechiness']:.3f}")
    print(f"  instrumentalness: {avg['instrumentalness']:.3f}")
    print(f"  loudness:         {avg['loudness']:.1f}")

# ──────────────────────────────────────────────
# 3.  PUBLIC API
# ──────────────────────────────────────────────

# Valid moods are now derived from the data, not hardcoded
VALID_MOODS = set(df["mood"].unique())
print("VALID MOODS:", VALID_MOODS)


def get_playlist_tracks(mood: str, n: int = 25) -> list[dict]:
    mood_key = mood.strip().lower()
    if mood_key not in VALID_MOODS:
        raise ValueError(f"Unknown mood '{mood}'. Valid options: {', '.join(sorted(VALID_MOODS))}")

    n = min(max(n, 1), 50)

    pool = df[df["mood"] == mood_key].copy()

    # Deduplicate by track name + artist — keep highest confidence per track
    pool = (pool.sort_values("confidence", ascending=False)
                .drop_duplicates(subset=["track_name", "artists"])
                .reset_index(drop=True))

    if len(pool) < n:
        raise RuntimeError(f"Only {len(pool)} tracks for '{mood_key}', but {n} requested.")

    n_popular   = math.ceil(n * 0.80)
    n_confident = n - n_popular

    top_popular = pool.sort_values("popularity", ascending=False).head(n_popular)
    top_confident = (pool[~pool.index.isin(set(top_popular.index))]
                        .sort_values("confidence", ascending=False)
                        .head(n_confident))

    playlist_df = (pd.concat([top_popular, top_confident])
                     .sample(frac=1, random_state=None) 
                     .reset_index(drop=True))

    return playlist_df.where(pd.notnull(playlist_df), None).to_dict(orient="records")

# ──────────────────────────────────────────────
# 4.  QUICK SMOKE-TEST  (python playlist_generator.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import json

    for test_mood in MOOD_LABEL_MAP:
        try:
            tracks = get_playlist_tracks(test_mood, n=25)
            print(f"\n{'─'*50}")
            print(f"Mood: {test_mood}  →  {len(tracks)} tracks returned")
            print(f"  Sample: {tracks[0]['track_name']} — {tracks[0]['artists']}")
            print(f"  Popularity: {tracks[0]['popularity']}  |  "
                  f"Confidence: {tracks[0]['confidence']:.3f}")
        except Exception as exc:
            print(f"[ERROR] {test_mood}: {exc}")
