import json
import pickle
import numpy as np
from collections import Counter

FEATURES = ["valence", "energy", "danceability", "tempo",
            "acousticness", "speechiness", "instrumentalness",
            "loudness"]


# ── Load model ────────────────────────────────────────────────────────
def load_model(path="mood_classifier_kmeans.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["classifier"], data["scaler"], data["cluster_names"]


# ── Load your personal library ────────────────────────────────────────
def load_library(path="track_data.json"):
    with open(path) as f:
        tracks = json.load(f)
    print(f"Loaded {len(tracks)} tracks from your library")
    return tracks


# ── Check which features are missing ─────────────────────────────────
def check_features(tracks):
    missing = [f for f in FEATURES if f not in tracks[0]]
    if missing:
        print(f"WARNING: these features are missing from your library: {missing}")
        print("They need to be fetched from freqblog before we can predict.")
    return missing


# ── Build feature matrix from your library ────────────────────────────
def build_X(tracks):
    rows = []
    valid_tracks = []
    for t in tracks:
        # Skip tracks missing any required feature
        if not all(t.get(f) is not None for f in FEATURES):
            continue
        rows.append([t[f] for f in FEATURES])
        valid_tracks.append(t)

    print(f"Tracks with all features: {len(valid_tracks)} / {len(tracks)}")
    return np.array(rows), valid_tracks


# ── Predict mood for each track ───────────────────────────────────────
def predict(clf, scaler, cluster_names, X, tracks):
    X_scaled  = scaler.transform(X)
    cluster_ids = clf.predict(X_scaled)
    proba       = clf.predict_proba(X_scaled)

    labeled = []
    for track, cid, prob in zip(tracks, cluster_ids, proba):
        labeled.append({
            **track,
            "mood":       cluster_names[cid],
            "mood_id":    int(cid),
            "confidence": round(float(prob[cid]), 4),
        })

    return labeled


# ── Print summary ─────────────────────────────────────────────────────
def print_summary(labeled_tracks, cluster_names):
    mood_counts = Counter(t["mood"] for t in labeled_tracks)

    print("\n" + "=" * 55)
    print("YOUR LIBRARY BY MOOD")
    print("=" * 55)

    for cid, name in cluster_names.items():
        count   = mood_counts.get(name, 0)
        pct     = 100 * count / len(labeled_tracks)
        bar     = "█" * int(pct / 2)
        print(f"\n{name.upper():<15} {count:>3} tracks  {pct:.1f}%  {bar}")

        # Show top 5 tracks by confidence for this mood
        mood_tracks = sorted(
            [t for t in labeled_tracks if t["mood"] == name],
            key=lambda x: x["confidence"],
            reverse=True
        )[:5]
        for t in mood_tracks:
            print(f"  {t['confidence']:.2f}  {t['name']} — {t['artist']}")


# ── Save labeled library ──────────────────────────────────────────────
def save(labeled_tracks, path="labeled_library.json"):
    with open(path, "w") as f:
        json.dump(labeled_tracks, f, indent=2)
    print(f"\nSaved to labeled_library.json")


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf, scaler, cluster_names = load_model()
    tracks                     = load_library()
    missing                    = check_features(tracks)

    if missing:
        print("\nCannot predict — fetch missing features first.")
    else:
        X, valid_tracks  = build_X(tracks)
        labeled_tracks   = predict(clf, scaler, cluster_names, X, valid_tracks)
        print_summary(labeled_tracks, cluster_names)
        save(labeled_tracks)