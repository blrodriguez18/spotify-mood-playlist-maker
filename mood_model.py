import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle


# ── 1. Load tracks ────────────────────────────────────────────────────
def load_tracks(path="track_data.json"):
    with open(path, "r") as f:
        tracks = json.load(f)

    # Only keep tracks that have all the numeric features we need
    valid = [t for t in tracks if t.get("valence") and t.get("energy")]
    print(f"Valid tracks for clustering: {len(valid)} / {len(tracks)} total")
    return valid

# ── 2. Build the feature matrix ───────────────────────────────────────
def build_X(tracks):
    """
    Pure audio features only — real measured values for every track.
    No imputation, no noise from sparse mood vectors.
    """
    rows = []
    for t in tracks:
        rows.append([
            t["valence"],
            t["energy"],
            t["danceability"],
            t["tempo"] / 200.0,
            t["acousticness"],
        ])

    X = np.array(rows)
    print(f"Feature matrix: {X.shape[0]} tracks × {X.shape[1]} features")
    return X


# ── 3. Scale the features ─────────────────────────────────────────────
def scale(X):
    """
    K-means uses distance between points, so features need to be on
    the same scale. StandardScaler makes every feature have mean=0
    and standard deviation=1 so no single feature dominates.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ── 4. Find the best number of clusters ──────────────────────────────
def find_best_k(X_scaled, k_min=4, k_max=9):
    """
    Silhouette score measures how well separated clusters are.
    Ranges from -1 to 1 — higher is better.
    We try k=4 to k=9 and pick the winner.
    """
    print("\nFinding best number of mood clusters...")
    best_k, best_score = k_min, -1

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  k={k}  silhouette score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest k: {best_k}  (score: {best_score:.4f})")
    return best_k


# ── 5. Train the final model ──────────────────────────────────────────
def train(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    return km


# ── 6. Name each cluster ──────────────────────────────────────────────
def label_clusters(tracks, labels, k):
    """
    For each cluster, average the feature values and mood_vector scores
    across all its tracks. This tells us what mood that cluster represents.
    """
    feature_names = ["valence", "energy", "danceability", "acousticness"]
    mood_names    = ["happy", "sad", "aggressive", "relaxed", "party"]

    profiles = {}
    for cluster_id in range(k):
        members = [t for t, l in zip(tracks, labels) if l == cluster_id]

        # Average each audio feature across the cluster
        avg_features = {}
        for feat in feature_names:
            vals = [t[feat] for t in members if t.get(feat) is not None]
            avg_features[feat] = round(sum(vals) / len(vals), 3) if vals else 0

        # Average each mood_vector dimension
        avg_moods = {}
        for mood in mood_names:
            vals = [t["mood_vector"][mood] for t in members
                    if t.get("mood_vector") and mood in t["mood_vector"]]
            avg_moods[mood] = round(sum(vals) / len(vals), 3) if vals else 0

        # The dominant mood is whichever mood_vector dimension scores highest
        dominant_mood = max(avg_moods, key=avg_moods.get) if avg_moods else "unknown"

        profiles[cluster_id] = {
            "dominant_mood":  dominant_mood,
            "avg_features":   avg_features,
            "avg_moods":      avg_moods,
            "track_count":    len(members),
            "sample_tracks":  [(t["name"], t["artist"]) for t in members[:4]],
        }

    return profiles


# ── 7. Print a summary ────────────────────────────────────────────────
def print_summary(profiles):
    print("\n" + "=" * 55)
    print("YOUR MOOD CLUSTERS")
    print("=" * 55)

    for cid, p in profiles.items():
        print(f"\nCluster {cid} — dominant mood: {p['dominant_mood'].upper()}")
        print(f"  Tracks   : {p['track_count']}")
        print(f"  Valence  : {p['avg_features']['valence']}   "
              f"Energy: {p['avg_features']['energy']}   "
              f"Dance: {p['avg_features']['danceability']}")
        mood_str = "  ".join([f"{m}: {v}" for m, v in p['avg_moods'].items()])
        print(f"  Moods    : {mood_str}")
        print(f"  Examples : {p['sample_tracks']}")


# ── 8. Save everything ────────────────────────────────────────────────
def save_model(km, scaler, profiles, tracks, labels):
    labeled_tracks = []
    for track, label in zip(tracks, labels):
        labeled_tracks.append({**track, "cluster": int(label)})

    with open("labeled_tracks.json", "w") as f:
        json.dump(labeled_tracks, f, indent=2)

    with open("mood_model.pkl", "wb") as f:
        pickle.dump({
            "model":    km,
            "scaler":   scaler,
            "profiles": profiles,
        }, f)

    print("\nSaved mood_model.pkl and labeled_tracks.json")


# ── Run everything ────────────────────────────────────────────────────
# build_X now returns just X
# update if __name__ == "__main__":
if __name__ == "__main__":
    tracks   = load_tracks()
    X        = build_X(tracks)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_k   = find_best_k(X_scaled)
    km       = train(X_scaled, best_k)
    labels   = km.labels_
    profiles = label_clusters(tracks, labels, best_k)
    print_summary(profiles)
    save_model(km, scaler, profiles, tracks, labels)