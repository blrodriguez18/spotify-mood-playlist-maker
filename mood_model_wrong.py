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
    base_rows  = []
    mood_rows  = []
    mood_names = ["happy", "sad", "aggressive", "relaxed", "party"]

    for t in tracks:
        base_rows.append([
            t["valence"],
            t["energy"],
            t["danceability"],
            t["tempo"] / 200.0,
            t["acousticness"],
        ])

        mv = t.get("mood_vector")
        if mv and any(mv.get(m, 0) > 0 for m in mood_names):
            mood_rows.append([mv.get(m, 0) for m in mood_names])
        else:
            mood_rows.append(None)

    # ── Build per-mood-label mean vectors from tracks that DO have data ──
    # Group real mood vectors by the track's freqblog 'mood' label
    # e.g. all tracks labeled "happy" that have real vectors get averaged
    mood_label_groups = {}
    for t, mv_row in zip(tracks, mood_rows):
        if mv_row is None:
            continue
        label = t.get("mood", "unknown").lower()
        if label not in mood_label_groups:
            mood_label_groups[label] = []
        mood_label_groups[label].append(mv_row)

    # Compute the mean vector for each mood label
    mood_label_means = {}
    for label, vectors in mood_label_groups.items():
        arr = np.array(vectors)
        mood_label_means[label] = arr.mean(axis=0).tolist()
        print(f"  Mood '{label}': {len(vectors)} tracks with real vectors → mean computed")

    # Global fallback mean in case a mood label has zero real vectors
    all_real = [mv for mv in mood_rows if mv is not None]
    global_mean = np.array(all_real).mean(axis=0).tolist() if all_real else [0.2] * 5

    # ── Impute missing mood vectors using the per-label mean ─────────────
    mood_rows_filled = []
    imputed_count = 0
    for t, mv_row in zip(tracks, mood_rows):
        if mv_row is not None:
            mood_rows_filled.append(mv_row)
        else:
            label = t.get("mood", "unknown").lower()
            if label in mood_label_means:
                # Use the mean of tracks with the same mood label — meaningful imputation
                mood_rows_filled.append(mood_label_means[label])
            else:
                # Last resort: global mean (rare — only if label has no real vectors at all)
                mood_rows_filled.append(global_mean)
            imputed_count += 1

    print(f"\nImputed {imputed_count} missing mood vectors using per-label means")

    # ── Scale both feature sets ───────────────────────────────────────────
    base_array = np.array(base_rows)
    mood_array = np.array(mood_rows_filled)

    scaler_base = StandardScaler()
    scaler_mood = StandardScaler()
    base_scaled = scaler_base.fit_transform(base_array)
    mood_scaled = scaler_mood.fit_transform(mood_array)

    # Weight mood at 0.5x — it's derived/imputed for many tracks
    # so we don't want it to dominate over the real audio features
    X = np.hstack([base_scaled, mood_scaled * 0.5])

    has_real_mood = len(tracks) - imputed_count
    print(f"Feature matrix: {X.shape[0]} tracks × {X.shape[1]} features")
    print(f"Real mood vectors: {has_real_mood}  |  Imputed: {imputed_count}")

    return X, scaler_base, scaler_mood


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
def save_model(km, scaler_base, profiles, tracks, labels):
    labeled_tracks = []
    for track, label in zip(tracks, labels):
        labeled_tracks.append({**track, "cluster": int(label)})

    with open("labeled_tracks.json", "w") as f:
        json.dump(labeled_tracks, f, indent=2)

    with open("mood_model.pkl", "wb") as f:
        pickle.dump({
            "model":       km,
            "scaler_base": scaler_base,
            "scaler_mood": scaler_mood,  # ← added
            "profiles":    profiles,
        }, f)

    print("\nSaved mood_model_wrong.pkl and labeled_tracks_wrong.json")


# ── Run everything ────────────────────────────────────────────────────
if __name__ == "__main__":
    tracks                      = load_tracks()
    X, scaler_base, scaler_mood = build_X(tracks)
    best_k                      = find_best_k(X)
    km                          = train(X, best_k)
    labels                      = km.labels_
    profiles                    = label_clusters(tracks, labels, best_k)
    print_summary(profiles)
    save_model(km, scaler_base, profiles, tracks, labels)