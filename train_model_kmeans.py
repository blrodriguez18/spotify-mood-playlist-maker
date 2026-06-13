import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import pickle

FEATURES = ["valence", "energy", "danceability", "tempo",
            "acousticness", "speechiness", "instrumentalness", "loudness"]

DROP_CLUSTERS = {"comedy"} 


# ── 1. Load data ──────────────────────────────────────────────────────
def load_data(path="dataset.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES)
    print(f"Loaded {len(df)} tracks")
    return df

# ── 2. Scale features ─────────────────────────────────────────────────
def scale(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)
    return X, scaler

# ── 3. Find best k ────────────────────────────────────────────────────
def find_best_k(X, k_min=4, k_max=9):
    """
    Use MiniBatchKMeans for speed on 114k rows —
    same algorithm as KMeans but processes data in chunks.
    Results are near-identical but 10x faster.
    """
    print("\nFinding best k on 114k tracks (using MiniBatchKMeans for speed)...")
    best_k, best_score = k_min, -1

    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=42,
                             batch_size=4096, n_init=10)
        labels = km.fit_predict(X)

        # Silhouette on full 114k is slow — sample 10k for speed
        sample_idx = np.random.choice(len(X), 10000, replace=False)
        score = silhouette_score(X[sample_idx], labels[sample_idx])
        print(f"  k={k}  silhouette: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest k: {best_k}  (score: {best_score:.4f})")
    return best_k

# ── 4. Train final KMeans ─────────────────────────────────────────────
def cluster(X, k):
    print(f"\nTraining final KMeans with k={k}...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42,
                         batch_size=4096, n_init=10)
    labels = km.fit_predict(X)
    return km, labels

# ── 5. Profile each cluster ───────────────────────────────────────────
def profile_clusters(df, labels, k):
    """
    For each cluster print the average of every feature
    and 5 sample tracks so you can name the mood yourself.
    """
    df = df.copy()
    df["cluster"] = labels

    print("\n" + "=" * 65)
    print("CLUSTER PROFILES — inspect these to assign mood names")
    print("=" * 65)

    profiles = {}
    for cid in range(k):
        members = df[df["cluster"] == cid]

        avg = members[FEATURES].mean()
        samples = members[["track_name", "artists", "track_genre"]].sample(
            min(5, len(members)), random_state=42
        )

        print(f"\nCluster {cid} -- ({len(members)} tracks)")
        print(f"  valence:         {avg['valence']:.3f}   (0=sad, 1=happy)")
        print(f"  energy:          {avg['energy']:.3f}   (0=calm, 1=intense)")
        print(f"  danceability:    {avg['danceability']:.3f}   (0=stiff, 1=groovy)")
        print(f"  tempo:           {avg['tempo']:.1f} BPM")
        print(f"  acousticness:    {avg['acousticness']:.3f}   (0=electric, 1=acoustic)")
        print(f"  speechiness:     {avg['speechiness']:.3f}   (0=music, 1=spoken)")
        print(f"  instrumentalness:{avg['instrumentalness']:.3f}   (0=vocals, 1=instrumental)")
        print(f"  loudness:        {avg['loudness']:.1f} dB")
        print(f"  Sample tracks:")
        for _, row in samples.iterrows():
            print(f"    - {row['track_name']} — {row['artists']} [{row['track_genre']}]")

        profiles[cid] = avg.to_dict()
        profiles[cid]["track_count"] = len(members)

    cluster_names = assign_mood_names(profiles)
    print("\nAuto-assigned mood names:")
    for cid, name in cluster_names.items():
        print(f"  Cluster {cid}: {name}")

    return profiles, df, cluster_names

# ── 6. Train classifier on cluster labels ─────────────────────────────
def train_classifier(df_labeled):
    """
    Now that clusters are labeled by the algorithm, train a Random Forest
    to generalize — so we can predict mood for new tracks not in the dataset.
    """
    X = df_labeled[FEATURES].values
    y = df_labeled["cluster"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Random Forest on cluster labels...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURES, clf.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 100)
        print(f"  {feat:<20} {imp:.4f}  {bar}")

    return clf


def assign_mood_names(profiles: dict) -> dict:
    """
    Score each cluster against mood archetypes derived from the actual
    centroid values observed after training on this dataset.
    Returns {cluster_id: mood_name_string}
    """
    # Archetypes are anchored to the real centroid values from the training run.
    # comedy and focus are included so the distance matcher can identify them —
    # comedy gets dropped at playlist-generation time, focus becomes a button.
    archetypes = {
        "happy": {
            "valence": 0.733, "energy": 0.740, "danceability": 0.698,
            "tempo": 118.5,   "acousticness": 0.218, "speechiness": 0.063,
            "instrumentalness": 0.020, "loudness": -6.3,
        },
        "epic": {
            "valence": 0.325, "energy": 0.741, "danceability": 0.520,
            "tempo": 112.0,   "acousticness": 0.108, "speechiness": 0.060,
            "instrumentalness": 0.030, "loudness": -6.2,
        },
        "epic_instrumental": {
            # Second epic cluster — high instrumentalness is the key differentiator.
            # Mapped to "epic" in MOOD_MERGE below so both feed the same button.
            "valence": 0.340, "energy": 0.744, "danceability": 0.589,
            "tempo": 126.4,   "acousticness": 0.112, "speechiness": 0.069,
            "instrumentalness": 0.797, "loudness": -8.5,
        },
        "melancholic": {
            "valence": 0.407, "energy": 0.363, "danceability": 0.529,
            "tempo": 114.5,   "acousticness": 0.724, "speechiness": 0.048,
            "instrumentalness": 0.030, "loudness": -10.9,
        },
        "energized": {
            "valence": 0.459, "energy": 0.829, "danceability": 0.451,
            "tempo": 163.2,   "acousticness": 0.101, "speechiness": 0.089,
            "instrumentalness": 0.042, "loudness": -5.2,
        },
        "chill": {
            "valence": 0.568, "energy": 0.672, "danceability": 0.707,
            "tempo": 120.6,   "acousticness": 0.270, "speechiness": 0.309,
            "instrumentalness": 0.023, "loudness": -7.1,
        },
        "focused": {
            "valence": 0.181, "energy": 0.177, "danceability": 0.343,
            "tempo": 102.7,   "acousticness": 0.860, "speechiness": 0.051,
            "instrumentalness": 0.791, "loudness": -21.2,
        },
        "comedy": {
            # Kept so the matcher can identify and name it —
            # dropped at playlist-generation time, never shown to the user.
            "valence": 0.427, "energy": 0.686, "danceability": 0.566,
            "tempo": 99.9,    "acousticness": 0.761, "speechiness": 0.876,
            "instrumentalness": 0.010, "loudness": -11.3,
        },
    }

    # Normalise features with very different scales so they contribute equally
    scale_factors = {
        "valence": 1.0, "energy": 1.0, "danceability": 1.0,
        "tempo": 0.007, "acousticness": 1.0, "speechiness": 1.0,
        "instrumentalness": 1.0, "loudness": 0.02,
    }

    assigned   = {}
    used_moods = set()

    # Build all (distance, cluster_id, mood) combos, sort closest-first
    scores = []
    for cid, profile in profiles.items():
        for mood, archetype in archetypes.items():
            dist = sum(
                ((profile[f] - archetype[f]) * scale_factors[f]) ** 2
                for f in FEATURES
            ) ** 0.5
            scores.append((dist, cid, mood))

    scores.sort()

    for dist, cid, mood in scores:
        if cid in assigned or mood in used_moods:
            continue
        assigned[cid]  = mood
        used_moods.add(mood)
        if len(assigned) == len(profiles):
            break

    return assigned


MOOD_MERGE = {
    "epic_instrumental": "epic",   # both epic clusters → same mood button
    "comedy":            None,     # None = drop entirely
}



# ── 7. Save everything ────────────────────────────────────────────────
# Update save() to store MOOD_MERGE alongside everything else:
def save(km, scaler, clf, profiles, cluster_names):
    with open("mood_classifier_kmeans.pkl", "wb") as f:
        pickle.dump({
            "kmeans":        km,
            "scaler":        scaler,
            "classifier":    clf,
            "profiles":      profiles,
            "features":      FEATURES,
            "cluster_names": cluster_names,
            "mood_merge":    MOOD_MERGE,
        }, f)
    print("\nSaved mood_classifier_kmeans.pkl")

# ── 8. Run full pipeline ──────────────────────────────────────────────
# Update __main__ to unpack the new return value:
if __name__ == "__main__":
    df                           = load_data("dataset.csv")
    X, scaler                    = scale(df)
    best_k                       = find_best_k(X)
    km, labels                   = cluster(X, best_k)
    profiles, df_labeled, cluster_names = profile_clusters(df, labels, best_k)
    clf                          = train_classifier(df_labeled)
    save(km, scaler, clf, profiles, cluster_names)

    print("\n✓ Retraining complete.")
    print(f"  Clusters: {best_k}  |  Tracks: {len(df):,}")
    print("  Mood assignments:")
    for cid, name in cluster_names.items():
        count = profiles[cid]["track_count"]
        merge_note = f" → merged into '{MOOD_MERGE[name]}'" if name in MOOD_MERGE and MOOD_MERGE[name] else \
                     " → DROPPED" if name in MOOD_MERGE else ""
        print(f"    Cluster {cid}: {name:<20} ({count:,} tracks){merge_note}")
