import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import pickle

FEATURES = ["valence", "energy", "danceability", "tempo",
            "acousticness", "speechiness", "instrumentalness",
            "loudness"]

CLUSTER_NAMES = {
    0: "epic",
    1: "energized",
    2: "melancholic",
    3: "happy",
    4: "chill",
}

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

        print(f"\nCluster {cid} — {CLUSTER_NAMES[cid].upper()}  ({len(members)} tracks)")
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

    return profiles, df

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

# ── 7. Save everything ────────────────────────────────────────────────
def save(km, scaler, clf, profiles):
    with open("mood_classifier_kmeans.pkl", "wb") as f:
        pickle.dump({
            "kmeans":        km,
            "scaler":        scaler,
            "classifier":    clf,
            "profiles":      profiles,
            "features":      FEATURES,
            "cluster_names": CLUSTER_NAMES,
        }, f)
    print("\nSaved mood_classifier_kmeans.pkl")

# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df              = load_data()
    X, scaler       = scale(df)
    best_k          = find_best_k(X)
    km, labels      = cluster(X, best_k)
    profiles, df_labeled = profile_clusters(df, labels, best_k)
    clf             = train_classifier(df_labeled)
    save(km, scaler, clf, profiles)