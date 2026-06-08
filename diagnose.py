# import json
# import numpy as np

# with open("track_data.json") as f:
#     tracks = json.load(f)

# features = ["valence", "energy", "danceability", "acousticness"]

# print("Feature ranges across your library:")
# print(f"{'Feature':<15} {'Min':>6} {'Max':>6} {'Mean':>6} {'Std':>6}")
# print("-" * 45)
# for feat in features:
#     vals = [t[feat] for t in tracks if t.get(feat) is not None]
#     print(f"{feat:<15} {min(vals):>6.3f} {max(vals):>6.3f} "
#           f"{np.mean(vals):>6.3f} {np.std(vals):>6.3f}")

# print("\nTempo range:")
# tempos = [t["tempo"] for t in tracks if t.get("tempo")]
# print(f"  Min: {min(tempos):.0f} BPM  Max: {max(tempos):.0f} BPM  "
#       f"Mean: {np.mean(tempos):.0f} BPM  Std: {np.std(tempos):.0f}")


# # add to diagnose.py and rerun
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# import numpy as np

# X = np.array([[t["valence"], t["energy"], t["danceability"],
#                t["tempo"]/200.0, t["acousticness"]] for t in tracks])

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# print(f"\n{'k':<5} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>19}")
# print("-" * 55)
# for k in range(3, 10):
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = km.fit_predict(X_scaled)
#     sil = silhouette_score(X_scaled, labels)
#     db  = davies_bouldin_score(X_scaled, labels)
#     ch  = calinski_harabasz_score(X_scaled, labels)
#     print(f"{k:<5} {sil:>12.4f} {db:>16.4f} {ch:>19.1f}")

# print("\nWhat good looks like:")
# print("  Silhouette      → higher is better  (max 1.0)")
# print("  Davies-Bouldin  → lower is better   (min 0.0)")
# print("  Calinski-Harabasz → higher is better (no upper bound)")


# import json

# with open("track_data.json") as f:
#     tracks = json.load(f)

# needed = ["valence", "energy", "danceability", "tempo",
#           "acousticness", "speechiness", "instrumentalness", "loudness"]

# for feat in needed:
#     count = sum(1 for t in tracks if t.get(feat) is not None)
#     print(f"{feat:<20} {count:>3} / {len(tracks)} tracks have it")