# import json
# import pandas as pd


# # Load your library
# with open("track_data.json") as f:
#     tracks = json.load(f)

# # Load the kaggle dataset
# df = pd.read_csv("dataset.csv")

# # Normalize for matching — lowercase, strip whitespace
# df["track_name_norm"]  = df["track_name"].str.lower().str.strip()
# df["artist_norm"]      = df["artists"].str.lower().str.strip()

# matched   = 0
# unmatched = []

# for t in tracks:
#     name   = t["name"].lower().strip()
#     artist = t["artist"].lower().strip()

#     hit = df[(df["track_name_norm"] == name) &
#              (df["artist_norm"].str.contains(artist, regex=False))]

#     if len(hit) > 0:
#         matched += 1
#     else:
#         unmatched.append(f"{t['name']} — {t['artist']}")

# print(f"Matched:   {matched} / {len(tracks)}")
# print(f"Unmatched: {len(unmatched)}")
# print(f"\nFirst 20 unmatched:")
# for t in unmatched[:20]:
#     print(f"  {t}")

import pandas as pd

df = pd.read_csv("dataset.csv")
df["track_name_norm"] = df["track_name"].str.lower().str.strip()

# Check each one individually
checks = [
    "apologize",
    "good days",
    "boyfriend",
    "break free",
    "gods",
]

for name in checks:
    hits = df[df["track_name_norm"] == name]
    print(f"\n'{name}' — {len(hits)} matches:")
    for _, row in hits.iterrows():
        print(f"  {row['track_name']} — {row['artists']}")