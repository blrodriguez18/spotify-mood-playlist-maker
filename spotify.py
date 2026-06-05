import requests
import json

# Every Spotify API call needs this base URL
BASE_URL = "https://api.spotify.com/v1"


def get_headers(access_token):
    """Build the auth header required by every Spotify API request."""
    return {"Authorization": f"Bearer {access_token}"}


# ── 1. Get the user's top tracks ─────────────────────────────────────
def get_top_tracks(access_token, time_range="short_term", limit=50):
    """
    Fetch the user's most-listened-to tracks.
    time_range options:
      - short_term  = last 4 weeks
      - medium_term = last 6 months
      - long_term   = all time
    """
    url = f"{BASE_URL}/me/top/tracks"
    params = {"time_range": time_range, "limit": limit}

    response = requests.get(url, headers=get_headers(access_token), params=params)
    data = response.json()

    # Pull out just the fields we care about from each track
    tracks = []
    for item in data.get("items", []):
        tracks.append({
            "id":     item["id"],
            "name":   item["name"],
            "artist": item["artists"][0]["name"],  # just the first artist
        })

    return tracks


# ── 2. Get recently played tracks ────────────────────────────────────
def get_recently_played(access_token, limit=50):
    """
    Fetch the last N tracks the user played.
    This is great for detecting *current* mood.
    """
    url = f"{BASE_URL}/me/player/recently-played"
    params = {"limit": limit}

    response = requests.get(url, headers=get_headers(access_token), params=params)
    data = response.json()

    tracks = []
    for item in data.get("items", []):
        track = item["track"]
        tracks.append({
            "id":     track["id"],
            "name":   track["name"],
            "artist": track["artists"][0]["name"],
        })

    return tracks


# ── 3. Get audio features for a list of tracks ───────────────────────
def get_audio_features(access_token, track_ids):
    """
    Fetch Spotify's audio analysis for up to 100 tracks at once.
    This returns the ML features we'll use: valence, energy, etc.
    """
    url = f"{BASE_URL}/audio-features"
    # Spotify expects a comma-separated string of IDs
    params = {"ids": ",".join(track_ids)}

    response = requests.get(url, headers=get_headers(access_token), params=params)
    data = response.json()

    features = []
    for item in data.get("audio_features", []):
        if item is None:
            continue  # occasionally Spotify returns null for a track — skip it
        features.append({
            "id":             item["id"],
            "valence":        item["valence"],        # 0=sad, 1=happy
            "energy":         item["energy"],         # 0=calm, 1=intense
            "danceability":   item["danceability"],   # 0=stiff, 1=danceable
            "tempo":          item["tempo"],          # BPM
            "acousticness":   item["acousticness"],   # 0=electric, 1=acoustic
            "speechiness":    item["speechiness"],    # 0=music, 1=spoken word
        })

    return features


# ── 4. Combine tracks + features into one dataset ────────────────────
def build_feature_matrix(access_token):
    """
    Master function: pulls top tracks + recent plays, deduplicates,
    fetches audio features, and returns one combined list of dicts.
    """
    print("Fetching top tracks (short term)...")
    top_short  = get_top_tracks(access_token, time_range="short_term")

    print("Fetching top tracks (medium term)...")
    top_medium = get_top_tracks(access_token, time_range="medium_term")

    print("Fetching recently played...")
    recent     = get_recently_played(access_token)

    # Merge all three lists and deduplicate by track ID
    all_tracks = {t["id"]: t for t in top_short + top_medium + recent}
    unique_tracks = list(all_tracks.values())
    print(f"Total unique tracks: {len(unique_tracks)}")

    # Spotify's audio-features endpoint handles max 100 IDs at once
    # so we split into chunks if needed
    track_ids = [t["id"] for t in unique_tracks]
    all_features = []
    for i in range(0, len(track_ids), 100):
        chunk = track_ids[i:i+100]
        all_features += get_audio_features(access_token, chunk)

    # Build a lookup dict: id -> features
    features_by_id = {f["id"]: f for f in all_features}

    # Merge track info + audio features into one object per track
    matrix = []
    for track in unique_tracks:
        tid = track["id"]
        if tid in features_by_id:
            row = {**track, **features_by_id[tid]}  # merge both dicts
            matrix.append(row)

    with open("track_data.json", "w") as f:
        json.dump(matrix, f, indent=2)
    print("Saved to track_data.json")

    print(f"Feature matrix complete: {len(matrix)} tracks × 6 audio features")
    return matrix