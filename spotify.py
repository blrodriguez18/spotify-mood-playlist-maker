import requests
import os
import json
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()


BASE_URL     = "https://api.spotify.com/v1"
FREQBLOG_API_KEY = os.getenv("FREQBLOG_API_KEY")
FREQBLOG_BASE = "https://api.freqblog.com"

print("FREQBLOG KEY LOADED:", FREQBLOG_API_KEY)


def get_headers(access_token):
    return {"Authorization": f"Bearer {access_token}"}


# ── Spotify: get top tracks ───────────────────────────────────────────
def get_top_tracks(access_token, time_range="short_term", limit=50):
    url = f"{BASE_URL}/me/top/tracks"
    params = {"time_range": time_range, "limit": limit}
    response = requests.get(url, headers=get_headers(access_token), params=params)
    data = response.json()
    tracks = []
    for item in data.get("items", []):
        tracks.append({
            "id":     item["id"],
            "name":   item["name"],
            "artist": item["artists"][0]["name"],
        })
    print(f"Parsed {len(tracks)} tracks from {time_range}")
    return tracks


# ── Spotify: get recently played ─────────────────────────────────────
def get_recently_played(access_token, limit=50):
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
    print(f"Parsed {len(tracks)} recently played tracks")
    return tracks


# ── Freqblog: get audio features for one track ───────────────────────
def get_audio_features(track_name, artist_name):
    """
    Fetch audio features from freqblog — the drop-in replacement
    for Spotify's deprecated audio-features endpoint.
    Returns: valence, energy, danceability, tempo, mood, and more.
    """
    try:
        response = requests.get(
            f"{FREQBLOG_BASE}/lookup",
            params={
                "track":  track_name,
                "artist": artist_name,
            },
            headers={"X-Api-Key": FREQBLOG_API_KEY},  # ← moved here
            timeout=8,
        )
        data = response.json()

        print(f"Freqblog response for '{track_name}': {data}")

        # Return None if the track wasn't found
        if "error" in data or "valence" not in data:
            return None

        return {
            "valence":          float(data.get("valence", 0)),
            "energy":           float(data.get("energy", 0)),
            "danceability":     float(data.get("danceability", 0)),
            "tempo":            float(data.get("bpm", 0)),
            "acousticness":     float(data.get("acousticness", 0)),
            "mood":             data.get("mood", ""),
            "mood_vector":      data.get("mood_vector", None),
            "speechiness":      float(data.get("speechiness", 0)),      # ← add
            "instrumentalness": float(data.get("instrumentalness", 0)), # ← add
            "loudness":         float(data.get("loudness_db", 0)),      # ← add
        }
    except Exception as e:
        print(f"  Error fetching {track_name}: {e}")
        return None


# ── Master function ───────────────────────────────────────────────────
def build_feature_matrix(access_token):
    print("Fetching top tracks (short term)...")
    top_short  = get_top_tracks(access_token, time_range="short_term")

    print("Fetching top tracks (medium term)...")
    top_medium = get_top_tracks(access_token, time_range="medium_term")

    print("Fetching recently played...")
    recent     = get_recently_played(access_token)

    print("Fetching liked songs...")
    liked = get_liked_songs(access_token)

    # Deduplicate by Spotify track ID
    all_tracks = {t["id"]: t for t in top_short + top_medium + recent + liked}
    unique_tracks = list(all_tracks.values())
    print(f"Total unique tracks: {len(unique_tracks)}")
    

    print("Fetching audio features from freqblog...")
    matrix = []
    skipped = 0

    def fetch_one(track):
        features = get_audio_features(track["name"], track["artist"])
        if features is None:
            return None
        return {
            "id":               track["id"],
            "name":             track["name"],
            "artist":           track["artist"],
            "valence":          features["valence"],
            "energy":           features["energy"],
            "danceability":     features["danceability"],
            "tempo":            features["tempo"],
            "acousticness":     features["acousticness"],
            "mood":             features["mood"],
            "mood_vector":      features["mood_vector"],
            "speechiness":      features["speechiness"],      # ← add
            "instrumentalness": features["instrumentalness"], # ← add
            "loudness":         features["loudness"],         # ← add
        }

    # Fetch 10 tracks simultaneously instead of one at a time
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_one, track): track for track in unique_tracks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result:
                matrix.append(result)
            else:
                skipped += 1
            if completed % 20 == 0:
                print(f"  {completed}/{len(unique_tracks)} done ({len(matrix)} matched)...")

        # time.sleep(0.1)  # be polite to the API

    print(f"\nDone: {len(matrix)} tracks with features, {skipped} skipped")

    with open("track_data.json", "w") as f:
        json.dump(matrix, f, indent=2)
    print("Saved to track_data.json")

    return matrix

def get_liked_songs(access_token, limit=2000):
    """
    Fetch the user's saved/liked songs.
    Spotify returns max 50 per request so we page through them.
    """
    url = f"{BASE_URL}/me/tracks"
    tracks = []
    offset = 0

    while len(tracks) < limit:
        params = {"limit": 50, "offset": offset}
        response = requests.get(url, headers=get_headers(access_token), params=params)
        data = response.json()

        items = data.get("items", [])
        print(f"  Page at offset {offset}: got {len(items)} items, response keys: {list(data.keys())}")  # ← add
        if not items:
            break

        for item in items:
            track = item["track"]
            if track and track.get("id"):
                tracks.append({
                    "id":     track["id"],
                    "name":   track["name"],
                    "artist": track["artists"][0]["name"],
                })

        offset += 50
        print(f"  Fetched {len(tracks)} liked songs so far...")

        if len(items) < 50:
            break  # reached the end

    print(f"Total liked songs fetched: {len(tracks)}")
    return tracks

