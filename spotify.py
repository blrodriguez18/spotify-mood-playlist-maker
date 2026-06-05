import requests
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

BASE_URL      = "https://api.spotify.com/v1"
LASTFM_KEY    = os.getenv("LASTFM_API_KEY")
LASTFM_BASE   = "http://ws.audioscrobbler.com/2.0/"


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
    return tracks


# ── Last.fm: get mood tags for one track ─────────────────────────────
def get_lastfm_tags(track_name, artist_name):
    """
    Fetch crowd-sourced tags for a track from Last.fm.
    Tags look like: ['sad', 'indie', 'rainy day', 'chill', 'melancholic']
    We'll use these as mood signals in Phase 3.
    """
    params = {
        "method":  "track.getTopTags",
        "track":   track_name,
        "artist":  artist_name,
        "api_key": LASTFM_KEY,
        "format":  "json",
    }
    try:
        response = requests.get(LASTFM_BASE, params=params, timeout=5)
        data = response.json()

        tags = []
        for tag in data.get("toptags", {}).get("tag", [])[:8]:
            # Each tag has a 'count' (how many people tagged it) — only keep strong ones
            if int(tag.get("count", 0)) > 10:
                tags.append(tag["name"].lower())
        return tags

    except Exception:
        return []  # if Last.fm doesn't know the track, just return empty


# ── Master function ───────────────────────────────────────────────────
def build_feature_matrix(access_token):
    print("Fetching top tracks (short term)...")
    top_short  = get_top_tracks(access_token, time_range="short_term")

    print("Fetching top tracks (medium term)...")
    top_medium = get_top_tracks(access_token, time_range="medium_term")

    print("Fetching recently played...")
    recent     = get_recently_played(access_token)

    # Deduplicate by Spotify track ID
    all_tracks = {t["id"]: t for t in top_short + top_medium + recent}
    unique_tracks = list(all_tracks.values())
    print(f"Total unique tracks: {len(unique_tracks)}")

    # Fetch Last.fm tags for each track
    print("Fetching mood tags from Last.fm (this takes ~30 seconds)...")
    matrix = []
    for i, track in enumerate(unique_tracks):
        tags = get_lastfm_tags(track["name"], track["artist"])
        matrix.append({
            "id":     track["id"],
            "name":   track["name"],
            "artist": track["artist"],
            "tags":   tags,
        })
        # Be polite to the Last.fm API — don't hammer it
        time.sleep(0.2)

        # Print progress every 20 tracks
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(unique_tracks)} tracks processed...")

    print(f"Feature matrix complete: {len(matrix)} tracks with mood tags")

    with open("track_data.json", "w") as f:
        json.dump(matrix, f, indent=2)
    print("Saved to track_data.json")

    return matrix