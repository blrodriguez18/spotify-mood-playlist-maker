"""
app.py
──────
Flask web server for the mood-based Spotify playlist generator.

Routes
──────
GET  /                        → serves index.html
GET  /login                   → starts Spotify OAuth PKCE flow
GET  /callback                → handles Spotify redirect, exchanges code for token
GET  /generate/<mood>         → classifies tracks, returns playlist JSON
POST /save                    → creates the playlist in the user's Spotify account
GET  /logout                  → clears the session

Environment variables (loaded from .env)
─────────────────────────────────────────
SPOTIFY_CLIENT_ID      — from your Spotify Developer dashboard
SPOTIFY_CLIENT_SECRET  — from your Spotify Developer dashboard
SPOTIFY_REDIRECT_URI   — must be http://localhost:3000/callback
FLASK_SECRET_KEY       — any long random string for session signing
"""

import base64
import hashlib
import json
import os
import secrets
import urllib.parse

import requests
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from playlist_generator import get_playlist_tracks

# ──────────────────────────────────────────────
# 1.  APP SETUP
# ──────────────────────────────────────────────

load_dotenv()  # reads .env into os.environ

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
# ⚠️  If secret_key changes between restarts (when FLASK_SECRET_KEY isn't set),
#     existing sessions become invalid.  Always set FLASK_SECRET_KEY in .env.

# Spotify endpoints — these never change
SPOTIFY_AUTH_URL     = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL    = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE     = "https://api.spotify.com/v1"

# Pull credentials from environment
CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")   # used only in token exchange
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:3000/callback")
SCOPES        = "playlist-modify-public playlist-modify-private"


# ──────────────────────────────────────────────
# 2.  PKCE HELPERS
# ──────────────────────────────────────────────
# PKCE (Proof Key for Code Exchange) is the OAuth 2.0 flow recommended for
# apps where the client secret could theoretically be exposed.  It works by:
#   1. Generating a random `code_verifier` string
#   2. Hashing it with SHA-256 to produce a `code_challenge`
#   3. Sending the challenge to Spotify during /authorize
#   4. Sending the original verifier during /token — Spotify hashes it and
#      compares; if they match, it issues the access token
# Even though we're running server-side (and could use the basic Auth Code
# flow), PKCE is cleaner and avoids sending the client secret on every refresh.

def _generate_code_verifier(length: int = 64) -> str:
    """Random URL-safe string, 43–128 chars (Spotify requires ≥ 43)."""
    return secrets.token_urlsafe(length)


def _generate_code_challenge(verifier: str) -> str:
    """SHA-256 hash of verifier, base64url-encoded without padding."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ──────────────────────────────────────────────
# 3.  TOKEN HELPERS
# ──────────────────────────────────────────────

def _get_access_token() -> str | None:
    """Return the stored access token, or None if the user isn't logged in."""
    return session.get("access_token")


def _refresh_access_token() -> bool:
    """
    Use the stored refresh_token to get a new access_token.
    Returns True on success, False if the refresh failed (user must re-login).
    """
    refresh_token = session.get("refresh_token")
    if not refresh_token:
        return False

    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
            "client_id":     CLIENT_ID,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )

    if resp.status_code != 200:
        return False

    token_data = resp.json()
    session["access_token"] = token_data["access_token"]
    # Spotify sometimes issues a new refresh token; store it if present
    if "refresh_token" in token_data:
        session["refresh_token"] = token_data["refresh_token"]
    return True


def _spotify_get(endpoint: str, **kwargs) -> requests.Response:
    """
    Thin wrapper around requests.get that:
      • Injects the Bearer token header automatically
      • Retries once with a refreshed token on 401
    endpoint: everything after https://api.spotify.com/v1  (e.g. "/me")
    """
    token = _get_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(
        f"{SPOTIFY_API_BASE}{endpoint}",
        headers=headers,
        timeout=10,
        **kwargs,
    )

    if resp.status_code == 401:           # token expired
        if _refresh_access_token():
            headers["Authorization"] = f"Bearer {_get_access_token()}"
            resp = requests.get(
                f"{SPOTIFY_API_BASE}{endpoint}",
                headers=headers,
                timeout=10,
                **kwargs,
            )

    return resp


def _spotify_post(endpoint: str, **kwargs) -> requests.Response:
    """Same as _spotify_get but for POST requests."""
    token = _get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }

    resp = requests.post(
        f"{SPOTIFY_API_BASE}{endpoint}",
        headers=headers,
        timeout=10,
        **kwargs,
    )

    if resp.status_code == 401:
        if _refresh_access_token():
            headers["Authorization"] = f"Bearer {_get_access_token()}"
            resp = requests.post(
                f"{SPOTIFY_API_BASE}{endpoint}",
                headers=headers,
                timeout=10,
                **kwargs,
            )

    return resp


# ──────────────────────────────────────────────
# 4.  ROUTES — AUTH
# ──────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page frontend."""
    logged_in = bool(_get_access_token())
    return render_template("index.html", logged_in=logged_in)


@app.route("/login")
def login():
    """
    Step 1 of PKCE OAuth:
      • Generate code_verifier + code_challenge
      • Store verifier in session (needed later in /callback)
      • Redirect user to Spotify's authorization page
    """
    code_verifier  = _generate_code_verifier()
    code_challenge = _generate_code_challenge(code_verifier)

    # state is a random string we'll verify in /callback to prevent CSRF
    state = secrets.token_urlsafe(16)

    session["code_verifier"] = code_verifier
    session["oauth_state"]   = state

    params = {
        "client_id":             CLIENT_ID,
        "response_type":         "code",
        "redirect_uri":          REDIRECT_URI,
        "scope":                 SCOPES,
        "state":                 state,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    }

    auth_url = f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return redirect(auth_url)


@app.route("/callback")
def callback():
    """
    Step 2 of PKCE OAuth — Spotify redirects here with ?code=...&state=...

    We:
      1. Verify the state matches (CSRF check)
      2. Exchange the code + code_verifier for an access + refresh token
      3. Store both tokens in the Flask session
      4. Redirect to the homepage
    """
    # ── CSRF check ─────────────────────────────
    returned_state = request.args.get("state", "")
    if returned_state != session.pop("oauth_state", None):
        return jsonify({"error": "State mismatch — possible CSRF attack"}), 400

    # ── Error from Spotify? ────────────────────
    error = request.args.get("error")
    if error:
        return jsonify({"error": f"Spotify auth error: {error}"}), 400

    code          = request.args.get("code")
    code_verifier = session.pop("code_verifier", None)

    if not code or not code_verifier:
        return jsonify({"error": "Missing code or verifier"}), 400

    # ── Token exchange ─────────────────────────
    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
            "client_id":     CLIENT_ID,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )

    if resp.status_code != 200:
        return jsonify({
            "error":   "Token exchange failed",
            "details": resp.json(),
        }), 400

    token_data = resp.json()
    session["access_token"]  = token_data["access_token"]
    session["refresh_token"] = token_data.get("refresh_token")

    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    """Clear the session tokens and return to home."""
    session.clear()
    return redirect(url_for("index"))


# ──────────────────────────────────────────────
# 5.  ROUTES — PLAYLIST GENERATION
# ──────────────────────────────────────────────

@app.route("/generate/<mood>")
def generate(mood: str):
    """
    Classify tracks for the requested mood and return them as JSON.

    The frontend calls this to preview the playlist before saving it.
    No Spotify API call is made here — just the local ML model.

    Response shape
    ──────────────
    {
        "mood":   "happy",
        "count":  25,
        "tracks": [
            {
                "track_name": "...",
                "artists":    "...",
                "popularity": 87,
                "confidence": 0.923,
                ...                   ← all other CSV columns
            },
            ...
        ]
    }
    """
    n = request.args.get("n", 25, type=int)   # ?n=30 overrides default

    try:
        tracks = get_playlist_tracks(mood, n=n)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "mood":   mood,
        "count":  len(tracks),
        "tracks": tracks,
    })


@app.route("/save", methods=["POST"])
def save_playlist():
    """
    Create a new Spotify playlist and add the tracks to it.

    Expected request body (JSON)
    ─────────────────────────────
    {
        "mood":       "happy",
        "track_names": ["Song Title", ...],   ← used for display only
        "track_ids":   ["spotify:track:...", ...]  ← URIs to add
    }

    Because our dataset.csv doesn't contain Spotify track URIs, we search
    for each track by name + artist and resolve it to a URI here.

    Flow
    ────
    1. Verify the user is authenticated
    2. Get the user's Spotify ID  (/me)
    3. Create an empty playlist   (/users/{id}/playlists)
    4. Search for each track      (/search)  → collect URIs
    5. Add all URIs to playlist   (/playlists/{id}/tracks)
    6. Return the playlist URL

    Note: Spotify's /playlists/{id}/tracks accepts max 100 URIs per call.
    We chunk the list to stay within that limit.
    """
    if not _get_access_token():
        return jsonify({"error": "Not authenticated. Please /login first."}), 401

    body = request.get_json(silent=True) or {}
    mood         = body.get("mood", "unknown")
    track_list   = body.get("tracks", [])   # list of {track_name, artists, ...}

    if not track_list:
        return jsonify({"error": "No tracks provided"}), 400

    # ── Step 1: get Spotify user ID ────────────
    me_resp = _spotify_get("/me")
    if me_resp.status_code != 200:
        return jsonify({"error": "Could not fetch Spotify user profile"}), 502
    user_id = me_resp.json()["id"]

    # ── Step 2: create playlist ────────────────
    playlist_name = f"{mood.capitalize()} Vibes 🎵"
    create_resp   = _spotify_post(
        f"/users/{user_id}/playlists",
        json={
            "name":        playlist_name,
            "description": f"Auto-generated {mood} playlist — powered by ML",
            "public":      True,
        },
    )
    if create_resp.status_code not in (200, 201):
        return jsonify({
            "error":   "Could not create playlist",
            "details": create_resp.json(),
        }), 502

    playlist_id  = create_resp.json()["id"]
    playlist_url = create_resp.json()["external_urls"]["spotify"]

    # ── Step 3: search for each track ─────────
    # The CSV has track_name + artists but not Spotify URIs, so we search.
    # We take the top result and trust it — for a personal tool this is fine.
    uris     = []
    not_found = []

    for track in track_list:
        name    = track.get("track_name", "")
        artist  = track.get("artists",    "")

        # Build a targeted query: track:"Song Name" artist:"Artist"
        query = f'track:"{name}"'
        if artist:
            # Some entries have multiple artists separated by "; " — use only
            # the first one for the search query to avoid over-constraining it.
            first_artist = str(artist).split(";")[0].strip()
            query += f' artist:"{first_artist}"'

        search_resp = _spotify_get(
            "/search",
            params={"q": query, "type": "track", "limit": 1},
        )

        if search_resp.status_code == 200:
            items = search_resp.json().get("tracks", {}).get("items", [])
            if items:
                uris.append(items[0]["uri"])
            else:
                not_found.append(name)
        else:
            not_found.append(name)

    if not uris:
        return jsonify({"error": "Could not resolve any tracks on Spotify"}), 404

    # ── Step 4: add tracks (chunked at 100) ────
    CHUNK_SIZE = 100
    for i in range(0, len(uris), CHUNK_SIZE):
        chunk = uris[i : i + CHUNK_SIZE]
        add_resp = _spotify_post(
            f"/playlists/{playlist_id}/tracks",
            json={"uris": chunk},
        )
        if add_resp.status_code not in (200, 201):
            return jsonify({
                "error":   "Failed to add tracks to playlist",
                "details": add_resp.json(),
            }), 502

    return jsonify({
        "success":      True,
        "playlist_url": playlist_url,
        "playlist_name": playlist_name,
        "tracks_added": len(uris),
        "not_found":    not_found,   # tracks searched but not found on Spotify
    })


# ──────────────────────────────────────────────
# 6.  RUN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Port 3000 must match SPOTIFY_REDIRECT_URI in your .env and Spotify dashboard
    app.run(
        host="127.0.0.1",
        port=3000,
        debug=True,       # ← set False before any kind of production deployment
    )
