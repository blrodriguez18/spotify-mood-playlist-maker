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
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
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
SCOPES = "playlist-modify-public playlist-modify-private user-read-private"

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
    return render_template("index2.html", logged_in=logged_in)


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
    try:
        if not _get_access_token():
            return jsonify({"error": "Not authenticated. Please /login first."}), 401

        body = request.get_json(silent=True) or {}
        mood       = body.get("mood", "unknown")
        track_list = body.get("tracks", [])

        if not track_list:
            return jsonify({"error": "No tracks provided"}), 400

        # ── Step 1: get Spotify user ID ────────────
        me_resp = _spotify_get("/me")
        print("ME STATUS:", me_resp.status_code)
        print("ME BODY:", me_resp.json())
        import base64, json as _json
        token = _get_access_token()
        try:
            payload = token.split('.')[1]
            payload += '=' * (4 - len(payload) % 4)  # fix padding
            decoded = _json.loads(base64.urlsafe_b64decode(payload))
            print("TOKEN PAYLOAD:", decoded)
        except Exception as ex:
            print("Could not decode token:", ex)

        if me_resp.status_code != 200:
            return jsonify({"error": "Could not fetch Spotify user profile"}), 502
        user_id = me_resp.json()["id"]

        # ── Step 2: create playlist ────────────────
        playlist_name = f"{mood.capitalize()} Vibes 🎵"
        create_resp = _spotify_post(
            "/me/playlists", 
            json={
                "name":        playlist_name,
                "description": f"Auto-generated {mood} playlist — powered by ML",
                "public":      False,
            },
        )
        print("CREATE STATUS:", create_resp.status_code)
        print("CREATE BODY:", create_resp.json())

        if create_resp.status_code not in (200, 201):
            return jsonify({
                "error":   "Could not create playlist",
                "details": create_resp.json(),
            }), 502

        playlist_id  = create_resp.json()["id"]
        playlist_url = create_resp.json()["external_urls"]["spotify"]

        # ── Step 3: search for each track ─────────
        uris      = []
        not_found = []

        for track in track_list:
            name   = track.get("track_name", "")
            artist = track.get("artists", "")
            query  = f'track:"{name}"'
            if artist:
                first_artist = str(artist).split(";")[0].strip()
                query += f' artist:"{first_artist}"'

            search_resp = _spotify_get(
                "/search",
                params={"q": query, "type": "track", "limit": 1},
            )
            # print(f"SEARCH '{name}': status={search_resp.status_code}")
            if search_resp.status_code == 200:
                items = search_resp.json().get("tracks", {}).get("items", [])
                if items:
                    uri = items[0]["uri"]
                    print(f"  FOUND: {uri} — {items[0]['name']}")
                    uris.append(uri)
                else:
                    not_found.append(name)

        if not uris:
            return jsonify({"error": "Could not resolve any tracks on Spotify"}), 404

        # ── Step 4: add tracks ─────────────────────
        # Check what scopes the token actually has
        scope_resp = _spotify_get("/me")
        print("TOKEN SCOPES:", scope_resp.headers.get("X-OAuth-Scopes", "not found"))
        print("GRANTED SCOPES:", scope_resp.headers.get("X-Accepted-OAuth-Scopes", "not found"))
        print(f"Total URIs to add: {len(uris)}")
        print("First 3 URIs:", uris[:3])
        for i in range(0, len(uris), 100):
            chunk    = uris[i:i+100]
            add_resp = _spotify_post(
                f"/playlists/{playlist_id}/tracks",
                json={"uris": chunk},
            )
            print(f"ADD STATUS: {add_resp.status_code}")  # ← add this
            print(f"ADD BODY: {add_resp.json()}")
            if add_resp.status_code not in (200, 201):
                return jsonify({
                    "error":   "Failed to add tracks to playlist",
                    "details": add_resp.json(),
                }), 502

        return jsonify({
            "success":       True,
            "playlist_url":  playlist_url,
            "playlist_name": playlist_name,
            "tracks_added":  len(uris),
            "not_found":     not_found,
        })

    except Exception as e:
        import traceback
        print("SAVE ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 502



from flask import send_file
import io

@app.route("/export", methods=["POST"])
def export_playlist():
    body      = request.get_json(silent=True) or {}
    mood      = body.get("mood", "playlist")
    tracks    = body.get("tracks", [])

    if not tracks:
        return jsonify({"error": "No tracks provided"}), 400

    # Build M3U content
    lines = ["#EXTM3U", f"#PLAYLIST:{mood.capitalize()} Vibes"]
    for t in tracks:
        name   = t.get("track_name", "Unknown")
        artist = t.get("artists", "Unknown")
        lines.append(f"#EXTINF:-1,{artist} - {name}")
        lines.append(f"# spotify search: {artist} {name}")

    m3u_content = "\n".join(lines)
    buffer = io.BytesIO(m3u_content.encode("utf-8"))
    buffer.seek(0)

    filename = f"{mood}_vibes.m3u"
    return send_file(
        buffer,
        mimetype="audio/x-mpegurl",
        as_attachment=True,
        download_name=filename,
    )



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
