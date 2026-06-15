"""
app.py
──────
Flask web server for the mood-based Spotify playlist generator.

Routes
──────
GET  /              → serves index.html
GET  /login         → starts Spotify OAuth PKCE flow
GET  /callback      → handles Spotify redirect, exchanges code for token
GET  /generate/<mood> → classifies tracks, returns playlist JSON
POST /export        → generates and downloads an M3U playlist file
GET  /logout        → clears the session
"""

import base64
import hashlib
import io
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
    send_file,
    session,
    url_for,
)

from playlist_generator import get_playlist_tracks

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = False
app.config["SESSION_COOKIE_HTTPONLY"] = True

SPOTIFY_AUTH_URL  = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE  = "https://api.spotify.com/v1"

CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:3000/callback")
SCOPES        = "playlist-modify-public playlist-modify-private user-read-private"


# ── PKCE HELPERS ──────────────────────────────────────────────────────

def _generate_code_verifier(length: int = 64) -> str:
    return secrets.token_urlsafe(length)


def _generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ── TOKEN HELPERS ─────────────────────────────────────────────────────

def _get_access_token() -> str | None:
    return session.get("access_token")


def _refresh_access_token() -> bool:
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
    if "refresh_token" in token_data:
        session["refresh_token"] = token_data["refresh_token"]
    return True


def _spotify_get(endpoint: str, **kwargs) -> requests.Response:
    token   = _get_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{SPOTIFY_API_BASE}{endpoint}", headers=headers, timeout=10, **kwargs
    )
    if resp.status_code == 401 and _refresh_access_token():
        headers["Authorization"] = f"Bearer {_get_access_token()}"
        resp = requests.get(
            f"{SPOTIFY_API_BASE}{endpoint}", headers=headers, timeout=10, **kwargs
        )
    return resp


# ── ROUTES — AUTH ─────────────────────────────────────────────────────

@app.route("/")
def index():
    logged_in = bool(_get_access_token())
    return render_template("index.html", logged_in=logged_in)


@app.route("/login")
def login():
    code_verifier  = _generate_code_verifier()
    code_challenge = _generate_code_challenge(code_verifier)
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
        "show_dialog":           "true",
    }
    return redirect(f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}")


@app.route("/callback")
def callback():
    returned_state = request.args.get("state", "")
    if returned_state != session.pop("oauth_state", None):
        return jsonify({"error": "State mismatch — possible CSRF attack"}), 400

    error = request.args.get("error")
    if error:
        return jsonify({"error": f"Spotify auth error: {error}"}), 400

    code          = request.args.get("code")
    code_verifier = session.pop("code_verifier", None)
    if not code or not code_verifier:
        return jsonify({"error": "Missing code or verifier"}), 400

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
        return jsonify({"error": "Token exchange failed", "details": resp.json()}), 400

    token_data = resp.json()
    session["access_token"]  = token_data["access_token"]
    session["refresh_token"] = token_data.get("refresh_token")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ── ROUTES — PLAYLIST ─────────────────────────────────────────────────

@app.route("/generate/<mood>")
def generate(mood: str):
    n = request.args.get("n", 25, type=int)
    try:
        tracks = get_playlist_tracks(mood, n=n)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"mood": mood, "count": len(tracks), "tracks": tracks})


@app.route("/export", methods=["POST"])
def export_playlist():
    body   = request.get_json(silent=True) or {}
    mood   = body.get("mood", "playlist")
    tracks = body.get("tracks", [])

    if not tracks:
        return jsonify({"error": "No tracks provided"}), 400

    lines = ["#EXTM3U", f"#PLAYLIST:{mood.capitalize()} Vibes"]
    for t in tracks:
        name         = t.get("track_name", "Unknown")
        artist_raw   = t.get("artists", "Unknown")
        artist_clean = str(artist_raw).replace(";", ", ")
        duration_sec = int(t.get("duration_ms", 0) / 1000) or -1
        lines.append(f"#EXTINF:{duration_sec},{artist_clean} - {name}")
        lines.append(f"# spotify search: {artist_clean} {name}")

    buffer = io.BytesIO("\n".join(lines).encode("utf-8"))
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="audio/x-mpegurl",
        as_attachment=True,
        download_name=f"{mood}_vibes.m3u",
    )


# ── RUN ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)