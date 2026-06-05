import os
import urllib.parse
import secrets
import requests
from dotenv import load_dotenv
from flask import Flask, redirect, request, session
from spotify import build_feature_matrix

# Load the credentials from your .env file
load_dotenv()

CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI")

# These are the permissions we're asking the user to grant us
SCOPES = " ".join([
    "user-read-recently-played",
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
])

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # needed for Flask sessions


# ── Route 1: the login page ──────────────────────────────────────────
@app.route("/login")
def login():
    # A random string we generate to protect against CSRF attacks
    state = secrets.token_hex(16)
    session["oauth_state"] = state

    # Build the URL that sends the user to Spotify's login page
    params = {
        "client_id":     CLIENT_ID,
        "response_type": "code",
        "redirect_uri":  REDIRECT_URI,
        "scope":         SCOPES,
        "state":         state,
    }
    auth_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)
    return redirect(auth_url)


# ── Route 2: Spotify sends the user back here after they log in ──────
@app.route("/callback")
def callback():
    # Make sure the state matches what we sent (security check)
    if request.args.get("state") != session.get("oauth_state"):
        return "State mismatch — possible CSRF attack.", 400

    # Spotify sends us a one-time 'code' we exchange for real tokens
    code = request.args.get("code")
    if not code:
        return f"Login failed: {request.args.get('error')}", 400

    # Exchange the code for an access token + refresh token
    token_response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )

    tokens = token_response.json()

    # Save tokens to the session (in a real app you'd save these to a database)
    session["access_token"]  = tokens["access_token"]
    session["refresh_token"] = tokens["refresh_token"]

    # Fetch the feature matrix right after login
    matrix = build_feature_matrix(session["access_token"])

    # Show the first 5 tracks so we can verify it's working
    preview = ""
    for track in matrix[:5]:
        preview += f"""
            <tr>
                <td>{track['name']}</td>
                <td>{track['artist']}</td>
                <td>{track['valence']:.2f}</td>
                <td>{track['energy']:.2f}</td>
                <td>{track['danceability']:.2f}</td>
                <td>{int(track['tempo'])} BPM</td>
            </tr>
        """

    return f"""
        <h2>✅ Auth + data fetch successful!</h2>
        <p>Pulled <b>{len(matrix)}</b> tracks with audio features.</p>
        <table border="1" cellpadding="8">
            <tr>
                <th>Track</th><th>Artist</th>
                <th>Valence 😊</th><th>Energy ⚡</th>
                <th>Danceability 💃</th><th>Tempo</th>
            </tr>
            {preview}
        </table>
        <p><i>Showing first 5 of {len(matrix)} tracks.</i></p>
    """


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(port=3000, debug=True)