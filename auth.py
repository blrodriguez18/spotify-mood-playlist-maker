import os
import urllib.parse
import secrets
import requests
from dotenv import load_dotenv
from flask import Flask, redirect, request, session

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

    return f"""
        <h2>Auth successful!</h2>
        <p><b>Access token:</b> {tokens['access_token'][:40]}...</p>
        <p><b>Expires in:</b> {tokens['expires_in']} seconds</p>
        <p><b>Refresh token:</b> {tokens['refresh_token'][:40]}...</p>
    """


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(port=3000, debug=True)