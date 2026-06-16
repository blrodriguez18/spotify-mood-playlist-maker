# spotify-mood-playlist-maker

### Instructions to run the website locally:
This project is a Flask app, so the Python server has to run locally instead of using GitHub Pages. The app serves the frontend at `/`, handles Spotify login/callback routes, and uses local model files to generate playlists. The training script builds and saves `mood_classifier_kmeans.pkl`.

### 1) Install Python dependencies

Create and activate a virtual environment, then install the required packages:
- bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt

### 2) Create a .env file in the project root:
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:3000/callback
FLASK_SECRET_KEY=your_random_secret_key

### 3) Train the model (required once before first launch):
- bash
python train_model_kmeans.py

### 4) Create a Spotify Developer application and add the following Redirect URI:
http://127.0.0.1:3000/callback
Copy the Client ID and Client Secret into your .env file.

### 5) Run the Flask server
- bash
python app.py

### 6) Open the Site
http://127.0.0.1:3000
