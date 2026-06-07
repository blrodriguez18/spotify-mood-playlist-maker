import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle

# ── 1. Mood mapping — all 114 genres → 6 moods ───────────────────────
GENRE_TO_MOOD = {
    # ENERGIZED — upbeat, driving, workout
    "dance":             "energized",
    "dancehall":         "energized",
    "disco":             "energized",
    "edm":               "energized",
    "electro":           "energized",
    "electronic":        "energized",
    "funk":              "energized",
    "garage":            "energized",
    "groove":            "energized",
    "house":             "energized",
    "j-dance":           "energized",
    "chicago-house":     "energized",
    "deep-house":        "energized",
    "detroit-techno":    "energized",
    "drum-and-bass":     "energized",
    "german":            "energized",
    "dubstep":           "energized",
    "hardstyle":         "energized",
    "minimal-techno":    "energized",
    "progressive-house": "energized",
    "techno":            "energized",
    "trance":            "energized",
    "breakbeat":         "energized",
    "club":              "energized",
    "idm":               "energized",

    # HAPPY — bright, feel-good, positive
    "happy":             "happy",
    "pop":               "happy",
    "indie-pop":         "happy",
    "power-pop":         "happy",
    "synth-pop":         "happy",
    "k-pop":             "happy",
    "j-pop":             "happy",
    "j-idol":            "happy",
    "cantopop":          "happy",
    "mandopop":          "happy",
    "disney":            "happy",
    "children":          "happy",
    "kids":              "happy",
    "comedy":            "happy",
    "show-tunes":        "happy",
    "pop-film":          "happy",
    "party":             "happy",

    # CHILL — relaxed, easy, background
    "chill":             "chill",
    "acoustic":          "chill",
    "ambient":           "chill",
    "folk":              "chill",
    "guitar":            "chill",
    "indie":             "chill",
    "jazz":              "chill",
    "new-age":           "chill",
    "piano":             "chill",
    "singer-songwriter": "chill",
    "songwriter":        "chill",
    "soul":              "chill",
    "trip-hop":          "chill",
    "bossa-nova":        "chill",
    "dub":               "chill",
    "reggae":            "chill",
    "ska":               "chill",
    "world-music":       "chill",
    "mpb":               "chill",
    "pagode":            "chill",
    "samba":             "chill",
    "forro":             "chill",
    "sertanejo":         "chill",
    "romance":           "chill",
    "french":            "chill",
    "swedish":           "chill",
    "malay":             "chill",
    "brazil":            "chill",
    "latin":             "chill",
    "latino":            "chill",
    "afrobeat":          "chill",
    "reggaeton":         "chill",
    "salsa":             "chill",
    "tango":             "chill",
    "study":             "chill",
    "ambient":           "chill",
    "instrumental":      "chill",
    "new-age":           "chill",  

    # MELANCHOLIC — sad, emotional, reflective
    "sad":               "melancholic",
    "emo":               "melancholic",
    "classical":         "melancholic", 
    "piano":             "melancholic", 
    "goth":              "melancholic",
    "blues":             "melancholic",
    "alternative":       "melancholic",
    "alt-rock":          "melancholic",
    "grunge":            "melancholic",
    "psych-rock":        "melancholic",
    "r-n-b":             "melancholic",
    "sleep":             "melancholic",
    "opera":             "melancholic",
    "classical":         "melancholic",
    "gospel":            "melancholic",
    "bluegrass":         "melancholic",
    "country":           "melancholic",
    "honky-tonk":        "melancholic",
    "rockabilly":        "melancholic",
    "iranian":           "melancholic",
    "indian":            "melancholic",
    "turkish":           "melancholic",
    "spanish":           "melancholic",
    "anime":             "melancholic",
    "j-rock":            "melancholic",

    # AGGRESSIVE — intense, hard, high energy
    "metal":             "aggressive",
    "black-metal":       "aggressive",
    "death-metal":       "aggressive",
    "hard-rock":         "aggressive",
    "hardcore":          "aggressive",
    "heavy-metal":       "aggressive",
    "metalcore":         "aggressive",
    "grindcore":         "aggressive",
    "industrial":        "aggressive",
    "punk":              "aggressive",
    "punk-rock":         "aggressive",
    "rock":              "aggressive",
    "rock-n-roll":       "aggressive",
    "hip-hop":           "aggressive",
    "british":           "aggressive",
}



FEATURES = ["valence", "energy", "danceability", "tempo",
            "acousticness", "speechiness", "instrumentalness",
            "liveness", "loudness",]


# ── 2. Load and prepare data ──────────────────────────────────────────
def load_data(path="dataset.csv"):
    df = pd.read_csv(path)  

    df["mood"] = df["track_genre"].map(GENRE_TO_MOOD)
    before = len(df)
    df = df.dropna(subset=["mood"])
    print(f"Mapped {len(df)} / {before} tracks to moods "
          f"({before - len(df)} genres unmapped and dropped)")

    # Show class distribution so we can spot imbalances
    print("\nMood distribution:")
    mood_list = df["mood"].value_counts().index.tolist()
    val_list = df["mood"].value_counts().tolist()
    for i in range(len(df["mood"].value_counts())):
        print(f"{mood_list[i]:<15} {val_list[i]:>6} : {100*val_list[i]/len(df):.2f}%")

    return df

# ── 3. Train the classifier ───────────────────────────────────────────
def train(df):
    X = df[FEATURES].values
    y = df["mood"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clf = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=0.8, stratify=y_encoded, random_state=18)

    print(f"\nTraining on {len(X_train)} tracks, testing on {len(X_test)}...")

    clf = RandomForestClassifier(
        n_estimators=500,       # more trees = more stable predictions
        max_depth=20,           # allow deeper trees to capture nuance
        min_samples_leaf=5,     # prevent overfitting on noise
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Evaluate
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual: {c}" for c in le.classes_],
        columns=[f"pred: {c}" for c in le.classes_]
    )
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    # Feature importance
    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 100)
        print(f"  {feat:<20} {imp:.4f}  {bar}")


    return clf, le

# ── 4. Save the classifier ────────────────────────────────────────────
def save(clf, le):
    with open("mood_classifier.pkl", "wb") as f:
        pickle.dump({"classifier": clf, "label_encoder": le,
                     "features": FEATURES}, f)
    print("\nSaved mood_classifier.pkl")


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df       = load_data()
    clf, le  = train(df)
    save(clf, le)