import pickle
from pathlib import Path

# ── Path Resolution ──────────────────────────────────────────
# This gets the absolute path to the directory where this file sits
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Define absolute paths to your artifacts
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTOR_PATH = BASE_DIR / "models" / "vectorizer.pkl"
CONFIG_PATH = BASE_DIR / "models" / "config.pkl"

# ── Load Artifacts ────────────────────────────────────────────
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTOR_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)
        
    threshold = config.get('threshold', 0.5)
    print("ML Artifacts loaded successfully using absolute paths.")
    
except FileNotFoundError as e:
    print(f"ERROR: Could not find model files at {BASE_DIR}/models/")
    print(f"Details: {e}")
    # Optional: raise error to stop server if models are missing
    raise e

# ── Helper to extract why the model made the decision ──────────
def get_top_features(processed_text, n=3):
    """Returns the most important words from the text that influenced the model."""
    feature_names = vectorizer.get_feature_names_out()
    # Get indices of words present in the user text
    feature_index = vectorizer.vocabulary_
    words_in_text = set(processed_text.split())
    
    # Filter for words that the model actually knows
    present_features = [word for word in words_in_text if word in feature_index]
    
    if not present_features:
        return ["Generic structural patterns"]

    # Pair words with their importance weights from the model
    # Note: RF importances tell us how much a feature matters overall
    scored_features = []
    for word in present_features:
        idx = feature_index[word]
        score = model.feature_importances_[idx]
        scored_features.append((word, score))
    
    # Sort by importance
    scored_features.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in scored_features[:n]]
