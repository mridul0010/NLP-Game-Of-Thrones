import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_text():
    tokens = joblib.load(os.path.join(BASE_DIR, "models", "tokens.joblib"))
    text = " ".join(tokens)
    return text, tokens
