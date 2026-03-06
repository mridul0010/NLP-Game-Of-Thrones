import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_text():
    tokens_path = os.path.join(BASE_DIR, "models", "tokens.joblib")
    tokens = joblib.load(tokens_path)

    text = " ".join(tokens)
    return text, tokens
