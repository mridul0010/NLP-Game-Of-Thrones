import joblib
import os

def load_text():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tokens_path = os.path.join(BASE_DIR, "models", "tokens.joblib")

    tokens = joblib.load(tokens_path)

    text = " ".join(tokens)

    return text, tokens
