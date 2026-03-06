import joblib

def load_text():

    tokens = joblib.load("models/tokens.joblib")

    text = " ".join(tokens)

    return text, tokens
