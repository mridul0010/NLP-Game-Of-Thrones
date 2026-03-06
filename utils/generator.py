import joblib
import os
import gdown 
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gpt2_model.joblib")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "gpt2_tokenizer.joblib")

MODEL_URL = "https://drive.google.com/uc?id=1F6X8eqzENHQEFdp9BfgN6U87NrJ1ev8P"
TOKENIZER_URL = "https://drive.google.com/uc?id=1q0ehm32BF32FGTyifVuOWtHvgQtDzahe"


def download_models():

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    if not os.path.exists(TOKENIZER_PATH):
        gdown.download(TOKENIZER_URL, TOKENIZER_PATH, quiet=False)


def load_generator():

    download_models()

    tokenizer = joblib.load(TOKENIZER_PATH)
    model = joblib.load(MODEL_PATH)

    return tokenizer, model


def generate_text(prompt, tokenizer, model, max_len,
                  temperature=1.0, top_k=80, top_p=0.95):

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        inputs,
        max_length=max_len,
        min_length=40,
        do_sample=True,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
