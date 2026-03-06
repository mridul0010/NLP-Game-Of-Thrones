import joblib


def load_generator():
    tokenizer = joblib.load("models/gpt2_tokenizer.joblib")
    model = joblib.load("models/gpt2_model.joblib")

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
