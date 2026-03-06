from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import gdown

# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_DIR = "models"
GPT2_DIR = os.path.join(MODEL_DIR, "gpt2")

# Google Drive folder ID (replace with your folder id)
FOLDER_ID = "12Ioprj1RJ0uzigNiZ9S8qdBywyCJ_DpQ"

# --------------------------------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# --------------------------------------------------

def download_model():

    if not os.path.exists(GPT2_DIR):

        os.makedirs(MODEL_DIR, exist_ok=True)

        gdown.download_folder(
            id=FOLDER_ID,
            output=GPT2_DIR,
            quiet=False,
            use_cookies=False
        )

# --------------------------------------------------
# LOAD GENERATOR
# --------------------------------------------------

def load_generator():

    download_model()

    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_DIR)
    model = GPT2LMHeadModel.from_pretrained(GPT2_DIR)

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    return tokenizer, model

# --------------------------------------------------
# TEXT GENERATION
# --------------------------------------------------

def generate_text(
        prompt,
        tokenizer,
        model,
        max_len=150,
        temperature=1.0,
        top_k=80,
        top_p=0.95
):

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():

        output = model.generate(
            inputs,
            max_length=max_len,
            do_sample=True,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
