import regex as re
import os

def load_text():

    files = [
        "data/001ssb.txt",
        "data/002ssb.txt",
        "data/003ssb.txt",
        "data/004ssb.txt",
        "data/005ssb.txt"
    ]

    raw_text = ""

    for file in files:

        if os.path.exists(file):

            with open(file,"r",encoding="latin-1") as f:
                raw_text += f.read()

    clean = raw_text.lower()
    clean = re.sub(r"\n+"," ",clean)

    alpha = re.sub(r"[^a-zA-Z\s]","",clean)

    tokens = alpha.split()

    return clean, tokens