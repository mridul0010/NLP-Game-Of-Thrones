from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_generator():

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    return tokenizer, model


def generate_text(prompt, tokenizer, model, max_len):

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        inputs,
        max_length=max_len,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)