import torch
import random
import numpy as np
import os

def set_seed(seed=42):
    """
    Ensures reproducibility — so your model gives the same results every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_poems_in_file(file_path):
    """
    Counts how many poems exist in the cleaned text file.
    (It searches for <|startofpoem|> markers)
    """
    if not os.path.exists(file_path):
        return 0

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        return content.count("<|startofpoem|>")

def check_gpu():
    """
    Checks if GPU (CUDA) is available for faster training.
    """
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("GPU not available. Using CPU.")
        return torch.device("cpu")

def load_model_and_tokenizer(model_class, tokenizer_class, model_dir):
    """
    Loads a saved model + tokenizer cleanly.
    """
    model = model_class.from_pretrained(model_dir)
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer

def generate_poem(prompt, model, tokenizer, max_length=200, temperature=0.8, top_p=0.95):
    """
    Generates a poem given a short text prompt.
    Uses nucleus sampling (top-p) for creativity.
    """
    input_ids = tokenizer.encode(f"<|startofpoem|>{prompt}", return_tensors="pt")

    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    poem = decoded.split("<|endofpoem|>")[0]
    return poem.strip()
