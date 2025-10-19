import sys, os
import re

# Add project root to sys.path BEFORE importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import DATA_PATH  # <-- now it will work


def clean_poem(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace("’", "'")
    return text


def format_dataset(input_path=DATA_PATH, output_path="data/cleaned_poems.txt"):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    poems = re.split(r'<\|endofpoem\|>', raw_text)
    cleaned_poems = []

    for poem in poems:
        poem = poem.strip()
        if poem:
            poem = clean_poem(poem)
            poem = f"<|startofpoem|>\n{poem}\n<|endofpoem|>"
            cleaned_poems.append(poem)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(cleaned_poems))

    print(f"✅ Cleaned and formatted {len(cleaned_poems)} poems.")
    return output_path


if __name__ == "__main__":
    format_dataset()
