import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.config import OUTPUT_DIR


def generate_poem(prompt, max_length=150, temperature=0.8, top_k=50, top_p=0.9):
    print("🎭 Loading trained VanshVerse model...")
    tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
    model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)

    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n🖋️ Generated Poem:\n")
    print(generated_text)

if __name__ == "__main__":
    user_prompt = input("Enter a theme, phrase, or first line for your poem:\n> ")
    generate_poem(user_prompt)
