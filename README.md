# 🖋️ VanshVerse — The AI Poet

VanshVerse is an AI language model fine-tuned on my original poetry.  
It generates new poems that reflect my writing tone, rhythm, and emotion.

## 🧠 Model
- Base: GPT-2 Medium (355M parameters)
- Framework: Hugging Face Transformers
- Training: 5 epochs, LR 5e-5
- Data: My own poem corpus (`data/poems.txt`)

## 🧰 Tech Stack
- Python
- Flask (Web Server)
- PyTorch + Transformers

## ⚙️ Usage
```bash
cd app
python main.py
