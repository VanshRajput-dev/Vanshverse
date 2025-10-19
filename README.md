# 🖋️ VanshVerse — The AI Poet Trained on My Soul

> *“If I could teach a machine to feel,  
maybe it would write like me.”* — Vansh

---

## 🌌 About the Project

**VanshVerse** is not your average AI project.  
It’s a **Machine Learning model** trained entirely on my own poems —  
my heartbreaks, my musings, my chaos.  

The goal?  
To create an **AI poet** that *writes like me* — blending English, Hindi, and raw emotion — using the power of **GPT-2** fine-tuning.

So yeah, technically it’s a *language model*,  
but emotionally... it’s my clone with a pen. ✍️💔  

---

## 🧠 Tech Stack

| Component | Purpose |
|------------|----------|
| **Python** | The magic wand 🪄 |
| **PyTorch** | Makes GPT-2 do push-ups |
| **Hugging Face Transformers** | Gives our model the poetic brain |
| **Flask** | For the web app (optional) |
| **TQDM** | So you can *watch training pain in real time* |
| **Accelerate** | Because training without it = pain |
| **Your poems** | The real data — no ChatGPT copy pasta here |

---

## 🧩 Folder Structure

```
VanshVerse/
│
├── data/
│   ├── poems.txt              # Raw poems (yours)
│   └── cleaned_poems.txt      # Processed poems (auto-generated)
│
├── model/
│   └── vansh_poet_model/      # Saved fine-tuned GPT-2 model
│
├── src/
│   ├── preprocess.py          # Cleans + formats your poems
│   ├── train.py               # Fine-tunes GPT-2 on your words
│   └── generate.py            # Lets your AI poet write
│
├── utils/
│   └── config.py              # Settings for model + training
│
├── app/
│   ├── main.py                # Flask web app (optional)
│   └── templates/             # For HTML if you go web-based
│
├── requirements.txt           # All dependencies
├── .gitignore                 # Keeps your repo clean
└── README.md                  # You’re reading it 😎
```

---

## ⚙️ Setup & Run

### 1️⃣ Clone or download the repo
```bash
git clone https://github.com/VanshRajput-dev/VanshVerse.git
cd VanshVerse
```

### 2️⃣ Create your environment
```bash
conda create -n vanshverse python=3.10
conda activate vanshverse
pip install -r requirements.txt
```

### 3️⃣ Train your AI poet
```bash
python src/preprocess.py
python src/train.py
```

### 4️⃣ Make it write
```bash
python src/generate.py
```

---

## 💬 Example Output

```
🖋️ Generated Poem:

love in silence feels like a secret wind
it touches everything yet says nothing
every pause becomes a heartbeat
every word a wound that refuses to heal

✨ End of Poem ✨
```

---

## 🧠 Future Ideas

- Add a Flask web app so anyone can “Ask VanshVerse to write.”  
- Add multilingual support for better Hindi-English blend.  
- Release a small version on Hugging Face Hub.  

---

## ❤️ Credits

Project by **Vansh C. Rajput**  
> SRM University Kattankulathur, 2023–2027  
> GitHub: [VanshRajput-dev](https://github.com/VanshRajput-dev)

