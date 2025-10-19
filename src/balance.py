import os
import sys

# Fix imports for project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TORCHDYNAMO_DISABLE"] = "1"
import pandas as pd
from sklearn.utils import resample

df = pd.read_csv("poems.csv")  # or however your dataset is stored

# Example labeling logic
def detect_lang(text):
    if any(ord(c) > 256 for c in text):  # Unicode Hindi chars
        return "hindi"
    elif any(word in text.lower() for word in ["hai", "nahi", "tu", "tera", "mujhe", "kya", "vo"]):
        return "hinglish"
    else:
        return "english"

df["lang"] = df["text"].apply(detect_lang)

# Balance all categories
df_balanced = (
    df.groupby("lang", group_keys=False)
    .apply(lambda x: resample(x, replace=True, n_samples=df['lang'].value_counts().max(), random_state=42))
)

df_balanced.to_csv("poems_balanced.csv", index=False)
print(df_balanced['lang'].value_counts())
