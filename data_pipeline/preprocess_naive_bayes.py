import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

nltk.download("stopwords", quiet=True)

INPUT_CSV  = os.path.join("data", "filtered", "dataset_filtered_balanced.csv")
OUTPUT_CSV = os.path.join("data", "processed", "dataset_naive_bayes.csv")

STOPWORDS = set(stopwords.words("english"))
STOPWORDS.add("xxxx")
STOPWORDS.add("xx")

stemmer = PorterStemmer()


def preprocess(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


df = pd.read_csv(INPUT_CSV)

print(f"Shape: {df.shape}")
print(f"\nBefore (first text, 200 chars):")
print(df["Consumer complaint narrative"].iloc[0][:200])
print("\nStemming started, this may take 10-15 minutes...")

tqdm.pandas(desc="Preprocessing")
df["Consumer complaint narrative"] = df["Consumer complaint narrative"].progress_apply(preprocess)

print(f"\nAfter (first text, 200 chars):")
print(df["Consumer complaint narrative"].iloc[0][:200])

before = len(df)
df = df[df["Consumer complaint narrative"].str.strip() != ""]
print(f"\nDropped empty rows: {before - len(df)}")
print(f"Final shape: {df.shape}")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\nSaved: {OUTPUT_CSV}")
