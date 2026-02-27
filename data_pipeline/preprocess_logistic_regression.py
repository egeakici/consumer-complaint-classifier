import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from tqdm import tqdm

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

INPUT_CSV  = os.path.join("data", "filtered", "dataset_filtered_balanced.csv")
OUTPUT_CSV = os.path.join("data", "processed", "dataset_logistic_regression.csv")

STOPWORDS = set(stopwords.words("english"))
STOPWORDS.add("xxxx")
STOPWORDS.add("xx")

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(tag))
        for token, tag in pos_tags
        if token not in STOPWORDS and len(token) > 2
    ]
    return " ".join(tokens)


df = pd.read_csv(INPUT_CSV)

print(f"Shape: {df.shape}")
print(f"\nBefore (first text, 200 chars):")
print(df["Consumer complaint narrative"].iloc[0][:200])
print("\nLemmatization started, this may take 10-15 minutes...")

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
