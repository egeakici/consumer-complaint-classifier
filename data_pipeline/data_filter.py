import os
import pandas as pd
from tqdm import tqdm

INPUT_CSV  = os.path.join("data", "raw", "complaints.csv")
OUTPUT_CSV = os.path.join("data", "filtered", "dataset_filtered_balanced.csv")

CHUNK_SIZE = 20_000
MAX_ROWS   = 150_000

COLUMNS = [
    "Consumer complaint narrative",
    "Consumer disputed?",
    "Date received",
]

file_size_mb    = os.path.getsize(INPUT_CSV) / (1024 * 1024)
estimated_chunks = int((file_size_mb / 7800) * 500_000 / CHUNK_SIZE) + 1

filtered_chunks = []
total_read      = 0
total_saved     = 0

with tqdm(
    total=estimated_chunks,
    desc="Scanning file",
    unit="chunk",
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} chunks  |  Saved: {postfix}"
) as pbar:

    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False
    ):
        total_read += len(chunk)

        chunk["Date received"] = pd.to_datetime(chunk["Date received"], errors="coerce")

        mask = (
            (chunk["Date received"].dt.year >= 2012) &
            (chunk["Date received"].dt.year <= 2017) &
            (chunk["Consumer disputed?"].isin(["Yes", "No"])) &
            (chunk["Consumer complaint narrative"].notna())
        )

        clean = chunk[mask].copy()
        clean.drop(columns=["Date received"], inplace=True)

        if len(clean) > 0:
            remaining = MAX_ROWS - total_saved
            if len(clean) > remaining:
                clean = clean.iloc[:remaining]
            filtered_chunks.append(clean)
            total_saved += len(clean)

        pbar.set_postfix_str(f"{total_saved:,} / {MAX_ROWS:,} rows")
        pbar.update(1)

        if total_saved >= MAX_ROWS:
            tqdm.write(f"\nReached {MAX_ROWS:,} row target. Stopping scan.")
            break

raw = pd.concat(filtered_chunks, ignore_index=True)

print(f"\n{'─'*50}")
print(f"Total scanned : {total_read:,} rows")
print(f"Raw data      : {len(raw):,} rows")
print(f"\nRaw Yes/No distribution:")
print(raw["Consumer disputed?"].value_counts().to_string())

yes_df = raw[raw["Consumer disputed?"] == "Yes"]
no_df  = raw[raw["Consumer disputed?"] == "No"]

yes_count = len(yes_df)
no_count  = len(no_df)

print(f"\nYes: {yes_count:,} | No: {no_count:,}")

if no_count > yes_count:
    no_df = no_df.sample(n=yes_count, random_state=42)
    print(f"Undersampling: No reduced to {yes_count:,} rows.")
else:
    yes_df = yes_df.sample(n=no_count, random_state=42)
    print(f"Undersampling: Yes reduced to {no_count:,} rows.")

result = pd.concat([yes_df, no_df], ignore_index=True)
result = result.sample(frac=1, random_state=42).reset_index(drop=True)
result.insert(0, "id", range(1, len(result) + 1))

print(f"\n{'─'*50}")
print(f"Balanced data : {len(result):,} rows")
print(f"Columns       : {result.columns.tolist()}")
print(f"\nFinal Yes/No distribution:")
print(result["Consumer disputed?"].value_counts().to_string())
print(f"Ratio         : 50% Yes — 50% No")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\nSaved: {OUTPUT_CSV}")
