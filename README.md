# Consumer Complaint Classifier

Binary classification on the CFPB Consumer Complaint Database.  
**Task:** Predict whether a consumer will dispute a company's response (`Yes` / `No`) based solely on the complaint narrative.

Both models — Complement Naive Bayes and Logistic Regression — are implemented from scratch without scikit-learn.

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Project Structure](#2-project-structure)
3. [Pipeline](#3-pipeline)
4. [Preprocessing](#4-preprocessing)
5. [Feature Extraction](#5-feature-extraction)
6. [Models](#6-models)
7. [Results](#7-results)
8. [API](#8-api)
9. [Setup](#9-setup)

---

## 1. Dataset

**Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

The raw dataset is approximately 8 GB and contains ~18 columns. This project uses two:

| Column | Role |
|---|---|
| `Consumer complaint narrative` | Input feature (X) |
| `Consumer disputed?` | Target label (y) |

The `Consumer disputed?` field is only populated for complaints filed between **2012–2017**. Outside this window the column is empty and those rows are discarded.

**Class imbalance:** The raw data is heavily skewed — roughly 78% `No` and 22% `Yes`. Random undersampling is applied to produce a balanced 50/50 split before any modeling.

**Final dataset:** 65,470 rows (32,735 per class).

---

## 2. Project Structure

```
consumer-complaint-classifier/
├── pipeline_naive_bayes.py          # End-to-end runner for Naive Bayes
├── pipeline_logistic_regression.py  # End-to-end runner for Logistic Regression
│
├── data_pipeline/
│   ├── data_filter.py               # Chunk-based filtering + undersampling
│   ├── preprocess_naive_bayes.py    # Stemming pipeline
│   └── preprocess_logistic_regression.py  # Lemmatization pipeline
│
├── naive_bayes/
│   ├── complement_nb.py             # Complement NB — from scratch
│   ├── tfidf.py                     # TF-IDF vectorizer — from scratch
│   ├── metrics.py                   # Accuracy, F1, confusion matrix — from scratch
│   ├── data_utils.py                # Stratified train/test split — from scratch
│   ├── model_utils.py               # build / train / evaluate / save
│   ├── preprocess_for_inference.py  # Inference-time preprocessing
│   ├── config.py                    # All hyperparameters and paths
│   ├── train.py                     # Training entry point
│   ├── test.py                      # Evaluation entry point
│   └── app.py                       # Flask + Swagger UI (port 5000)
│
├── logistic_regression/
│   ├── logistic_regression.py       # Logistic Regression (Mini-batch SGD) — from scratch
│   ├── tfidf.py                     # TF-IDF vectorizer — from scratch
│   ├── metrics.py                   # Accuracy, F1, confusion matrix — from scratch
│   ├── data_utils.py                # Stratified train/test split — from scratch
│   ├── model_utils.py               # build / train / evaluate / save
│   ├── preprocess_for_inference.py  # Inference-time preprocessing
│   ├── config.py                    # All hyperparameters and paths
│   ├── train.py                     # Training entry point
│   ├── test.py                      # Evaluation entry point
│   └── app.py                       # Flask + Swagger UI (port 5001)
│
└── data/
    ├── raw/complaints.csv           # Original CFPB file (not tracked by git)
    ├── filtered/                    # Output of data_filter.py
    └── processed/                   # Output of preprocessing steps
```

---

## 3. Pipeline

Each model has a dedicated end-to-end pipeline script that runs four sequential steps:

| Step | Script | Description |
|---|---|---|
| 1 | `data_pipeline/data_filter.py` | Chunk-read the 8 GB CSV, filter by date and non-null fields, undersample to 50/50 |
| 2 | `data_pipeline/preprocess_naive_bayes.py` or `preprocess_logistic_regression.py` | Clean and normalize complaint text |
| 3 | `<model>/train.py` | Vectorize with TF-IDF, train the model, serialize artifacts |
| 4 | `<model>/test.py` | Load serialized artifacts, run evaluation, print metrics |

**Run Naive Bayes end-to-end:**
```bash
python pipeline_naive_bayes.py
```

**Run Logistic Regression end-to-end:**
```bash
python pipeline_logistic_regression.py
```

**Skip data steps if processed data already exists:**
```bash
python pipeline_naive_bayes.py --start-from 3
python pipeline_logistic_regression.py --start-from 3
```

---

## 4. Preprocessing

Both pipelines share the same base cleaning steps. The normalization strategy diverges at the final step:

| Step | Both Models |
|---|---|
| Lowercasing | All text converted to lowercase |
| URL removal | `http://...` and `www....` tokens stripped |
| Punctuation & digit removal | Only `[a-z]` and whitespace retained |
| Whitespace normalization | Consecutive spaces collapsed |
| Stopword removal | NLTK English stopwords + `xxxx` / `xx` (anonymization tokens) |
| Length filter | Tokens with `len <= 2` discarded |

| Final Step | Naive Bayes | Logistic Regression |
|---|---|---|
| Normalization | **Stemming** (PorterStemmer) | **Lemmatization** (WordNetLemmatizer + POS tagging) |
| Rationale | NB operates on term counts and is insensitive to morphological detail | LR is sensitive to feature differences; preserving morphological form improves discrimination |

---

## 5. Feature Extraction

Both models use the same TF-IDF implementation, written from scratch.

**TF-IDF formula:**

```
TF(t, d)  = count(t in d) / total tokens in d
IDF(t)    = log((1 + N) / (1 + df(t))) + 1     # smooth IDF
TF-IDF    = TF × IDF
```

Each document vector is L2-normalized to remove length bias.

**Configuration:**

| Parameter | Naive Bayes | Logistic Regression |
|---|---|---|
| `ngram_range` | (1, 2) | (1, 2) |
| `min_df` | 5 | 10 |
| Vocabulary size | ~141,000 tokens | ~67,000 tokens |

Bigrams capture negation patterns such as `"not satisfied"` and `"no response"` that unigrams would miss.

> Vocabulary and IDF values are fitted on the training set only. The test set is transformed without refitting to prevent data leakage.

---

## 6. Models

### Complement Naive Bayes

Standard Naive Bayes accumulates positive evidence for each class. **Complement Naive Bayes (CNB)** instead measures evidence from all *other* classes and selects the class with the least complement evidence. This approach is more robust on short, imbalanced text.

**Training:**
1. Accumulate per-class and global TF-IDF sums across all training documents
2. Compute complement sum per class: `global_sum − class_sum`
3. Apply Laplace smoothing (`alpha = 1.0`) and compute log weights
4. Optionally L2-normalize the weight vectors

**Inference:** For each document, compute the complement score for every class; assign the class with the **lowest** score.

| Hyperparameter | Value |
|---|---|
| `alpha` | 1.0 |
| `norm` | True |
| `min_df` | 5 |
| `ngram_range` | (1, 2) |

---

### Logistic Regression

Custom implementation using **Mini-batch Stochastic Gradient Descent** with L2 regularization and learning rate decay.

**Objective function:**

```
L = BCE(y, ŷ) + (1 / 2C) * ||w||²
```

**Training loop (per epoch):**
1. Shuffle training indices
2. For each mini-batch: forward pass → compute BCE + L2 loss → compute gradients → update weights
3. Multiply learning rate by `lr_decay`

**Inference:** Compute `sigmoid(Xw + b)`; threshold at 0.5 to assign class label.

| Hyperparameter | Value |
|---|---|
| `learning_rate` | 0.1 |
| `max_iter` | 50 |
| `batch_size` | 512 |
| `C` | 100.0 |
| `lr_decay` | 0.95 |
| `min_df` | 10 |
| `ngram_range` | (1, 2) |

**Note on convergence:** Loss plateaued near `ln(2) ≈ 0.693` across all 50 epochs, indicating the model approached but did not meaningfully surpass random-chance loss. This is consistent with the weak causal signal in the data — see [Results](#7-results).

---

## 7. Results

All metrics computed from scratch (no scikit-learn).

| Metric | Complement NB | Logistic Regression |
|---|---|---|
| Accuracy | **0.5942** | 0.5864 |
| F1 (macro) | **0.5934** | 0.5832 |
| F1 (Yes) | **0.6109** | 0.5464 |
| F1 (No) | 0.5760 | **0.6200** |

**Complement NB — Confusion Matrix:**
```
               Pred Yes    Pred No
Actual Yes      4111        2332
Actual No       2905        3557
```

**Logistic Regression — Confusion Matrix:**
```
               Pred Yes    Pred No
Actual Yes      3214        3229
Actual No       2108        4354
```

**Interpretation:**

Both models perform only modestly above the 50% random baseline, which reflects a fundamental constraint of the problem: the target label (`Consumer disputed?`) is determined by the consumer's reaction to the *company's response*, whereas the input is the *complaint text* written before any response was received. The complaint narrative alone carries limited predictive signal for the dispute outcome.

Complement NB outperforms Logistic Regression on overall F1, likely because CNB is better suited to sparse, high-dimensional bag-of-words representations. LR shows stronger recall on the `No` class but struggles with `Yes`.

---

## 8. API

Both models are served as REST APIs via Flask and documented with Swagger UI.

| Model | Port | Swagger UI |
|---|---|---|
| Naive Bayes | 5000 | http://localhost:5000/apidocs |
| Logistic Regression | 5001 | http://localhost:5001/apidocs |

**Start Naive Bayes API:**
```bash
cd naive_bayes
python app.py
```

**Start Logistic Regression API:**
```bash
cd logistic_regression
python app.py
```

Both can run simultaneously on their respective ports.

**Endpoints:**

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Model status check |
| POST | `/predict` | Single complaint prediction |
| POST | `/predict/batch` | Batch prediction (max 100) |

**Example request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I contacted the bank multiple times but received no response."}'
```

**Example response:**
```json
{
  "prediction": "Yes",
  "clean_text": "contact bank multipl time receiv respons",
  "text_length": 63,
  "latency_ms": 3.14
}
```

---

## 9. Setup

**Requirements:**
```
flask
flasgger
pandas
numpy
scipy
nltk
tqdm
```

**Install:**
```bash
pip install flask flasgger pandas numpy scipy nltk tqdm
```

**Run a full pipeline:**
```bash
# Place complaints.csv in data/raw/
python pipeline_naive_bayes.py
python pipeline_logistic_regression.py
```