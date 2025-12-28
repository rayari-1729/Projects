#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiâ€‘modal LightGBM for the Insult detection competition.
Features:
  * DistilBERT CLS embeddings (frozen)
  * TFâ€‘IDF (1â€‘2â€‘grams) â†’ TruncatedSVD (100 dim)
  * Lexical cues (len, capsâ€‘ratio, !, ?)
  * Temporal cues (hour, weekday, month)
5â€‘fold stratified CV + early stopping.
"""

import os, gc, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
os.environ.pop("HF_HUB_OFFLINE", None)

DATA_DIR = "./data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = "./submission.csv"

SEED = 42
MAX_TFIDF_FEATURES = 20000
SVD_DIM = 100
BATCH_SIZE = 64
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------
# 1. Helper utilities
# ----------------------------------------------------------------------
def clean_comment(text):
    """Remove surrounding quotes, unâ€‘escape unicode and normalise spaces."""
    if pd.isna(text):
        return ""
    if isinstance(text, str) and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.replace("\\n", " ").replace("\\t", " ")
    try:
        text = bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        pass
    return text.strip()


def lexical_features(text):
    """Return simple lexical statistics."""
    length = len(text)
    caps = sum(1 for c in text if c.isupper())
    caps_ratio = caps / length if length > 0 else 0.0
    exclam = text.count("!")
    qmark = text.count("?")
    return np.array([length, caps_ratio, exclam, qmark], dtype=np.float32)


def temporal_features(date_str):
    """Parse YYYYMMDDhhmmssZ â†’ hour, weekday (0â€‘Mon), month."""
    if pd.isna(date_str) or len(date_str) < 14:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    try:
        dt = pd.to_datetime(date_str[:14], format="%Y%m%d%H%M%S")
        hour = dt.hour
        weekday = dt.dayofweek
        month = dt.month
        return np.array([hour, weekday, month], dtype=np.float32)
    except Exception:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)


def batch_bert_embeddings(texts, tokenizer, model, batch_size=64):
    """Return CLS embeddings for a list of texts."""
    model.eval()
    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT embed", leave=False):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding="longest",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(DEVICE)
            out = model(**enc, output_hidden_states=False, return_dict=True)
            cls = out.last_hidden_state[:, 0, :]  # (B, hidden)
            all_emb.append(cls.cpu().numpy())
    return np.vstack(all_emb)


# ----------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["clean"] = train_df["Comment"].apply(clean_comment)
test_df["clean"] = test_df["Comment"].apply(clean_comment)

y = train_df["Insult"].values
X_text = train_df["clean"].tolist()
X_test_text = test_df["clean"].tolist()

# ----------------------------------------------------------------------
# 3. TFâ€‘IDF â†’ SVD
# ----------------------------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=MAX_TFIDF_FEATURES,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words="english",
)
tfidf.fit(X_text)  # fit on train only
X_train_tfidf = tfidf.transform(X_text)
X_test_tfidf = tfidf.transform(X_test_text)

svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

# ----------------------------------------------------------------------
# 4. Lexical + temporal features
# ----------------------------------------------------------------------
lex_train = np.vstack([lexical_features(t) for t in X_text])
lex_test = np.vstack([lexical_features(t) for t in X_test_text])

temp_train = np.vstack([temporal_features(d) for d in train_df["Date"]])
temp_test = np.vstack([temporal_features(d) for d in test_df["Date"]])

# Fill NaNs (possible for malformed dates)
temp_train = np.nan_to_num(temp_train, nan=-1)
temp_test = np.nan_to_num(temp_test, nan=-1)

# ----------------------------------------------------------------------
# 5. BERT CLS embeddings (frozen)
# ----------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)

emb_train = batch_bert_embeddings(X_text, tokenizer, bert_model, batch_size=BATCH_SIZE)
emb_test = batch_bert_embeddings(
    X_test_text, tokenizer, bert_model, batch_size=BATCH_SIZE
)

# Optional: reduce embedding dimension to keep LightGBM fast
svd_bert = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
emb_train_red = svd_bert.fit_transform(emb_train)
emb_test_red = svd_bert.transform(emb_test)

# ----------------------------------------------------------------------
# 6. Assemble final feature matrices
# ----------------------------------------------------------------------
X_train_full = np.hstack([X_train_svd, lex_train, temp_train, emb_train_red]).astype(
    np.float32
)
X_test_full = np.hstack([X_test_svd, lex_test, temp_test, emb_test_red]).astype(
    np.float32
)

# Scale features (LightGBM works fine without, but scaling helps PCAâ€‘reduced parts)
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)

# ----------------------------------------------------------------------
# 7. LightGBM 5â€‘fold CV
# ----------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": SEED,
    # "device": "gpu" if torch.cuda.is_available() else "cpu",
}

print("Starting 5â€‘fold LightGBM training...")

best_iterations = []
best_iterations.append(model.best_iteration)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y), 1):
    X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iterations.append(model.best_iteration)

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test_full, num_iteration=model.best_iteration) / skf.n_splits
    )

    # clean up
    del model, dtrain, dval, X_tr, X_val
    gc.collect()

cv_auc = roc_auc_score(y, oof_preds)
print(f"5â€‘fold CV AUC: {cv_auc:.5f}")

final_num_boost_round = int(np.mean(best_iterations))

# ----------------------------------------------------------------------
# 8. Train final model on full data and predict test (already averaged)
# ----------------------------------------------------------------------
final_model = lgb.train(
    lgb_params,
    lgb.Dataset(X_train_full, label=y),
    num_boost_round=final_num_boost_round,  # reuse best iteration
)

final_test_pred = test_preds  # already averaged over folds

# ----------------------------------------------------------------------
# 9. Save submission
# ----------------------------------------------------------------------
submission = pd.DataFrame({"Comment": test_df["Comment"], "Insult": final_test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
