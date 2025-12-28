# --------------------------------------------------------------
# Automated Essay Scoring â€“ TFâ€‘IDF + LightGBM baseline
# --------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import warnings
from scipy import sparse

warnings.filterwarnings("ignore")

DATA_DIR = "./data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

# ---------- 1. Load data ----------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train_text = train_df["full_text"].astype(str).values
y_train = train_df["score"].astype(int).values
X_test_text = test_df["full_text"].astype(str).values
test_ids = test_df["essay_id"].values

# ---------- 2. TFâ€‘IDF feature extraction ----------
# combine word and character nâ€‘grams for richer representation
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=50000,
    stop_words="english",
    dtype=np.float32,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=20000,
    dtype=np.float32,
)

# Fit on training + test to keep vocab consistent
combined_corpus = np.concatenate([X_train_text, X_test_text])
word_vectorizer.fit(combined_corpus)
char_vectorizer.fit(combined_corpus)

X_train_word = word_vectorizer.transform(X_train_text)
X_train_char = char_vectorizer.transform(X_train_text)
X_train = sparse.hstack([X_train_word, X_train_char]).tocsr()

X_test_word = word_vectorizer.transform(X_test_text)
X_test_char = char_vectorizer.transform(X_test_text)
X_test = sparse.hstack([X_test_word, X_test_char]).tocsr()

# ---------- 3. 5â€‘fold stratified CV with LightGBM ----------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kappas = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Predict, round to integer scores 1â€‘6
    val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    val_pred_rounded = np.clip(np.rint(val_pred), 1, 6).astype(int)

    kappa = cohen_kappa_score(y_val, val_pred_rounded, weights="quadratic")
    kappas.append(kappa)
    print(f"Fold {fold}: Quadratic Weighted Kappa = {kappa:.5f}")

print(f"\nMean 5â€‘fold QWK: {np.mean(kappas):.5f}")


# ---------- 4. Train on full data ----------
full_train = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    full_train,
    num_boost_round=gbm.best_iteration,
)

# ---------- 5. Predict test set ----------
test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
test_pred_rounded = np.clip(np.rint(test_pred), 1, 6).astype(int)

# ---------- 6. Save submission ----------
submission = pd.DataFrame({"essay_id": test_ids, "score": test_pred_rounded})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission saved to {SUBMISSION_PATH}")
