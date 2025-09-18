# =======================
# train.py
# =======================
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import warnings, json, joblib
warnings.filterwarnings("ignore")

# =====================
# CONFIG
# =====================
DATA_PATH = "features.csv"
TARGET_COL = "result"
N_TRIALS = 40
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded features: {df.shape}")

# =====================
# CLEANING
# =====================
irrelevant_cols = ["date", "season", "home_team_api_id", "away_team_api_id"]
bad_cols = irrelevant_cols + df.select_dtypes(include=["object"]).columns.tolist()
bad_cols = list(set(bad_cols))
if bad_cols:
    print(f"⚠️ Dropping columns: {bad_cols}")
    df = df.drop(columns=bad_cols, errors="ignore")

df = df.fillna(df.mean(numeric_only=True))
print(f"✅ After filling NaNs: {df.shape}")

# =====================
# TARGET
# =====================
if TARGET_COL not in df.columns:
    raise ValueError(f"❌ ERROR: No '{TARGET_COL}' column found!")

print(f"🔍 Unique values in '{TARGET_COL}':", df[TARGET_COL].unique())
df[TARGET_COL] = df[TARGET_COL].map({1: 1, 0: 0, 2: 0})
if df[TARGET_COL].isnull().any():
    raise ValueError("❌ ERROR: NaN values in target after mapping!")

print("🎯 Unique labels after remap:", df[TARGET_COL].unique())
df = df.dropna(subset=[TARGET_COL])
print(f"✅ After dropping NaNs in target: {df.shape}")

# =====================
# TRAIN/TEST SPLIT
# =====================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"Train rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}")
print("Class counts (train):", dict(Counter(y_train)))

# Class balancing
class_counts = Counter(y_train)
majority = max(class_counts.values())
sample_weights = y_train.map(lambda c: majority / class_counts[c]).astype(np.float32)

# =====================
# OPTUNA
# =====================
def objective(trial):
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),
        "seed": RANDOM_STATE,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    bst = lgb.train(
        param,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    preds = (bst.predict(X_test) > 0.5).astype(int)
    return f1_score(y_test, preds)

print(f"🚀 Starting Optuna tuning ({N_TRIALS} trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

print("✅ Best parameters:", study.best_trial.params)
print("✅ Best F1 score:", study.best_value)

# =====================
# FINAL MODEL
# =====================
best_params = study.best_trial.params
best_params.update({
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "bagging_freq": 1,
})

dtrain_final = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
bst_final = lgb.train(best_params, dtrain_final, num_boost_round=200)

preds = (bst_final.predict(X_test) > 0.5).astype(int)
print("\n📊 Final Model Report:")
print(classification_report(y_test, preds, digits=4))

# =====================
# SAVE ARTIFACTS
# =====================
bst_final.save_model("best_model.txt")
print("💾 Model saved as best_model.txt")

preprocess_info = {
    "dropped_cols": bad_cols,
    "feature_order": list(X.columns)
}
with open("preprocessing.json", "w") as f:
    json.dump(preprocess_info, f, indent=4)

print("💾 Preprocessing info saved as preprocessing.json")
