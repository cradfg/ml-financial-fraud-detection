import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"

#Paths of the dataset and models

RAW_DIR    = Path("/home/aravind/repos/ml-financial-fraud-detection/data/raw")
MODELS_DIR = Path("/home/aravind/repos/ml-financial-fraud-detection/models")
MODELS_DIR.mkdir(exist_ok=True)

#1. We are loading the IEEE-CIS dataset here and are also optimizing dataset

def load_data():
    print("[1/5] Loading data...")

    usecols_txn = [
        "TransactionID", "isFraud", "TransactionAmt", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "dist2",
        "P_emaildomain", "R_emaildomain",
        "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
        "D1","D2","D3","D4","D5","D10","D15",
        "M1","M2","M3","M4","M5","M6","M7","M8","M9",
        "V1","V2","V3","V4","V12","V13","V14",
        "V20","V21","V22","V23","V24","V25","V26",
        "V35","V36","V37","V38","V40",
        "V44","V45","V46","V47","V48","V49","V50","V51",
        "V54","V55","V56","V57","V58","V59","V60","V61","V62","V63","V64","V65","V66","V67",
        "V70","V71","V72","V73","V74","V75","V76","V78","V79","V80","V81","V82","V83",
        "V91","V92","V93","V94","V95","V96","V97","V98","V99","V100",
    ]

    usecols_id = [
        "TransactionID",
        "id_01","id_02","id_03","id_04","id_05","id_06",
        "id_09","id_10","id_11","id_12","id_13","id_15",
        "id_17","id_19","id_20","id_28","id_29","id_31",
        "id_35","id_36","id_37","id_38",
        "DeviceType", "DeviceInfo",
    ]

    dtype_txn = {
        "TransactionAmt": "float32",
        "card1": "float32", "card2": "float32",
        "card3": "float32", "card5": "float32",
        "addr1": "float32", "addr2": "float32",
        "dist1": "float32", "dist2": "float32",
        **{f"C{i}": "float32" for i in range(1, 15)},
        **{f"D{i}": "float32" for i in [1,2,3,4,5,10,15]},
        **{f"V{i}": "float32" for i in [
            1,2,3,4,12,13,14,20,21,22,23,24,25,26,
            35,36,37,38,40,44,45,46,47,48,49,50,51,
            54,55,56,57,58,59,60,61,62,63,64,65,66,67,
            70,71,72,73,74,75,76,78,79,80,81,82,83,
            91,92,93,94,95,96,97,98,99,100,
        ]},
        "isFraud": "int8",
    }

    dtype_id = {
    "id_01": "float32",
    "id_02": "float32",
    "id_03": "float32",
    "id_04": "float32",
    "id_05": "float32",
    "id_06": "float32",
    }

    train_txn = pd.read_csv(
        RAW_DIR / "train_transaction.csv",
        usecols=usecols_txn,
        dtype=dtype_txn,
    )

    train_id = pd.read_csv(
        RAW_DIR / "train_identity.csv",
        usecols=usecols_id,
        dtype=dtype_id,
    )

    df = train_txn.merge(train_id, on="TransactionID", how="left")
    del train_txn, train_id

    print(f"      Merged shape: {df.shape}")
    print(f"      Fraud rate:   {df['isFraud'].mean():.4f} ({df['isFraud'].sum()} fraud cases)")
    return df

#2. Preprocessing of the IEEE-CIS Dataset.

def preprocess(df):
    print("[2/5] Preprocessing...")

    # drop columns with >80% missing
    missing_rate = df.isnull().mean()
    drop_cols = [
        c for c in missing_rate[missing_rate > 0.8].index.tolist()
        if c != "isFraud"
    ]
    df = df.drop(columns=drop_cols)
    print(f"      Dropped {len(drop_cols)} columns with >80% missing values")

    # drop identifiers
    df = df.drop(columns=[c for c in ["TransactionID", "TransactionDT"] if c in df.columns])

    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])

    # label encode categoricals
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        non_null = X[col].dropna()
        le.fit(non_null.astype(str))
        X[col] = X[col].apply(
            lambda v: le.transform([str(v)])[0] if pd.notnull(v) else -1
        )
        encoders[col] = le

    print(f"      Label-encoded {len(cat_cols)} categorical columns")
    print(f"      Final feature count: {X.shape[1]}")
    return X, y, encoders

#3. Feature Engineering

def feature_engineering(X):
    print("[3/5] Feature engineering...")

    if "TransactionAmt" in X.columns:
        X["TransactionAmt_log"] = np.log1p(X["TransactionAmt"]).astype("float32")

    X["null_count"] = X.isnull().sum(axis=1).astype("float32")

    print(f"      Feature count after engineering: {X.shape[1]}")
    return X

#4. Training the model

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",
    "boosting_type":     "gbdt",
    "num_leaves":        64,
    "learning_rate":     0.05,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "scale_pos_weight":  10,
    "n_estimators":      1000,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

def train(X, y):
    print("[4/5] Training LightGBM with 5-fold cross validation...")

    skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds   = np.zeros(len(y))
    fold_scores = []
    models      = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        score = average_precision_score(y_val, val_preds)
        fold_scores.append(score)
        models.append(model)
        print(f"      Fold {fold} PR-AUC: {score:.4f}")

    print(f"\n      OOF PR-AUC:       {average_precision_score(y, oof_preds):.4f}")
    print(f"      Mean fold PR-AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    best_idx   = int(np.argmax(fold_scores))
    best_model = models[best_idx]
    print(f"      Best model: fold {best_idx + 1}")

    return best_model, oof_preds, X.columns.tolist()

#5. Threshold Tuning + Save

def tune_threshold_and_save(model, X, y, oof_preds, feature_names, encoders):
    print("[5/5] Tuning threshold and saving artifacts...")

    precisions, recalls, thresholds = precision_recall_curve(y, oof_preds)
    f1_scores      = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx       = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1        = float(f1_scores[best_idx])

    print(f"      Best threshold: {best_threshold:.4f}  (F1 = {best_f1:.4f})")

    final_preds = (oof_preds >= best_threshold).astype(int)
    final_f1    = f1_score(y, final_preds)
    print(f"      Final OOF F1 at threshold: {final_f1:.4f}")

    joblib.dump(model, MODELS_DIR / "lgbm_tuned.pkl")
    print(f"      Saved → models/lgbm_tuned.pkl")

    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({
            "threshold":  best_threshold,
            "oof_f1":     round(final_f1, 4),
            "oof_pr_auc": round(average_precision_score(y, oof_preds), 4),
        }, f, indent=2)
    print(f"      Saved → models/threshold.json")

    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"      Saved → models/feature_names.json")

    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    print(f"      Saved → models/encoders.pkl")

#Main

if __name__ == "__main__":
    df                           = load_data()
    X, y, encoders               = preprocess(df)
    del df
    X                            = feature_engineering(X)
    model, oof_preds, feat_names = train(X, y)
    tune_threshold_and_save(model, X, y, oof_preds, feat_names, encoders)
    print("\nDone. All artifacts saved to models/")
    print("Next: cd app && python main.py")