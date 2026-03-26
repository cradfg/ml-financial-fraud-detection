import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_artifacts():
    model      = joblib.load(MODELS_DIR / "lgbm_tuned.pkl")
    encoders   = joblib.load(MODELS_DIR / "encoders.pkl")
    threshold  = json.loads((MODELS_DIR / "threshold.json").read_text())["threshold"]
    feat_names = json.loads((MODELS_DIR / "feature_names.json").read_text())
    return model, encoders, threshold, feat_names


# load once at startup — stays in memory for all requests
model, encoders, THRESHOLD, FEATURE_NAMES = load_artifacts()

CAT_COLS = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
    "M4", "M6",
]

CAT_COLS_SET = set(CAT_COLS)


def preprocess_input(data: dict) -> pd.DataFrame:
    # start with a NaN row for every feature the model was trained on
    row = {col: np.nan for col in FEATURE_NAMES}

    # fill in whatever the user provided
    for key, val in data.items():
        if key in row:
            row[key] = val

    df = pd.DataFrame([row])

    # engineered features — must mirror train.py exactly
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(
            pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0)
        ).astype("float32")

    df["null_count"] = df.isnull().sum(axis=1).astype("float32")

    # label encode categoricals using saved encoders
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(
                lambda v: le.transform([str(v)])[0]
                if pd.notnull(v) and str(v) in le.classes_
                else -1
            )
        else:
            df[col] = -1

    # align to exact trained feature order
    df = df.reindex(columns=FEATURE_NAMES, fill_value=np.nan)

    # force all non-categorical columns to float32
    # lgbm cannot handle object dtype — this fixes missing/unfilled fields
    for col in df.columns:
        if col not in CAT_COLS_SET:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df


def predict(data: dict) -> dict:
    df   = preprocess_input(data)
    prob = float(model.predict_proba(df)[0][1])
    fraud = prob >= THRESHOLD

    if prob < 0.3:
        risk = "LOW"
    elif prob < THRESHOLD:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud":          fraud,
        "risk_level":        risk,
        "threshold_used":    round(THRESHOLD, 4),
    }