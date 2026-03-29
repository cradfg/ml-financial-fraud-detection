# ml-financial-fraud-detection

I built an end to end ml model that works on LightGBM model and it is trained on the IEEE-CIS dataset that has approximately 600k+ transactions . I used fastapi for backend and bridging the model with frontend , i deployed the frontend with github pages using CI/CD actions and the backend with render.

Below are the links of:
**Live Demo:** [cradfg.github.io/ml-financial-fraud-detection](https://cradfg.github.io/ml-financial-fraud-detection)  
**API:** [model-financial-crime-detection.onrender.com](https://model-financial-crime-detection.onrender.com/docs)

> First request may take ~20s because i am using Render's free trail.
---

## Results

| Metric | Value |
|---|---|
| PR-AUC (OOF) | **0.8057** |
| F1 Score | **0.7563** |
| Decision Threshold | 0.6637|
| Cross Validation | 5-fold Stratified KFold |
| Training Rows | 590,540 |
| Fraud Rate | 3.50% (20,663 cases) |

> the threshold is based on the best fold out of 5 of the model , also users can change the threshold to their discretion in ui.

> PR-AUC of 0.8057 against a 3.5% fraud rate baseline of 0.035 — 23x above random chance.

---

## Demo

<img width="1389" height="881" alt="image" src="https://github.com/user-attachments/assets/73111b9d-191b-4891-a93e-887213373ab3" />


Enter a transaction amount, card network, card type, email domain, product code, and device type. The model returns a fraud probability score, risk level, and APPROVED / BLOCKED verdict in real time. The decision threshold is adjustable post-prediction — lower it to catch more fraud at the cost of false positives, raise it to only flag high-confidence cases.

---

## Project Structure

```
ml-financial-fraud-detection/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS middleware, /predict route
│   ├── model.py             # artifact loading, preprocessing, inference
│   ├── schemas.py           # Pydantic input/output validation models
│   └── requirements.txt     # backend-only dependencies
├── index.html               # dashboard UI
├── style.css                # cybersecurity theme
├── script.js                # fetch calls, threshold re-evaluation
├── train.py                 # full training pipeline (run once locally)
├── download_models.py       # pulls model artifacts from Google Drive at build
├── Dockerfile               # container spec for Render deployment
├── requirements.txt         # training dependencies
└── README.md
```

---

## Dataset

**IEEE-CIS Fraud Detection** — provided by Vesta Corporation via Kaggle.

- 590,540 transactions split across two tables (`train_transaction.csv` + `train_identity.csv`)
- Merged on `TransactionID` — identity table covers ~24% of transactions
- 434 raw features reduced to ~130 via missing value filtering and column selection
- Class imbalance: 3.5% fraud, 96.5% legitimate (this is one of the biggest drawbacks of using this dataset).

Download: [kaggle.com/competitions/ieee-fraud-detection/data](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

Place files in `data/raw/` before running `train.py`:
```
data/raw/
  train_transaction.csv
  train_identity.csv
```

> `data/` is gitignored. Model artifacts in `models/` are also gitignored and downloaded at deploy time via `download_models.py`.

---

## Model

**LightGBM** gradient boosted decision tree — chosen for its native NaN handling, speed on tabular data, and compatibility with SHAP explainability.
Below is the hyper tuning parameters of my model.
```python
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
}
```

**Key decisions:**
- Columns with >80% missing values dropped before training
- Remaining NaNs passed through natively — no imputation
- `scale_pos_weight=10` handles class imbalance in the loss function
- Decision threshold tuned post-training on OOF predictions to maximise F1
- 5-fold stratified CV preserves fraud ratio across all folds

**Training artifacts saved to `models/`:**
```
lgbm_tuned.pkl         trained model
encoders.pkl           label encoders for categorical columns
feature_names.json     exact feature list and order for inference alignment
threshold.json         optimal decision threshold + OOF metrics
```

---

## API Reference

**Base URL:** `https://model-financial-crime-detection.onrender.com`

### `GET /health`
Returns server status.
```json
{ "status": "healthy" }
```

### `POST /predict`

**Request body:**
```json
{
  "TransactionAmt": 150.0,
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com",
  "ProductCD": "W",
  "DeviceType": "desktop"
}
```

**Response:**
```json
{
  "fraud_probability": 0.0423,
  "is_fraud": false,
  "risk_level": "LOW",
  "threshold_used": 0.6637
}
```

All fields except `TransactionAmt` are optional — missing fields default to `NaN` and are handled natively by the model. Full interactive docs at `/docs`.

---

## Running Locally

```bash
# 1. clone
git clone https://github.com/cradfg/ml-financial-fraud-detection.git
cd ml-financial-fraud-detection

# 2. install training dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. download dataset → data/raw/ then train
python train.py (i would recomend to run this file in colab then import the results in your IDE because for me the IDE always crashed since dataset was large.)
# artifacts saved to models/

# 4. run backend
cd app
python main.py
# API at http://localhost:8000
# Docs at http://localhost:8000/docs

# 5. open frontend with VS Code Live Server
# visit http://127.0.0.1:5500
```

---

## Deployment

| Layer | Platform | Config |
|---|---|---|
| Frontend | GitHub Pages | served from repo root |
| Backend | Render (free tier) | Docker container |
| Model artifacts | Google Drive → Render | via `download_models.py` at build |

Model artifacts are not stored in the repository. At build time, Render runs `download_models.py` which pulls the four artifact files from Google Drive before starting the server.

---

## Limitations

- Trained on a single Kaggle competition dataset — not validated on real production transaction streams
- Feature set limited to ~130 of the original 434 columns due to hardware memory constraints during training
- Model is static — does not retrain or adapt to new fraud patterns over time
- SHAP explainability not yet integrated into the dashboard
- Render free tier introduces ~20s cold start after 15 minutes of inactivity
- Would be a better model with ensemble techniques which i didn't introduce yet.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Model | LightGBM, scikit-learn, imbalanced-learn |
| Data | pandas, numpy, pyarrow |
| Backend | FastAPI, Uvicorn, Pydantic, joblib |
| Frontend | HTML, CSS, Vanilla JS |
| Deployment | Docker, Render, GitHub Pages |
| Training | Google Colab (12GB RAM) |

---

## Author

**cradfg** — [github.com/cradfg](https://github.com/cradfg)
