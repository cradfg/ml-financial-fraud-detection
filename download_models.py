import os
import gdown

os.makedirs("models", exist_ok=True)

files = {
    "models/lgbm_tuned.pkl":       "1D2Eeb2JBt2_uALbS5TSaLkRO2hgWznJY",
    "models/encoders.pkl":         "1G3ds9iFSICjqhhplvL8V4SMC1m5H4Joe",
    "models/feature_names.json":   "19Yh28reQI6arDOJfvB2XwKUcs5D5yawk",
    "models/threshold.json":       "1XzZgde6XgFAYc4XThK1zDzHNwOYs7os6",
}

for path, file_id in files.items():
    print(f"Downloading {path}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    print(f"Done — {path}")

print("All model artifacts downloaded.")