"""
train_model.py
--------------
Train a linear regression model on mtcars.csv using:
  - wt  (vehicle weight in 1000 lbs)
  - hp  (gross horsepower)

as predictors of:
  - mpg (miles per gallon)

The fitted model is saved to models/model.pkl.
Run from the repo root:
    python scripts/train_model.py
"""

import math
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

# ── Try to import sklearn (preferred); fall back to built-in OLS ────────────
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    USE_SKLEARN = True
except ImportError:
    USE_SKLEARN = False
    # Make app.model_utils importable from repo root
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    from app.model_utils import LinearRegressionModel

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "mtcars.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")
print(df[["mpg", "wt", "hp"]].describe().to_string())

# ── Features & target ─────────────────────────────────────────────────────
FEATURES = ["wt", "hp"]
TARGET = "mpg"

X = df[FEATURES].values
y = df[TARGET].values

# ── Train model ────────────────────────────────────────────────────────────
if USE_SKLEARN:
    print("\nUsing scikit-learn LinearRegression")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    coef = model.coef_
    intercept = model.intercept_
else:
    print("\nUsing built-in OLS (scikit-learn not installed)")
    X_b = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_b, y, rcond=None)[0]
    intercept = float(beta[0])
    coef = beta[1:]

    preds = X_b @ beta
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = math.sqrt(ss_res / len(y))

    model = LinearRegressionModel(coef_=coef, intercept_=intercept)

# ── Report ─────────────────────────────────────────────────────────────────
print(f"\nModel performance:")
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"\nCoefficients:")
for feat, c in zip(FEATURES, coef):
    print(f"  {feat}: {c:.4f}")
print(f"  intercept: {intercept:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to {MODEL_PATH}")

# ── Quick sanity check ──────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    loaded = pickle.load(f)
test_pred = loaded.predict([[2.62, 110]])[0]
print(f"Sanity check – wt=2.62, hp=110 → predicted mpg={test_pred:.2f}")
