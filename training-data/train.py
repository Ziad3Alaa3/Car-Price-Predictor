import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
COLUMNS_PATH = ARTIFACTS_DIR / "columns.json"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def ensure_brand(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "brand" not in df.columns and "name" in df.columns:
        df["brand"] = df["name"].astype(str).str.strip().str.split().str[0]
    return df


def pick_target_column(df: pd.DataFrame) -> str:
    for c in ["selling_price", "price", "target"]:
        if c in df.columns:
            return c
    raise ValueError("No target column found in the dataset")


df = pd.read_csv(DATA_PATH)
df = normalize_columns(df)
df = ensure_brand(df)

target_column = pick_target_column(df)

possible_features = ["brand", "year", "km_driven", "fuel", "transmission", "owner"]
feature_cols = [c for c in possible_features if c in df.columns]

if len(feature_cols) < 4:
    raise ValueError(f"Not enough features found. Found only: {feature_cols}")

df = df.dropna(subset=feature_cols + [target_column])

X = df[feature_cols].copy()
y = df[target_column].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numerical_features = [c for c in feature_cols if c in ["year", "km_driven"]]
categorical_features = [c for c in feature_cols if c not in numerical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ],
    remainder="drop",
)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model),
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("===== Evaluation results =====")
print(f"MAE: {mae:.2f}")
print(f"R2 : {r2:.2f}")

joblib.dump(pipeline, MODEL_PATH)

dropdowns = {col: sorted(df[col].astype(str).dropna().unique().tolist()) for col in categorical_features}

columns_payload = {
    "numerical": numerical_features,
    "categorical": categorical_features,
    "dropdowns": dropdowns,
    "features": feature_cols,
    "target": target_column,
    "model_version": "1.0.0",
}

COLUMNS_PATH.write_text(json.dumps(columns_payload, ensure_ascii=False, indent=4), encoding="utf-8")

print("✅ Saved successfully!")
print(f"Model   -> {MODEL_PATH}")
print(f"Columns -> {COLUMNS_PATH}")