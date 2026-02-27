# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib
import json
import pandas as pd

from .schemas import CarIn, Car_Price

app = FastAPI(title="Car Price API", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
MODEL_PATH = ART / "model.pkl"
META_PATH = ART / "columns.json"

pipe = None
meta = None


@app.on_event("startup")
def load_mode():
    global pipe, meta
    print("Loading model and metadata...")
    print("MODEL_PATH =", MODEL_PATH)
    print("META_PATH  =", META_PATH)

    if not MODEL_PATH.exists() or not META_PATH.exists():
        print("❌ Model or metadata not found in backend/artifacts/")
        print("   Put model.pkl and columns.json here:")
        print(f"   {ART}")
        return

    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    print("✅ Model and metadata loaded successfully.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def get_meta():
    if meta is None:
        raise HTTPException(status_code=503, detail="Meta not loaded")
    return meta


@app.post("/echo")
def echo(car: CarIn):
    return {"received_data": car}


@app.post("/car_price", response_model=Car_Price)
def car_price(payload: CarIn):
    if pipe is None or meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        row = pd.DataFrame([payload.model_dump()])

        # ✅ ضمان ترتيب الأعمدة زي اللي الموديل اتدرب عليه
        row = row[meta["features"]]

        pred = float(pipe.predict(row)[0])
        pred = max(pred, 0.0)

        return Car_Price(
            price=pred,
            model_version=meta.get("model_version", "1.0.0"),
        )

    except Exception as e:
        # نطلع السبب الحقيقي بدل 500 مبهم
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")