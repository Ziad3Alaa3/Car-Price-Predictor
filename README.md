# 🚗 Car Price Predictor

End-to-end Machine Learning web application that predicts used car selling prices.

🔗 Repo: https://github.com/Ziad3Alaa3/Car-Price-Predictor

---

## 🧠 Project Overview

This project demonstrates a complete ML workflow:

- Train a machine learning model
- Save the pipeline as an artifact
- Serve the model using FastAPI
- Build a frontend that consumes the API
- Dynamically load dropdown values from metadata

---

## ⚙️ Tech Stack

### 🔹 Machine Learning
- Scikit-learn
- Pandas
- RandomForestRegressor
- ColumnTransformer
- OneHotEncoder
- Joblib

### 🔹 Backend
- FastAPI
- Pydantic
- Uvicorn

### 🔹 Frontend
- HTML
- CSS
- Vanilla JavaScript (Fetch API)

---

## 🏗️ Project Structure



---


**Model:** RandomForestRegressor

### Features used:

- brand
- year
- km_driven
- fuel
- transmission
- owner

### Target:

- selling_price

### Pipeline:

- Numeric → passthrough
- Categorical → OneHotEncoder
- Model → RandomForestRegressor

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| MAE    | _Add your value_ |
| R²     | _Add your value_ |

---

## 🚀 Run Locally

### 1️⃣ Clone the repo

```bash
git clone https://github.com/Ziad3Alaa3/Car-Price-Predictor
cd Car-Price-Predictor

2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model
python training/train.py

This will generate:

model.pkl

columns.json

5️⃣ Run FastAPI
uvicorn backend.main:app --reload

Swagger:

http://127.0.0.1:8000/docs
6️⃣ Run Frontend
cd frontend
python -m http.server 5500

Open:

http://127.0.0.1:5500
🔌 API Endpoints
Health Check
GET /health
Metadata (dropdown values)
GET /meta
Predict Car Price
POST /car_price
Request Example:
{
  "brand": "Toyota",
  "year": 2015,
  "km_driven": 60000,
  "fuel": "Petrol",
  "transmission": "Manual",
  "owner": "First Owner"
}
🎨 UI Preview

Add screenshot here

assets/ui.png
🌟 Key Concepts Demonstrated

✔ End-to-end ML workflow
✔ Pipeline-based preprocessing
✔ Model artifact loading
✔ REST API for ML model
✔ Dynamic frontend from backend metadata
✔ CORS handling

🔮 Future Improvements

Deploy FastAPI

Dockerize the project

Add authentication

Model monitoring

Deep Learning version for image input

👨‍💻 Author
Ziad Alaa








