import pandas as pd
from pathlib import Path

print("🔥 Start reading file...")

base = Path(__file__).resolve().parent
print("📂 Current folder:", base)

data_path = base / "training-data" / "data.csv"
print("📄 Trying path:", data_path)

df = pd.read_csv(data_path)

print("✅ File loaded successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First 2 rows:")
print(df.head(2))