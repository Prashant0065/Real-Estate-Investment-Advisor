
import pandas as pd
from pathlib import Path
raw_path = Path("data/raw/india_housing_prices.csv")
processed_path = Path("data/processed/india_housing_prices_preprocessed.csv")
df = pd.read_csv(raw_path)
print("Original dataset shape:", df.shape)
print("\nMissing values before preprocessing:")
print(df.isnull().sum().sum())
print("\nDuplicate rows before preprocessing:")
print(df.duplicated().sum())
df["Calculated_Price_per_SqFt"] = (
    df["Price_in_Lakhs"] / df["Size_in_SqFt"]
).round(2)
df["Price_per_SqFt_Mismatch"] = (
    df["Price_per_SqFt"] != df["Calculated_Price_per_SqFt"]
)
print("\nPrice-per-sq-ft mismatches:")
print(df["Price_per_SqFt_Mismatch"].sum())
print("\nZero stored price-per-sq-ft values:")
print((df["Price_per_SqFt"] == 0).sum())
processed_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_path, index=False)
print("\nPreprocessed dataset saved to:", processed_path)
print("Final dataset shape:", df.shape)