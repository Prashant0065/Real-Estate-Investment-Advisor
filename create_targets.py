import pandas as pd
from pathlib import Path

input_path = Path("data/processed/india_housing_prices_preprocessed.csv")
output_path = Path("data/processed/india_housing_prices_targets.csv")

df = pd.read_csv(input_path)

print("ORIGINAL DATASET SHAPE:", df.shape)

median_price = df["Price_in_Lakhs"].median()

df["Good_Investment"] = (
    df["Price_in_Lakhs"] <= median_price
).astype(int)

annual_appreciation_rate = 0.08
years = 5

df["Future_Price_5Y"] = (
    df["Price_in_Lakhs"]
    * (1 + annual_appreciation_rate) ** years
).round(2)

print("\nMEDIAN PROPERTY PRICE (LAKHS):", median_price)

print("\nGOOD INVESTMENT CLASS COUNTS:")
print(df["Good_Investment"].value_counts().sort_index())

print("\nGOOD INVESTMENT CLASS PROPORTIONS:")
print(
    df["Good_Investment"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nFUTURE PRICE CONFIGURATION:")
print("Annual appreciation rate:", annual_appreciation_rate * 100, "%")
print("Forecast period:", years, "years")

print("\nFUTURE PRICE SUMMARY (LAKHS):")
print(df["Future_Price_5Y"].describe())

print("\nTARGET VALIDATION:")
print("Missing Good_Investment values:", df["Good_Investment"].isna().sum())
print("Missing Future_Price_5Y values:", df["Future_Price_5Y"].isna().sum())
print("Unique classification labels:", sorted(df["Good_Investment"].unique()))
print(
    "Incorrect future-price calculations:",
    (
        df["Future_Price_5Y"]
        != (df["Price_in_Lakhs"] * (1 + annual_appreciation_rate) ** years).round(2)
    ).sum()
)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nTARGET DATASET SAVED SUCCESSFULLY")
print("OUTPUT PATH:", output_path)
print("FINAL DATASET SHAPE:", df.shape)
print("FINAL COLUMN COUNT:", len(df.columns))
