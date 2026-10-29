
import pandas as pd

path = "data/processed/india_housing_prices_targets.csv"
df = pd.read_csv(path)

print("DATASET SHAPE:", df.shape)

print("\nTARGET COLUMNS:")
print(df[["Good_Investment", "Future_Price_5Y"]].head(10).to_string(index=False))

print("\nDATA TYPES:")
print(df[["Good_Investment", "Future_Price_5Y"]].dtypes)

print("\nMISSING VALUES:")
print(df[["Good_Investment", "Future_Price_5Y"]].isnull().sum())

print("\nCLASSIFICATION COUNTS:")
print(df["Good_Investment"].value_counts().sort_index())

print("\nFUTURE PRICE SUMMARY:")
print(df["Future_Price_5Y"].describe())

print("\nFORMULA CHECK:")
expected = (df["Price_in_Lakhs"] * (1.08 ** 5)).round(2)
print("Mismatches:", (df["Future_Price_5Y"] != expected).sum())
