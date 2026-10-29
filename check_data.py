import pandas as pd

DATA_PATH = "data/processed/india_housing_prices_targets.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nMissing values across all columns:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values found.")

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
print(df["Good_Investment"].value_counts())

print("\nFuture price target:")
print(df["Future_Price_5Y"].describe().round(2))
