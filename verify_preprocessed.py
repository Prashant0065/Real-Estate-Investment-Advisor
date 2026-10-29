import pandas as pd
path = "data/processed/india_housing_prices_preprocessed.csv"
df = pd.read_csv(path)

print("DATASET SHAPE:" , df.shape)
print("\nCOLUMN NAMES:")
print(df.columns.tolist())
print("\nMISMATCH FLAG COUNTS:")
print(df["Price_per_SqFt_Mismatch"].value_counts(dropna=False))

print("\nMISSING VALUES:")
print(df.isnull().sum().sum())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nFIRST FIVE ROWS:")
print(df.head().to_string(index=False))