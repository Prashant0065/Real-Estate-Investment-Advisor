
import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/india_housing_prices_preprocessed.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STEP 1: PRICE PER SQUARE FOOT INVESTIGATION")
print("=" * 70)

print("\nDATASET SHAPE:")
print(df.shape)

required_columns = [
    "Price_in_Lakhs",
    "Size_in_SqFt",
    "Price_per_SqFt"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# Recalculate the supplied column's apparent formula.
df["Expected_Stored_Ratio"] = (
    df["Price_in_Lakhs"] / df["Size_in_SqFt"]
).round(2)

# Calculate the actual price per square foot in rupees.
df["Calculated_Price_per_SqFt_INR"] = (
    df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
)

df["Stored_Ratio_Mismatch"] = ~np.isclose(
    df["Price_per_SqFt"],
    df["Expected_Stored_Ratio"],
    atol=1e-9,
    rtol=0
)

print("\n1. STORED COLUMN SUMMARY:")
print(df["Price_per_SqFt"].describe())

print("\n2. NUMBER OF STORED ZERO VALUES:")
print((df["Price_per_SqFt"] == 0).sum())

print("\n3. MISMATCH COUNT:")
print(df["Stored_Ratio_Mismatch"].sum())

print("\n4. FIRST 10 MISMATCHED RECORDS:")

mismatches = df.loc[
    df["Stored_Ratio_Mismatch"],
    [
        "Price_in_Lakhs",
        "Size_in_SqFt",
        "Price_per_SqFt",
        "Expected_Stored_Ratio",
        "Calculated_Price_per_SqFt_INR"
    ]
]

print(mismatches.head(10).to_string(index=True))

print("\n5. FIRST 10 STORED ZERO VALUES:")

zero_records = df.loc[
    df["Price_per_SqFt"] == 0,
    [
        "Price_in_Lakhs",
        "Size_in_SqFt",
        "Price_per_SqFt",
        "Expected_Stored_Ratio",
        "Calculated_Price_per_SqFt_INR"
    ]
]

print(zero_records.head(10).to_string(index=True))

print("\n6. CORRECTLY CONVERTED RUPEES PER SQUARE FOOT:")
print(
    df["Calculated_Price_per_SqFt_INR"]
    .describe()
    .round(2)
)

print("\n7. CHECK THE CONVERSION FOR THE FIRST 10 RECORDS:")

print(
    df[
        [
            "Price_in_Lakhs",
            "Size_in_SqFt",
            "Price_per_SqFt",
            "Expected_Stored_Ratio",
            "Calculated_Price_per_SqFt_INR"
        ]
    ].head(10).round(4).to_string(index=True)
)

print("\n8. MISMATCHES BY PROPERTY TYPE, IF AVAILABLE:")

if "Property_Type" in df.columns:
    print(
        df.groupby("Property_Type")[
            "Stored_Ratio_Mismatch"
        ].agg(["sum", "count"]).to_string()
    )

print("\nINVESTIGATION COMPLETE.")
print("No source values were modified or overwritten.")
