
import pandas as pd
df = pd.read_csv("data/raw/india_housing_prices.csv")
df["Calculated_Price_per_SqFt"] = (df["Price_in_Lakhs"] / df["Size_in_SqFt"]).round(2)
mismatches = df[df["Price_per_SqFt"] != df["Calculated_Price_per_SqFt"]]
zero_values = df.loc[df["Price_per_SqFt"].eq(0)]
print("\nDATASET SHAPE:", df.shape)
print("TOTAL MISMATCHES:", len(mismatches))
print("\nFIRST 30 MISMATCHES:")
print(mismatches[["ID", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Calculated_Price_per_SqFt"]].head(30).to_string(index=False))
print("\nFIRST 20 ZERO-VALUE ROWS:")
print(zero_values[["ID", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Calculated_Price_per_SqFt"]].head(20).to_string(index=False))
print("\nORIGINAL PRICE PER SQFT SUMMARY:")
print(df["Price_per_SqFt"].describe())
print("\nCALCULATED PRICE PER SQFT SUMMARY:")
print(df["Calculated_Price_per_SqFt"].describe())
print("\n--- MISMATCH ANALYSIS ---")
print(
    "Rows where stored and calculated values differ:",
    (df["Price_per_SqFt"] != df["Calculated_Price_per_SqFt"]).sum()
)
print(
    "Rows with positive price and positive area:",
    (
        (df["Price_in_Lakhs"] > 0)
        & (df["Size_in_SqFt"] > 0)
    ).sum()
)
print(
    "Rows with zero stored price per sqft but positive price and area:",
    (
        (df["Price_per_SqFt"] == 0)
        & (df["Price_in_Lakhs"] > 0)
        & (df["Size_in_SqFt"] > 0)
    ).sum()
)
print("\nMISMATCHES WITH FULL PRECISION:")
print(
    mismatches[
        [
            "ID",
            "Price_in_Lakhs",
            "Size_in_SqFt",
            "Price_per_SqFt",
            "Calculated_Price_per_SqFt",
        ]
    ].to_string(index=False)
)