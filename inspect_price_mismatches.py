
import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/processed/india_housing_prices_preprocessed.csv"
)

exact_ratio = df["Price_in_Lakhs"] / df["Size_in_SqFt"]

expected_ratio = exact_ratio.round(2)

mismatch = ~np.isclose(
    df["Price_per_SqFt"],
    expected_ratio,
    atol=1e-9,
    rtol=0
)

print("=" * 75)
print("ALL 21 PRICE-PER-SQFT MISMATCHES")
print("=" * 75)

result = df.loc[
    mismatch,
    ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]
].copy()

result["Exact_Ratio"] = exact_ratio[mismatch]
result["Python_Rounded_Ratio"] = expected_ratio[mismatch]
result["Difference"] = (
    result["Price_per_SqFt"] - result["Exact_Ratio"]
)

print(result.round(6).to_string())

print("\n" + "=" * 75)
print("INVESTIGATION OF STORED ZERO VALUES")
print("=" * 75)

zero_mask = df["Price_per_SqFt"].eq(0)

zero_ratios = exact_ratio[zero_mask]

print("Number of stored zero values:", int(zero_mask.sum()))
print("Minimum exact ratio among zero values:", zero_ratios.min())
print("Maximum exact ratio among zero values:", zero_ratios.max())
print(
    "Zero values with exact ratio >= 0.005:",
    int((zero_ratios >= 0.005).sum())
)
print(
    "Zero values with positive exact ratio:",
    int((zero_ratios > 0).sum())
)

print("\n" + "=" * 75)
print("RUPEES-PER-SQFT VALIDATION")
print("=" * 75)

price_per_sqft_inr = (
    df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
)

print("Missing values:", int(price_per_sqft_inr.isna().sum()))
print("Infinite values:", int(np.isinf(price_per_sqft_inr).sum()))
print("Zero or negative values:", int((price_per_sqft_inr <= 0).sum()))
print("Minimum:", round(price_per_sqft_inr.min(), 2))
print("Median:", round(price_per_sqft_inr.median(), 2))
print("Maximum:", round(price_per_sqft_inr.max(), 2))

print("\nDIAGNOSTIC COMPLETE. NO DATA WAS MODIFIED.")
