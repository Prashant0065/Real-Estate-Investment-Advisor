
import pandas as pd

df = pd.read_csv("data/processed/india_housing_prices_preprocessed.csv")

df["Floor_Inconsistency"] = df["Floor_No"] > df["Total_Floors"]

print("\nFLOOR INCONSISTENCY BY PROPERTY TYPE:")
summary = df.groupby("Property_Type").agg(
    Total_Properties=("ID", "count"),
    Inconsistent_Rows=("Floor_Inconsistency", "sum")
)

summary["Inconsistency_Percentage"] = (
    summary["Inconsistent_Rows"] / summary["Total_Properties"] * 100
).round(2)

print(summary.to_string())

print("\nFLOOR VALUE DISTRIBUTION:")
print("\nFloor_No:")
print(df["Floor_No"].describe())

print("\nTotal_Floors:")
print(df["Total_Floors"].describe())

print("\nSAMPLE OF CONSISTENT ROWS:")
print(
    df.loc[
        ~df["Floor_Inconsistency"],
        ["ID", "Property_Type", "Floor_No", "Total_Floors"]
    ].head(10).to_string(index=False)
)

print("\nFLOOR RELATIONSHIP SUMMARY:")

print("\nFloor_No frequency:")
print(df["Floor_No"].value_counts().sort_index().to_string())

print("\nTotal_Floors frequency:")
print(df["Total_Floors"].value_counts().sort_index().to_string())

print("\nRows where Floor_No equals Total_Floors:")
print((df["Floor_No"] == df["Total_Floors"]).sum())

print("\nRows where Floor_No is zero:")
print((df["Floor_No"] == 0).sum())

print("\nRows where Total_Floors is one:")
print((df["Total_Floors"] == 1).sum())
