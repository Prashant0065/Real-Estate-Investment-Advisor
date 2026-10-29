import pandas as pd
df = pd.read_csv("data/processed/india_housing_prices_preprocessed.csv")
print("YEAR BUILT RANGE:")
print(df["Year_Built"].min(), "to", df["Year_Built"].max())
print("\nAGE OF PROPERTY RANGE:")
print(df["Age_of_Property"].min(), "to", df["Age_of_Property"].max())
print("\nFLOOR NUMBER GREATER THAN TOTAL FLOORS:")
invalid_floors = df[df["Floor_No"] > df["Total_Floors"]]
print("Count:", len(invalid_floors))
print(
    invalid_floors[
        ["ID", "Floor_No", "Total_Floors"]
    ].head(20).to_string(index=False)
)
print("\nAGE CONSISTENCY CHECK:")
print(
    "Rows where Year_Built + Age_of_Property differs from 2025:",
    (df["Year_Built"] + df["Age_of_Property"] != 2025).sum()
)
print("\nNON-POSITIVE SIZE:")
print((df["Size_in_SqFt"] <= 0).sum())

print("\nNON-POSITIVE PRICE:")
print((df["Price_in_Lakhs"] <= 0).sum())