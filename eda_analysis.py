
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path("data/processed/india_housing_prices_targets.csv")
reports_dir = Path("reports")
output_dir = reports_dir / "figures"
reports_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

# Correctly calculate price per square foot in rupees.
# Preserve the original Price_per_SqFt column.
df["Price_per_SqFt_INR"] = (
    df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
)


print("DATASET SHAPE:", df.shape)

print("\nPROPERTY PRICE SUMMARY (LAKHS):")
print(df["Price_in_Lakhs"].describe())

print("\nPROPERTY PRICE MEDIAN (LAKHS):")
print(df["Price_in_Lakhs"].median())

plt.figure(figsize=(10, 6))
plt.hist(df["Price_in_Lakhs"], bins=50, edgecolor="black")
plt.title("Distribution of Property Prices")
plt.xlabel("Property Price (Lakhs)")
plt.ylabel("Number of Properties")
plt.tight_layout()

figure_path = output_dir / "price_distribution.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print("\nCHART SAVED TO:", figure_path)


print("\nPROPERTY SIZE SUMMARY (SQ FT):")
print(df["Size_in_SqFt"].describe())

print("\nPROPERTY SIZE MEDIAN (SQ FT):")
print(df["Size_in_SqFt"].median())

plt.figure(figsize=(10, 6))
plt.hist(df["Size_in_SqFt"], bins=50, edgecolor="black")
plt.title("Distribution of Property Sizes")
plt.xlabel("Property Size (Sq Ft)")
plt.ylabel("Number of Properties")
plt.tight_layout()

size_figure_path = output_dir / "size_distribution.png"
plt.savefig(size_figure_path, dpi=150)
plt.close()

print("\nSIZE CHART SAVED TO:", size_figure_path)


print("\nPRICE PER SQFT BY PROPERTY TYPE:")
price_by_type = df.groupby("Property_Type")["Price_per_SqFt_INR"].agg(
    ["count", "mean", "median", "min", "max"]
)
print(price_by_type)

plt.figure(figsize=(10, 6))
df.boxplot(column="Price_per_SqFt_INR", by="Property_Type")
plt.title("Price per Sq Ft by Property Type")
plt.suptitle("")
plt.xlabel("Property Type")
plt.ylabel("Price per Sq Ft (dataset units)")
plt.tight_layout()

type_figure_path = output_dir / "price_per_sqft_by_property_type.png"
plt.savefig(type_figure_path, dpi=150)
plt.close()

print("\nCHART SAVED TO:", type_figure_path)


print("\nSIZE AND PRICE CORRELATION:")
correlation = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
print("Pearson correlation:", round(correlation, 4))

plt.figure(figsize=(10, 6))
plt.scatter(
    df["Size_in_SqFt"],
    df["Price_in_Lakhs"],
    alpha=0.15,
    s=8
)
plt.title("Relationship Between Property Size and Price")
plt.xlabel("Property Size (Sq Ft)")
plt.ylabel("Property Price (Lakhs)")
plt.tight_layout()

scatter_path = output_dir / "size_vs_price.png"
plt.savefig(scatter_path, dpi=150)
plt.close()

print("\nSCATTER PLOT SAVED TO:", scatter_path)


print("\nCORRELATION BY PROPERTY TYPE:")
print(
    df.groupby("Property_Type").apply(
        lambda group: group["Size_in_SqFt"].corr(
            group["Price_in_Lakhs"]
        ),
        include_groups=False
    )
)

print("\nAVERAGE PRICE BY SIZE GROUP:")

df["Size_Group"] = pd.cut(
    df["Size_in_SqFt"],
    bins=[0, 1000, 2000, 3000, 4000, 5000],
    labels=[
        "Up to 1000",
        "1001-2000",
        "2001-3000",
        "3001-4000",
        "4001-5000"
    ],
    include_lowest=True
)

print(
    df.groupby("Size_Group", observed=False)["Price_in_Lakhs"]
    .agg(["count", "mean", "median"])
)


print("\nOUTLIER ANALYSIS USING IQR")

columns_to_check = [
    "Price_in_Lakhs",
    "Size_in_SqFt",
    "Price_per_SqFt_INR"
]

for column in columns_to_check:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_bound)
        | (df[column] > upper_bound)
    ]

    print(f"\n{column}:")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)
    print("Potential outliers:", len(outliers))
    print("Outlier percentage:", round(len(outliers) / len(df) * 100, 2), "%")


print("\nAVERAGE PROPERTY PRICE AND PRICE PER SQFT BY STATE:")

state_summary = df.groupby("State").agg(
    Property_Count=("ID", "count"),
    Average_Price_Lakhs=("Price_in_Lakhs", "mean"),
    Median_Price_Lakhs=("Price_in_Lakhs", "median"),
    Average_Price_per_SqFt_INR=("Price_per_SqFt_INR", "mean")
).round(4)

print(
    state_summary.sort_values(
        "Average_Price_per_SqFt_INR",
        ascending=False
    ).to_string()
)

state_summary.sort_values(
    "Average_Price_per_SqFt_INR",
    ascending=False
).to_csv("reports/state_price_summary.csv")

print("\nSTATE SUMMARY SAVED TO: reports/state_price_summary.csv")


print("\nAVERAGE PROPERTY PRICE BY CITY:")

city_summary = df.groupby("City").agg(
    Property_Count=("ID", "count"),
    Average_Price_Lakhs=("Price_in_Lakhs", "mean"),
    Median_Price_Lakhs=("Price_in_Lakhs", "median")
).round(2)

print("\nTOP 10 CITIES BY AVERAGE PROPERTY PRICE:")
print(
    city_summary.sort_values(
        "Average_Price_Lakhs",
        ascending=False
    ).head(10).to_string()
)

print("\nBOTTOM 10 CITIES BY AVERAGE PROPERTY PRICE:")
print(
    city_summary.sort_values(
        "Average_Price_Lakhs",
        ascending=True
    ).head(10).to_string()
)

city_summary.to_csv("reports/city_price_summary.csv")
print("\nCITY SUMMARY SAVED TO: reports/city_price_summary.csv")



print("\nMEDIAN PROPERTY AGE BY LOCALITY:")

locality_age = df.groupby("Locality").agg(
    Property_Count=("ID", "count"),
    Median_Age=("Age_of_Property", "median"),
    Average_Age=("Age_of_Property", "mean")
).round(2)

print("\nFIRST 10 LOCALITIES:")
print(locality_age.head(10).to_string())

print("\nOLDEST 10 LOCALITIES BY MEDIAN PROPERTY AGE:")
print(
    locality_age.sort_values(
        "Median_Age",
        ascending=False
    ).head(10).to_string()
)

print("\nYOUNGEST 10 LOCALITIES BY MEDIAN PROPERTY AGE:")
print(
    locality_age.sort_values(
        "Median_Age",
        ascending=True
    ).head(10).to_string()
)

locality_age.to_csv("reports/locality_age_summary.csv")
print("\nLOCALITY SUMMARY SAVED TO: reports/locality_age_summary.csv")


print("\nBHK DISTRIBUTION ACROSS CITIES:")

bhk_city = pd.crosstab(df["City"], df["BHK"])

print("\nBHK COUNTS FOR THE FIRST 10 CITIES:")
print(bhk_city.head(10).to_string())

print("\nTOTAL PROPERTIES BY BHK:")
print(df["BHK"].value_counts().sort_index())

print("\nPERCENTAGE DISTRIBUTION OF BHK:")
print(
    (df["BHK"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

bhk_city.to_csv("reports/bhk_distribution_by_city.csv")
print("\nBHK SUMMARY SAVED TO: reports/bhk_distribution_by_city.csv")


# ============================================================
# QUESTION 10: TOP 5 MOST EXPENSIVE LOCALITIES
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 10: TOP 5 MOST EXPENSIVE LOCALITIES")
print("=" * 65)

locality_price = df.groupby("Locality").agg(
    Property_Count=("Price_in_Lakhs", "count"),
    Average_Price_Lakhs=("Price_in_Lakhs", "mean"),
    Median_Price_Lakhs=("Price_in_Lakhs", "median")
)

top_5_localities = locality_price.sort_values(
    "Average_Price_Lakhs", ascending=False
).head(5)

print(top_5_localities.round(2).to_string())

top_5_localities.to_csv("reports/top_5_expensive_localities.csv")

plt.figure(figsize=(10, 6))
plt.bar(
    top_5_localities.index,
    top_5_localities["Average_Price_Lakhs"]
)
plt.title("Top 5 Localities by Average Property Price")
plt.xlabel("Locality")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("reports/figures/top_5_expensive_localities.png", dpi=150)
plt.close()

print("Saved: reports/top_5_expensive_localities.csv")
print("Saved: reports/figures/top_5_expensive_localities.png")


# ============================================================
# QUESTION 11: CORRELATION AMONG NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 11: NUMERICAL FEATURE CORRELATIONS")
print("=" * 65)

numeric_columns = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt_INR",
    "Year_Built",
    "Age_of_Property",
    "Floor_No",
    "Total_Floors",
    "Nearby_Schools",
    "Nearby_Hospitals"
]

numeric_columns = [
    col for col in numeric_columns if col in df.columns
]

correlation_matrix = df[numeric_columns].corr()

print(correlation_matrix.round(3).to_string())

correlation_matrix.to_csv("reports/numeric_correlation_matrix.csv")

plt.figure(figsize=(12, 9))
image_plot = plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)
plt.colorbar(image_plot, label="Correlation")
plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(numeric_columns)), numeric_columns)

for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        plt.text(
            j, i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig("reports/figures/numeric_correlation_matrix.png", dpi=150)
plt.close()

print("Saved: reports/numeric_correlation_matrix.csv")
print("Saved: reports/figures/numeric_correlation_matrix.png")


# ============================================================
# QUESTION 12: NEARBY SCHOOLS VS PRICE PER SQFT
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 12: NEARBY SCHOOLS VS PRICE PER SQFT")
print("=" * 65)

schools_summary = df.groupby("Nearby_Schools").agg(
    Property_Count=("Price_per_SqFt_INR", "count"),
    Average_Price_per_SqFt_INR=("Price_per_SqFt_INR", "mean"),
    Median_Price_per_SqFt_INR=("Price_per_SqFt_INR", "median")
).sort_index()

print(schools_summary.round(4).to_string())

schools_summary.to_csv("reports/schools_vs_price_per_sqft.csv")

plt.figure(figsize=(10, 6))
plt.plot(
    schools_summary.index,
    schools_summary["Average_Price_per_SqFt_INR"],
    marker="o"
)
plt.title("Nearby Schools vs Average Price per Sq Ft (₹)")
plt.xlabel("Number of Nearby Schools")
plt.ylabel("Average Price per Sq Ft (₹)")
plt.tight_layout()
plt.savefig("reports/figures/schools_vs_price_per_sqft.png", dpi=150)
plt.close()

print("Saved: reports/schools_vs_price_per_sqft.csv")
print("Saved: reports/figures/schools_vs_price_per_sqft.png")


# ============================================================
# QUESTION 13: NEARBY HOSPITALS VS PRICE PER SQFT
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 13: NEARBY HOSPITALS VS PRICE PER SQFT")
print("=" * 65)

hospitals_summary = df.groupby("Nearby_Hospitals").agg(
    Property_Count=("Price_per_SqFt_INR", "count"),
    Average_Price_per_SqFt_INR=("Price_per_SqFt_INR", "mean"),
    Median_Price_per_SqFt_INR=("Price_per_SqFt_INR", "median")
).sort_index()

print(hospitals_summary.round(4).to_string())

hospitals_summary.to_csv("reports/hospitals_vs_price_per_sqft.csv")

plt.figure(figsize=(10, 6))
plt.plot(
    hospitals_summary.index,
    hospitals_summary["Average_Price_per_SqFt_INR"],
    marker="o"
)
plt.title("Nearby Hospitals vs Average Price per Sq Ft (₹)")
plt.xlabel("Number of Nearby Hospitals")
plt.ylabel("Average Price per Sq Ft (₹)")
plt.tight_layout()
plt.savefig("reports/figures/hospitals_vs_price_per_sqft.png", dpi=150)
plt.close()

print("Saved: reports/hospitals_vs_price_per_sqft.csv")
print("Saved: reports/figures/hospitals_vs_price_per_sqft.png")


# ============================================================
# QUESTION 14: PRICE BY FURNISHED STATUS
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 14: PROPERTY PRICE BY FURNISHED STATUS")
print("=" * 65)

furnished_summary = df.groupby("Furnished_Status").agg(
    Property_Count=("Price_in_Lakhs", "count"),
    Average_Price_Lakhs=("Price_in_Lakhs", "mean"),
    Median_Price_Lakhs=("Price_in_Lakhs", "median")
).sort_values("Average_Price_Lakhs", ascending=False)

print(furnished_summary.round(2).to_string())

furnished_summary.to_csv("reports/price_by_furnished_status.csv")

plt.figure(figsize=(9, 6))
plt.bar(
    furnished_summary.index.astype(str),
    furnished_summary["Average_Price_Lakhs"]
)
plt.title("Average Property Price by Furnished Status")
plt.xlabel("Furnished Status")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("reports/figures/price_by_furnished_status.png", dpi=150)
plt.close()

print("Saved: reports/price_by_furnished_status.csv")
print("Saved: reports/figures/price_by_furnished_status.png")


# ============================================================
# QUESTION 15: PRICE PER SQFT BY FACING
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 15: PRICE PER SQFT BY FACING")
print("=" * 65)

facing_summary = df.groupby("Facing").agg(
    Property_Count=("Price_per_SqFt_INR", "count"),
    Average_Price_per_SqFt_INR=("Price_per_SqFt_INR", "mean"),
    Median_Price_per_SqFt_INR=("Price_per_SqFt_INR", "median")
).sort_values("Average_Price_per_SqFt_INR", ascending=False)

print(facing_summary.round(4).to_string())

facing_summary.to_csv("reports/price_per_sqft_by_facing.csv")

plt.figure(figsize=(10, 6))
plt.bar(
    facing_summary.index.astype(str),
    facing_summary["Average_Price_per_SqFt_INR"]
)
plt.title("Average Stored Price per Sq Ft (₹) by Facing")
plt.xlabel("Facing")
plt.ylabel("Average Price per Sq Ft (₹)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("reports/figures/price_per_sqft_by_facing.png", dpi=150)
plt.close()

print("Saved: reports/price_per_sqft_by_facing.csv")
print("Saved: reports/figures/price_per_sqft_by_facing.png")


# ============================================================
# QUESTION 16: OWNER TYPE COUNTS
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 16: OWNER TYPE DISTRIBUTION")
print("=" * 65)

owner_summary = df["Owner_Type"].value_counts(dropna=False).to_frame(
    name="Property_Count"
)

owner_summary["Percentage"] = (
    owner_summary["Property_Count"] / len(df) * 100
)

print(owner_summary.round(2).to_string())

owner_summary.to_csv("reports/owner_type_distribution.csv")

plt.figure(figsize=(9, 6))
plt.bar(
    owner_summary.index.astype(str),
    owner_summary["Property_Count"]
)
plt.title("Property Count by Owner Type")
plt.xlabel("Owner Type")
plt.ylabel("Number of Properties")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("reports/figures/owner_type_distribution.png", dpi=150)
plt.close()

print("Saved: reports/owner_type_distribution.csv")
print("Saved: reports/figures/owner_type_distribution.png")


# ============================================================
# QUESTION 17: AVAILABILITY STATUS COUNTS
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 17: AVAILABILITY STATUS DISTRIBUTION")
print("=" * 65)

availability_summary = df["Availability_Status"].value_counts(
    dropna=False
).to_frame(name="Property_Count")

availability_summary["Percentage"] = (
    availability_summary["Property_Count"] / len(df) * 100
)

print(availability_summary.round(2).to_string())

availability_summary.to_csv("reports/availability_status_distribution.csv")

plt.figure(figsize=(9, 6))
plt.bar(
    availability_summary.index.astype(str),
    availability_summary["Property_Count"]
)
plt.title("Property Count by Availability Status")
plt.xlabel("Availability Status")
plt.ylabel("Number of Properties")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(
    "reports/figures/availability_status_distribution.png",
    dpi=150
)
plt.close()

print("Saved: reports/availability_status_distribution.csv")
print("Saved: reports/figures/availability_status_distribution.png")


# ============================================================
# QUESTION 18: PARKING SPACE VS PROPERTY PRICE
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 18: PARKING SPACE VS PROPERTY PRICE")
print("=" * 65)

parking_summary = df.groupby("Parking_Space").agg(
    Property_Count=("Price_in_Lakhs", "count"),
    Average_Price_Lakhs=("Price_in_Lakhs", "mean"),
    Median_Price_Lakhs=("Price_in_Lakhs", "median")
).sort_values("Average_Price_Lakhs", ascending=False)

print(parking_summary.round(2).to_string())

parking_summary.to_csv("reports/parking_vs_price.csv")

plt.figure(figsize=(9, 6))
plt.bar(
    parking_summary.index.astype(str),
    parking_summary["Average_Price_Lakhs"]
)
plt.title("Average Property Price by Parking Space")
plt.xlabel("Parking Space")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("reports/figures/parking_vs_price.png", dpi=150)
plt.close()

print("Saved: reports/parking_vs_price.csv")
print("Saved: reports/figures/parking_vs_price.png")


# ============================================================
# ============================================================
# QUESTION 19: INDIVIDUAL AMENITIES VS PRICE PER SQFT
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 19: INDIVIDUAL AMENITIES VS PRICE PER SQFT")
print("=" * 65)

# Normalize each amenity and analyze each item separately.
# This avoids treating different orderings of the same amenities as different categories.
amenity_records = []
for row_index, row in df[["Amenities", "Price_per_SqFt_INR"]].iterrows():
    if pd.isna(row["Amenities"]):
        continue

    individual_amenities = {
        item.strip().casefold()
        for item in str(row["Amenities"]).split(",")
        if item.strip()
    }

    for amenity in individual_amenities:
        amenity_records.append({
            "Amenity": amenity,
            "Price_per_SqFt_INR": row["Price_per_SqFt_INR"],
            "Row_Index": row_index
        })

if amenity_records:
    amenity_df = pd.DataFrame(amenity_records)
    amenities_summary = amenity_df.groupby("Amenity").agg(
        Property_Count=("Row_Index", "nunique"),
        Average_Price_per_SqFt_INR=("Price_per_SqFt_INR", "mean"),
        Median_Price_per_SqFt_INR=("Price_per_SqFt_INR", "median")
    ).sort_values("Average_Price_per_SqFt_INR", ascending=False)

    print("\nTOP 15 INDIVIDUAL AMENITIES BY AVERAGE PRICE PER SQFT:")
    print(amenities_summary.head(15).round(2).to_string())
    amenities_summary.to_csv("reports/amenities_vs_price_per_sqft.csv")

    amenities_plot_data = amenities_summary.head(15).sort_values(
        "Average_Price_per_SqFt_INR"
    )

    plt.figure(figsize=(11, 7))
    plt.barh(
        amenities_plot_data.index.astype(str),
        amenities_plot_data["Average_Price_per_SqFt_INR"]
    )
    plt.title("Top 15 Individual Amenities by Average Price per Sq Ft")
    plt.xlabel("Average Price per Sq Ft (₹)")
    plt.ylabel("Individual Amenity")
    plt.tight_layout()
    plt.savefig("reports/figures/amenities_vs_price_per_sqft.png", dpi=150)
    plt.close()

    print("Saved: reports/amenities_vs_price_per_sqft.csv")
    print("Saved: reports/figures/amenities_vs_price_per_sqft.png")
else:
    print("No non-empty amenity values were found.")


# QUESTION 20: PUBLIC TRANSPORT ACCESSIBILITY
# ============================================================

print("\n" + "=" * 65)
print("QUESTION 20: PUBLIC TRANSPORT ACCESSIBILITY")
print("=" * 65)

transport_summary = df.groupby("Public_Transport_Accessibility").agg(
    Property_Count=("Price_in_Lakhs", "count"),
    Average_Price_Lakhs=("Price_in_Lakhs", "mean"),
    Median_Price_Lakhs=("Price_in_Lakhs", "median"),
    Average_Price_per_SqFt_INR=("Price_per_SqFt_INR", "mean"),
    Good_Investment_Rate=("Good_Investment", "mean")
).sort_values("Average_Price_Lakhs", ascending=False)

transport_summary["Good_Investment_Percentage"] = (
    transport_summary["Good_Investment_Rate"] * 100
)

print(
    transport_summary.drop(
        columns=["Good_Investment_Rate"]
    ).round(3).to_string()
)

transport_summary.to_csv(
    "reports/public_transport_vs_investment.csv"
)

plt.figure(figsize=(10, 6))
plt.bar(
    transport_summary.index.astype(str),
    transport_summary["Average_Price_Lakhs"]
)
plt.title("Average Property Price by Public Transport Accessibility")
plt.xlabel("Public Transport Accessibility")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(
    "reports/figures/public_transport_vs_price.png",
    dpi=150
)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(
    transport_summary.index.astype(str),
    transport_summary["Good_Investment_Percentage"]
)
plt.title("Good Investment Label Percentage by Transport Accessibility")
plt.xlabel("Public Transport Accessibility")
plt.ylabel("Properties Labelled Good Investment (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(
    "reports/figures/public_transport_vs_investment.png",
    dpi=150
)
plt.close()

print("Saved: reports/public_transport_vs_investment.csv")
print("Saved: reports/figures/public_transport_vs_price.png")
print("Saved: reports/figures/public_transport_vs_investment.png")


# ============================================================
# ============================================================
# ADDITIONAL DATA-QUALITY CHECKS
# ============================================================

print("\n" + "=" * 65)
print("ADDITIONAL DATA-QUALITY CHECKS")
print("=" * 65)

expected_stored_ratio = (
    df["Price_in_Lakhs"] / df["Size_in_SqFt"]
).round(2)

stored_ratio_mismatch = ~np.isclose(
    df["Price_per_SqFt"],
    expected_stored_ratio,
    atol=1e-9,
    rtol=0
)

zero_mask = df["Price_per_SqFt"].eq(0)
exact_zero_ratios = (
    df.loc[zero_mask, "Price_in_Lakhs"]
    / df.loc[zero_mask, "Size_in_SqFt"]
)

print("\nStored Price_per_SqFt zero values:", int(zero_mask.sum()))
print("Mismatches against Python's rounded ratio:", int(stored_ratio_mismatch.sum()))
print("Stored zero ratios with exact ratio >= 0.005:", int((exact_zero_ratios >= 0.005).sum()))
print("Price_per_SqFt_INR missing values:", int(df["Price_per_SqFt_INR"].isna().sum()))
print("Price_per_SqFt_INR infinite values:", int(np.isinf(df["Price_per_SqFt_INR"]).sum()))
print("Price_per_SqFt_INR non-positive values:", int((df["Price_per_SqFt_INR"] <= 0).sum()))

year_age_sum = df["Year_Built"] + df["Age_of_Property"]
print("\nRows where Year_Built + Age_of_Property != 2025:",
      int((year_age_sum != 2025).sum()))

floor_inconsistent = df["Floor_No"] > df["Total_Floors"]
print("\nRows where Floor_No > Total_Floors:", int(floor_inconsistent.sum()))
print("Percentage of dataset:", round(floor_inconsistent.mean() * 100, 2), "%")

floor_summary = df.assign(Floor_Inconsistent=floor_inconsistent).groupby(
    "Property_Type"
)["Floor_Inconsistent"].agg(["sum", "count"])
print("\nFloor inconsistencies by property type:")
print(floor_summary.to_string())
floor_summary.to_csv("reports/floor_inconsistency_summary.csv")

floor_examples = df.loc[
    floor_inconsistent,
    ["Property_Type", "Floor_No", "Total_Floors", "City", "Locality"]
].head(20)
floor_examples.to_csv("reports/floor_inconsistency_examples.csv", index=False)
print("Saved: reports/floor_inconsistency_summary.csv")
print("Saved: reports/floor_inconsistency_examples.csv")

# ============================================================
# PROJECT BRIEF DATA AVAILABILITY AND LIMITATIONS
# ============================================================

print("\n" + "=" * 65)
print("PROJECT BRIEF DATA AVAILABILITY CHECK")
print("=" * 65)

required_but_missing = [
    "Crime_Rate",
    "Infrastructure_Score",
    "School_Density_Score",
    "Latitude",
    "Longitude",
    "RERA_Status"
]

for column in required_but_missing:
    if column not in df.columns:
        print(f"NOT AVAILABLE: {column} - cannot analyze it using the current dataset.")

print(
    "\nNOTE: Price_per_SqFt is preserved from the source dataset. It appears "
    "to be a rounded ratio of price in lakhs to area, not a verified rupee "
    "per-square-foot value. Use Price_per_SqFt_INR for rupee-denominated EDA."
)
print(
    "\nNOTE: The 21 stored-ratio mismatches occur at half-cent boundaries "
    "and may result from different rounding conventions. Original values "
    "have not been overwritten."
)
print(
    "\nNOTE: Good_Investment is a constructed label based on whether the "
    "current price is at or below the dataset median. It is not observed "
    "investment profitability."
)
print(
    "\nNOTE: Future_Price_5Y is generated using the assumed 8% annual "
    "appreciation formula. It is a scenario estimate, not an observed future "
    "sale price or independent forecasting label."
)
print(
    "\nNOTE: Year_Built and Age_of_Property are redundant if their sum is "
    "always 2025. Avoid treating both as independent model signals without a reason."
)
print(
    "\nNOTE: EDA describes associations and does not establish causation. "
    "The dataset's balanced categories and weak size-price correlation should "
    "be investigated before model training."
)
print(
    "\nNOTE: Floor inconsistencies are reported but not automatically fixed "
    "or removed because their intended definitions are not yet confirmed."
)

print("\nALL EDA SECTIONS FINISHED.")
print("Review the summaries and generated files before modelling.")
