# Real Estate Investment Advisor --- EDA Findings

## 1. Purpose and dataset

This report documents the 20 Exploratory Data Analysis (EDA) questions
specified in the project brief for **Real Estate Investment Advisor:
Predicting Property Profitability & Future Value**.

-   Dataset shape reported by `eda_analysis.py`: **250,000 rows × 27
    columns**.
-   Current property price (`Price_in_Lakhs`): mean **₹254.59 lakh**,
    median **₹253.87 lakh**, standard deviation **₹141.35 lakh**,
    minimum **₹10 lakh**, maximum **₹500 lakh**.
-   Property size (`Size_in_SqFt`): mean **2,749.81 sq ft**, median
    **2,747 sq ft**, minimum **500 sq ft**, maximum **5,000 sq ft**.
-   The dataset check previously found **no missing values** and **no
    exact duplicate rows**.
-   Important data limitation: the stored `Price_per_SqFt` appears to be
    `Price_in_Lakhs / Size_in_SqFt`, rounded to two decimals. It is not
    a verified rupee-per-square-foot value. Where a true INR/sq-ft
    measure is needed, the analysis should use the separately derived
    calculation `Price_in_Lakhs * 100000 / Size_in_SqFt` and clearly
    label it as derived.

The findings below describe patterns in the supplied dataset. They
should not be interpreted as proof of causal relationships or as
market-wide facts about real estate in India.

------------------------------------------------------------------------

## 2. Price and size analysis (Questions 1--5)

### Q1. What is the distribution of property prices?

**Result:** Mean price is ₹254.59 lakh; median is ₹253.87 lakh; standard
deviation is ₹141.35 lakh; the range is ₹10--₹500 lakh. The 25th
percentile is ₹132.55 lakh and the 75th percentile is ₹376.88 lakh.

**Interpretation:** The mean and median are close, but prices span a
wide range. The dataset's price values are distributed across the full
stated range.

**Caution:** The dataset's price distribution appears unusually uniform
in some group comparisons. Validate the data-generation process before
assuming it represents observed market transactions.

### Q2. What is the distribution of property sizes?

**Result:** Mean size is 2,749.81 sq ft; median is 2,747 sq ft; standard
deviation is 1,300.61 sq ft; the range is 500--5,000 sq ft. The 25th and
75th percentiles are 1,623 and 3,874 sq ft.

**Interpretation:** The dataset covers a broad range of property sizes,
with the center near 2,750 sq ft.

### Q3. How does price per sq ft vary by property type?

**Result using the stored `Price_per_SqFt` column:**

  Property type          Count   Mean stored ratio   Median stored ratio
  ------------------- -------- ------------------- ---------------------
  Apartment             82,956            0.130493                  0.09
  Independent House     83,300            0.131023                  0.09
  Villa                 83,744            0.130276                  0.09

**Interpretation:** The stored ratio is very similar across property
types.

**Caution:** These values are not credible rupee-per-square-foot amounts
because the column appears to divide a price in lakhs by area. Do not
present this table as INR/sq ft. Recalculate a derived INR/sq-ft measure
for a properly labelled comparison.

### Q4. Is there a relationship between property size and price?

**Result:** Pearson correlation between `Size_in_SqFt` and
`Price_in_Lakhs` is **−0.0025**. The average prices by size group are
also close:

  -----------------------------------------------------------------------
  Size group (sq              Count      Mean price (₹    Median price (₹
  ft)                                            lakh)              lakh)
  -------------- ------------------ ------------------ ------------------
  Up to 1,000                27,997             255.95             257.54

  1,001--2,000               55,326             254.94             254.25

  2,001--3,000               55,664             254.03             253.59

  3,001--4,000               55,452             254.29             252.88

  4,001--5,000               55,561             254.41             252.71
  -----------------------------------------------------------------------

**Interpretation:** In this dataset, property size has virtually no
linear relationship with listed price. This is unusual for real estate
and is a likely contributor to poor model performance.

### Q5. Are there any outliers in price per sq ft or property size?

**Result using the IQR rule:**

-   `Price_in_Lakhs`: 0 potential outliers (0.00%).
-   `Size_in_SqFt`: 0 potential outliers (0.00%).
-   Stored `Price_per_SqFt`: 20,020 potential outliers (8.01%).

**Interpretation:** The IQR procedure did not flag price or size values,
but it flagged about 8.01% of stored price-per-sq-ft ratios.

**Caution:** Outlier flags do not automatically mean records are errors.
Because the stored ratio has a unit/definition concern, validate or
recalculate it before deciding whether to remove any observations. No
rows should be removed solely on the basis of this flag.

------------------------------------------------------------------------

## 3. Location-based analysis (Questions 6--10)

### Q6. What is the average price per sq ft by state?

**Result using the stored ratio:** Average values range from about
**0.1293** (Rajasthan and Punjab) to **0.1325** (Karnataka). Karnataka's
mean listed property price is ₹257.41 lakh; Delhi's is ₹252.53 lakh.

**Interpretation:** The stored ratio varies only slightly between
states.

**Caution:** Since the stored ratio is not a verified INR/sq-ft
measurement, this result is only a comparison of the stored column, not
a valid rupee-per-square-foot comparison. Recalculate the measure before
presenting a final price-per-sq-ft analysis.

### Q7. What is the average property price by city?

**Top 10 cities by average price from the recorded analysis:**

  City          Count   Mean price (₹ lakh)
  ----------- ------- ---------------------
  Bangalore     4,144                258.46
  Surat         6,167                258.08
  Kochi         6,305                257.42
  Gaya          6,170                257.05
  Mangalore     4,063                256.92
  Dehradun      6,234                256.89
  Mysore        4,217                256.85
  Chennai       6,168                256.83
  Hyderabad     6,273                256.59
  Bhopal        6,284                256.50

The lowest recorded city means range from Cuttack at ₹250.79 lakh to
Patna at ₹252.79 lakh in the bottom-10 list.

**Interpretation:** The difference between the highest and lowest listed
city means is relatively small compared with the overall price range.
This reinforces the need to investigate whether the dataset reflects
realistic location effects.

### Q8. What is the median age of properties by locality?

**Result:** In the recorded output, the oldest localities by median age
have median ages of **20--21 years**, while the youngest localities have
median ages of **16--17 years**. Example: Locality_72 has a median age
of 21 years; Locality_443 has a median age of 16 years.

**Interpretation:** The observed locality-level age differences are
modest.

**Caution:** Locality names such as `Locality_72` are coded labels, not
descriptive neighbourhood names. They should not be treated as
identifiable real-world localities without a mapping.

### Q9. How is BHK distributed across cities?

**Result:** Overall counts are:

    BHK   Properties    Share
  ----- ------------ --------
      1       50,196   20.08%
      2       49,815   19.93%
      3       50,067   20.03%
      4       49,788   19.92%
      5       50,134   20.05%

**Interpretation:** The five BHK categories are almost perfectly
balanced, each representing approximately 20% of the dataset. This
degree of balance may indicate synthetic or deliberately balanced data.

### Q10. What are the price trends for the top 5 most expensive localities?

**Result:** The top five localities by average price in the saved
analysis are:

  Locality         Count   Mean price (₹ lakh)   Median price (₹ lakh)
  -------------- ------- --------------------- -----------------------
  Locality_395       461                275.43                  282.68
  Locality_366       526                273.41                  282.33
  Locality_497       491                270.00                  273.69
  Locality_360       505                269.81                  278.85
  Locality_203       522                269.03                  280.32

**Interpretation:** These localities have higher average prices than the
dataset-wide mean of ₹254.59 lakh.

**Caution:** This is a comparison of locality-level prices, not a
time-series trend. The dataset does not establish how prices changed
over time.

------------------------------------------------------------------------

## 4. Feature relationships and correlation (Questions 11--15)

### Q11. How are numeric features correlated with each other?

**Result:** The recorded correlation matrix shows:

-   `Year_Built` and `Age_of_Property`: correlation **−1.000**.
-   `Size_in_SqFt` and stored `Price_per_SqFt`: correlation **−0.615**.
-   `Price_in_Lakhs` and stored `Price_per_SqFt`: correlation **0.556**.
-   Most other numeric-feature correlations with price are near zero;
    size and price correlation is **−0.0025**.

**Interpretation:** `Year_Built` and `Age_of_Property` are perfectly
redundant in this dataset because their sum is 2025 for every record.
Keeping both can introduce unnecessary redundancy. The price-per-sq-ft
correlations should be interpreted cautiously because the stored ratio's
units are questionable.

### Q12. How do nearby schools relate to price per sq ft?

**Result using the stored ratio:** Mean stored values by
`Nearby_Schools` count are approximately 0.1289--0.1329; medians are
0.09 for every category in the saved output.

**Interpretation:** The stored ratio shows little consistent variation
with the number of nearby schools.

**Caution:** This is not evidence that schools have no effect on real
property values. The dataset may lack realistic location variation, and
the ratio requires recalculation.

### Q13. How do nearby hospitals relate to price per sq ft?

**Result using the stored ratio:** Mean values by `Nearby_Hospitals`
count are approximately 0.1292--0.1326; medians are 0.09 across the
recorded categories.

**Interpretation:** No clear monotonic pattern appears in the recorded
averages.

### Q14. How does price vary by furnished status?

**Result:**

  -------------------------------------------------------------------------
  Furnished status              Count      Mean price (₹    Median price (₹
                                                   lakh)              lakh)
  ---------------- ------------------ ------------------ ------------------
  Unfurnished                  83,408             254.98             254.19

  Furnished                    83,218             254.45             253.56

  Semi-furnished               83,374             254.33             253.81
  -------------------------------------------------------------------------

**Interpretation:** Mean prices are close across the three categories,
with unfurnished properties having the highest mean in this dataset. The
small differences should not be interpreted as causal effects.

### Q15. How does price per sq ft vary by property facing direction?

**Result using the stored ratio:**

  Facing      Count   Mean stored ratio   Median stored ratio
  -------- -------- ------------------- ---------------------
  West       62,757              0.1314                  0.09
  South      62,337              0.1305                  0.09
  East       62,269              0.1303                  0.09
  North      62,637              0.1303                  0.09

**Interpretation:** The stored ratios are very similar by facing
direction. Recalculate the ratio before using this as a true
price-per-square-foot finding.

------------------------------------------------------------------------

## 5. Investment, amenities and ownership (Questions 16--20)

### Q16. How many properties belong to each owner type?

**Result:**

  Owner type      Count    Share
  ------------ -------- --------
  Broker         83,479   33.39%
  Owner          83,268   33.31%
  Builder        83,253   33.30%

**Interpretation:** Owner types are almost evenly distributed.

### Q17. How many properties are available under each availability status?

**Result:**

  Availability status       Count    Share
  --------------------- --------- --------
  Under construction      125,035   50.01%
  Ready to move           124,965   49.99%

**Interpretation:** The two availability categories are almost exactly
balanced.

### Q18. Does parking space affect property price?

**Result:**

  Parking space       Count   Mean price (₹ lakh)   Median price (₹ lakh)
  --------------- --------- --------------------- -----------------------
  Yes               124,544                254.75                  254.36
  No                125,456                254.43                  253.37

**Interpretation:** The average prices are very similar, with a
difference of ₹0.32 lakh between groups. This descriptive comparison
does not establish a causal effect of parking.

### Q19. How do amenities affect price per sq ft?

**Result from the recorded analysis of amenity combinations:** The top
15 combinations have mean stored ratios of approximately
**0.1400--0.1505**, with counts for each combination generally around
400--826 properties.

**Interpretation:** Some combinations have higher mean stored ratios
than the overall type averages, but many combinations have relatively
small counts. Because the stored ratio is not a verified INR/sq-ft
measure and amenity combinations vary in spelling/order, this result
should be treated as exploratory only. A stronger analysis would split
the amenities string into individual amenities and use a correctly
calculated INR/sq-ft measure.

### Q20. How does public transport accessibility relate to price per sq ft or investment potential?

**Result:**

  ---------------------------------------------------------------------------
  Accessibility            Count  Mean price (₹   Median price           Good
                                          lakh)       (₹ lakh)     Investment
                                                                        share
  --------------- -------------- -------------- -------------- --------------
  High                    83,705        254.920         254.71        49.860%

  Low                     83,287        254.864         254.27        49.927%

  Medium                  83,008        253.973         252.73        50.216%
  ---------------------------------------------------------------------------

**Interpretation:** Mean prices and the constructed investment-label
share are very similar across accessibility levels. The label share is
near 50% in every group, so it provides little evidence of meaningful
investment differentiation in this dataset.

------------------------------------------------------------------------

## 6. Cross-cutting data-quality findings

1.  The recorded dataset has 250,000 rows and 27 columns; prior checks
    reported no missing values and no exact duplicate rows.
2.  `Year_Built + Age_of_Property = 2025` for every row, so these
    features are perfectly redundant.
3.  `Floor_No > Total_Floors` was observed in 116,304 rows (46.52%).
    Investigate this anomaly; do not automatically alter or delete these
    records without a domain rule.
4.  The brief mentions `Crime_Rate`, `Infrastructure_Score`, and
    `School_Density_Score`, but these columns are not present in the
    supplied dataset. Their effects cannot be analysed from this data.
5.  `Good_Investment` is constructed using the median property price,
    rather than observed investment returns. The balanced label
    therefore does not measure actual profitability.
6.  `Future_Price_5Y` was generated using an assumed 8% annual growth
    rate. It is a constructed scenario target, not observed five-year
    sale data.
7.  The existing `Price_per_SqFt` appears to be a ratio of price in
    lakhs to area, not a correctly scaled rupee-per-square-foot value. A
    derived INR/sq-ft feature should be clearly labelled and used
    consistently.
8.  The very balanced category counts, near-identical group means, and
    near-zero correlations suggest that the data may be synthetic or
    have limited realistic signal. This is an inference from the
    observed patterns, not definitive proof of how the dataset was
    generated.

## 7. Overall conclusion

The EDA workflow covers all 20 questions in the project brief and
produces tables and charts for price, size, location, property
characteristics, amenities, ownership, availability, and transport
accessibility.

The most important finding is not a particular state or property type
being superior. It is that the dataset shows weak relationships between
many property characteristics and price, alongside target definitions
that are constructed rather than observed. These limitations help
explain why the current classification and regression models perform
poorly.

### Recommendations

-   Preserve the EDA outputs and document the target definitions
    transparently.
-   Use a correctly derived INR/sq-ft measure for any price-per-area
    comparisons.
-   Investigate the floor-number inconsistencies and redundant year/age
    features.
-   Do not claim causal relationships from group averages alone.
-   Do not claim that the current model reliably identifies profitable
    real estate investments or predicts actual future sale prices.
-   If real-world predictive performance is required, obtain data with
    actual transaction prices, location effects, and observed outcomes
    over time.

## 8. Existing output artifacts

The project logs indicate that `eda_analysis.py` saves question-specific
CSV summaries under `reports/` and figures under `reports/figures/`,
including:

-   `state_price_summary.csv`
-   `city_price_summary.csv`
-   `locality_age_summary.csv`
-   `bhk_distribution_by_city.csv`
-   `top_5_expensive_localities.csv`
-   `numeric_correlation_matrix.csv`
-   `schools_vs_price_per_sqft.csv`
-   `hospitals_vs_price_per_sqft.csv`
-   `price_by_furnished_status.csv`
-   `price_per_sqft_by_facing.csv`
-   `owner_type_distribution.csv`
-   `availability_status_distribution.csv`
-   `parking_vs_price.csv`
-   `amenities_vs_price_per_sqft.csv`
-   `public_transport_vs_investment.csv`

Confirm these files are present in the current project folder before
submission.
