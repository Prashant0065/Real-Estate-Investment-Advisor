from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Project paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "india_housing_prices_targets.csv"
CLASSIFIER_PATH = PROJECT_DIR / "models" / "investment_classifier.pkl"
REGRESSOR_PATH = PROJECT_DIR / "models" / "price_regressor.pkl"
REPORT_DIR = PROJECT_DIR / "reports"

EXCLUDED_COLUMNS = [
    "ID",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Calculated_Price_per_SqFt",
    "Price_per_SqFt_Mismatch",
    "Good_Investment",
    "Future_Price_5Y",
]

ANNUAL_GROWTH_RATE = 0.08

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 Real Estate Investment Advisor")
st.write(
    "Explore property characteristics, review experimental model predictions, "
    "and visualize an illustrative five-year price scenario."
)


# --------------------------------------------------
# Data and model loading
# --------------------------------------------------

@st.cache_data(show_spinner="Loading property dataset...")
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Loading saved models...")
def load_models():
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    return classifier, regressor


def load_report(filename):
    """Load a report and preserve a saved index as a meaningful label column."""
    path = REPORT_DIR / filename
    if not path.exists():
        return None

    report = pd.read_csv(path)
    unnamed_columns = [
        column for column in report.columns
        if str(column).startswith("Unnamed:")
    ]

    # The first unnamed column in these CSVs contains row labels (e.g. 0, 1,
    # accuracy, macro avg). Preserve those labels instead of displaying an
    # empty index or deleting useful information.
    if unnamed_columns:
        first_unnamed = unnamed_columns[0]
        if filename == "confusion_matrix.csv":
            report = report.rename(columns={first_unnamed: "Actual Class"})
        elif filename == "classification_report.csv":
            report = report.rename(columns={first_unnamed: "Class / Metric"})
        else:
            report = report.drop(columns=unnamed_columns)

    # Remove any other accidental CSV index columns.
    remaining_unnamed = [
        column for column in report.columns
        if str(column).startswith("Unnamed:")
    ]
    if remaining_unnamed:
        report = report.drop(columns=remaining_unnamed)

    return report


def metric_from_classification_report(report, row_name, column_name):
    """Return a metric as a percentage from a sklearn classification report."""
    if report is None or column_name not in report.columns:
        return None

    label_column = "Class / Metric"
    if label_column not in report.columns:
        return None

    labels = report[label_column].astype(str).str.strip().str.lower()
    matches = report.loc[labels == row_name.lower(), column_name]
    if matches.empty:
        return None

    value = pd.to_numeric(matches.iloc[0], errors="coerce")
    if pd.isna(value):
        return None
    return float(value) * 100


def metric_from_regression_report(report, model_name, metric_name):
    """Find a metric for a named model in the regression comparison report."""
    if report is None or "Model" not in report.columns or metric_name not in report.columns:
        return None

    names = report["Model"].astype(str).str.strip().str.lower()
    matches = report.loc[names == model_name.lower(), metric_name]
    if matches.empty:
        return None

    value = pd.to_numeric(matches.iloc[0], errors="coerce")
    return None if pd.isna(value) else float(value)


def format_percent(value):
    return "Not available" if value is None else f"{value:.2f}%"


def format_lakh(value):
    return "Not available" if value is None else f"₹{value:,.2f} lakh"


# Validate required files before loading.
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
    st.error(
        "One or both saved models are missing. Check the models folder for "
        "investment_classifier.pkl and price_regressor.pkl."
    )
    st.stop()

try:
    df = load_data()
    classifier, regressor = load_models()
except Exception as exc:
    st.error(f"Could not load the dataset or saved models: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset is empty. Check the CSV file and try again.")
    st.stop()

# Keep only the features expected to be supplied to the trained models.
X_data = df.drop(columns=EXCLUDED_COLUMNS, errors="ignore")

if X_data.empty:
    st.error("No model input features were found after excluding target columns.")
    st.stop()

# If the pipeline exposes feature_names_in_, detect a dataset/model mismatch early.
for model_label, model in (("classifier", classifier), ("regressor", regressor)):
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None and list(expected_features) != list(X_data.columns):
        st.error(
            f"The saved {model_label} expects a different set/order of input features. "
            "Please check the training script and dataset columns before using predictions."
        )
        st.stop()

numeric_columns = X_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X_data.select_dtypes(
    include=["object", "category", "str", "string"]
).columns.tolist()


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------

st.caption(
    f"Dataset: {len(df):,} properties | "
    f"Model inputs: {len(X_data.columns)} features"
)

overview_cols = st.columns(3)
overview_cols[0].metric("Properties in Dataset", f"{len(df):,}")
overview_cols[1].metric("Model Input Features", len(X_data.columns))
overview_cols[2].metric("Categorical Features", len(categorical_columns))


# --------------------------------------------------
# Property evaluation form
# --------------------------------------------------

st.divider()
st.header("Evaluate a Property")

with st.form("property_form"):
    st.subheader("Property Details")
    st.write("Enter the property characteristics you want to evaluate.")

    inputs = {}
    numeric_layout = st.columns(2)

    for i, column in enumerate(numeric_columns):
        values = pd.to_numeric(X_data[column], errors="coerce").dropna()

        if values.empty:
            inputs[column] = 0
            continue

        minimum = float(values.min())
        maximum = float(values.max())
        default = float(values.median())

        # Streamlit number_input requires min_value <= value <= max_value.
        default = min(max(default, minimum), maximum)
        with numeric_layout[i % 2]:
            inputs[column] = st.number_input(
                column.replace("_", " "),
                min_value=minimum,
                max_value=maximum,
                value=default,
                help=f"Observed dataset range: {minimum:g} to {maximum:g}",
            )

    st.subheader("Location and Property Features")
    categorical_layout = st.columns(2)

    for i, column in enumerate(categorical_columns):
        options = sorted(X_data[column].dropna().astype(str).unique().tolist())

        if not options:
            inputs[column] = ""
            continue

        with categorical_layout[i % 2]:
            inputs[column] = st.selectbox(
                column.replace("_", " "),
                options=options,
            )

    submitted = st.form_submit_button(
        "Evaluate Property",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction results and five-year projection
# --------------------------------------------------

if submitted:
    property_data = pd.DataFrame(
        [[inputs[column] for column in X_data.columns]],
        columns=X_data.columns,
    )

    try:
        prediction = int(classifier.predict(property_data)[0])
        future_price = float(regressor.predict(property_data)[0])

        if not np.isfinite(future_price):
            raise ValueError("The regression model returned a non-finite price.")
        if future_price <= 0:
            st.error(
                f"The regression model returned a non-positive estimated price "
                f"({future_price:.2f} lakh). This prediction should not be interpreted "
                "as a valid property valuation."
            )
        else:
            st.divider()
            st.header("Property Evaluation")

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                if prediction == 1:
                    st.success("Predicted category: Good Investment")
                else:
                    st.warning("Predicted category: Not Good Investment")

            with result_col2:
                st.metric(
                    "Estimated Price After 5 Years",
                    f"₹{future_price:,.2f} lakh",
                )

            st.subheader("Five-Year Price Projection")
            st.caption(
                "Illustrative scenario using an assumed 8% annual growth rate. "
                "Intermediate values are calculated, not independently predicted."
            )

            # The model predicts the constructed five-year target. Year 0 is
            # calculated by reversing the assumed five-year growth factor.
            current_price = future_price / ((1 + ANNUAL_GROWTH_RATE) ** 5)
            projection_years = list(range(6))
            projection_prices = [
                current_price * ((1 + ANNUAL_GROWTH_RATE) ** year)
                for year in projection_years
            ]

            projection_df = pd.DataFrame(
                {
                    "Year": [f"Year {year}" for year in projection_years],
                    "Estimated Price (₹ lakh)": projection_prices,
                }
            )

            st.line_chart(
                projection_df.set_index("Year"),
                y="Estimated Price (₹ lakh)",
                height=350,
            )

            display_projection = projection_df.copy()
            display_projection["Estimated Price (₹ lakh)"] = display_projection[
                "Estimated Price (₹ lakh)"
            ].round(2)
            st.dataframe(
                display_projection,
                hide_index=True,
                use_container_width=True,
            )

            st.info(
                "The Good_Investment label was constructed using the dataset's "
                "median property price. It does not establish actual investment "
                "profitability."
            )
            st.warning(
                "The five-year target was generated using an assumed 8% annual "
                "growth rate. Treat the output as an experimental model estimate, "
                "not a verified market forecast."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# Model performance dashboard
# --------------------------------------------------

st.divider()
st.header("Model Performance")
st.caption(
    "These metrics come from evaluation of the saved models on 50,000 test rows. "
    "They do not guarantee future performance."
)

classification_report = load_report("classification_report.csv")
confusion_matrix = load_report("confusion_matrix.csv")
regression_report = load_report("regression_metrics.csv")

accuracy = metric_from_classification_report(
    classification_report, "accuracy", "precision"
)
weighted_precision = metric_from_classification_report(
    classification_report, "weighted avg", "precision"
)
weighted_recall = metric_from_classification_report(
    classification_report, "weighted avg", "recall"
)
weighted_f1 = metric_from_classification_report(
    classification_report, "weighted avg", "f1-score"
)

linear_mae = metric_from_regression_report(
    regression_report, "Linear Regression", "MAE"
)
linear_rmse = metric_from_regression_report(
    regression_report, "Linear Regression", "RMSE"
)
linear_r2 = metric_from_regression_report(
    regression_report, "Linear Regression", "R2"
)

classification_col, regression_col = st.columns(2)

with classification_col:
    st.subheader("Classification")
    st.metric("Accuracy", format_percent(accuracy))
    st.metric("Weighted Precision", format_percent(weighted_precision))
    st.metric("Weighted Recall", format_percent(weighted_recall))
    st.metric("Weighted F1-score", format_percent(weighted_f1))

    if confusion_matrix is not None:
        st.write("Confusion Matrix")
        st.dataframe(
            confusion_matrix,
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Rows represent actual classes; columns represent predicted classes."
        )
    else:
        st.info("Confusion matrix report not found in the reports folder.")

    if classification_report is not None:
        st.write("Detailed Classification Report")
        st.dataframe(
            classification_report,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Classification report not found in the reports folder.")

with regression_col:
    st.subheader("Regression")
    st.metric("Mean Absolute Error (MAE)", format_lakh(linear_mae))
    st.metric("Root Mean Squared Error (RMSE)", format_lakh(linear_rmse))
    st.metric(
        "R² Score",
        "Not available" if linear_r2 is None else f"{linear_r2:.5f}",
    )

    if regression_report is not None:
        st.write("Regression Model Comparison")
        st.dataframe(
            regression_report,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Regression metrics report not found in the reports folder.")

st.warning(
    "The current models have limited predictive performance. The classification "
    "target is based on median property price, and the five-year target was "
    "generated using an assumed 8% annual growth rate."
)


# --------------------------------------------------
# Exploratory data analysis dashboard
# --------------------------------------------------

st.divider()
st.header("Property Market Insights")
st.write(
    "Explore property price distribution and compare average prices across "
    "states, property types, and BHK categories."
)

if "Price_in_Lakhs" in df.columns:
    price_data = pd.to_numeric(df["Price_in_Lakhs"], errors="coerce")
    price_data = price_data.replace([np.inf, -np.inf], np.nan).dropna()

    if not price_data.empty:
        st.subheader("Property Price Distribution")

        # Use 15 bins to keep the price-range labels more readable.
        histogram_counts, bin_edges = np.histogram(price_data, bins=15)
        histogram_labels = [
            f"{bin_edges[i]:.0f}–{bin_edges[i + 1]:.0f}"
            for i in range(len(bin_edges) - 1)
        ]
        histogram_data = pd.DataFrame(
            {
                "Price Range (₹ lakh)": histogram_labels,
                "Number of Properties": histogram_counts,
            }
        )

        # Explicit axis encodings prevent Streamlit from treating numeric values
        # as categories. Price ranges are labels; counts are quantitative values.
        histogram_chart = (
            alt.Chart(histogram_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Number of Properties:Q",
                    title="Number of Properties",
                    axis=alt.Axis(format=",d"),
                ),
                y=alt.Y(
                    "Price Range (₹ lakh):N",
                    title="Price Range (₹ lakh)",
                    sort=None,
                    axis=alt.Axis(labelLimit=110),
                ),
                tooltip=[
                    alt.Tooltip("Price Range (₹ lakh):N", title="Price range"),
                    alt.Tooltip("Number of Properties:Q", title="Properties", format=",d"),
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(histogram_chart, use_container_width=True)

        price_stats = st.columns(3)
        price_stats[0].metric("Median Price", f"₹{price_data.median():,.2f} lakh")
        price_stats[1].metric("Mean Price", f"₹{price_data.mean():,.2f} lakh")
        price_stats[2].metric("Maximum Price", f"₹{price_data.max():,.2f} lakh")
    else:
        st.info("No valid property prices were found for the distribution chart.")


if {"State", "Price_in_Lakhs"}.issubset(df.columns):
    st.subheader("Average Price by State")
    state_data = df[["State", "Price_in_Lakhs"]].copy()
    state_data["Price_in_Lakhs"] = pd.to_numeric(
        state_data["Price_in_Lakhs"], errors="coerce"
    )
    state_data = state_data.dropna(subset=["State", "Price_in_Lakhs"])

    state_summary = (
        state_data.groupby("State")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
    )

    if not state_summary.empty:
        max_states = min(20, len(state_summary))
        min_states = min(5, max_states)
        default_states = min(10, max_states)
        number_of_states = st.slider(
            "Number of states to display",
            min_value=min_states,
            max_value=max_states,
            value=default_states,
            key="state_count_slider",
        )

        selected_states = state_summary.head(number_of_states).rename(
            "Average Price (₹ lakh)"
        )
        state_chart_data = selected_states.rename_axis("State").reset_index()
        state_chart = (
            alt.Chart(state_chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Average Price (₹ lakh):Q",
                    title="Average Price (₹ lakh)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=".1f"),
                ),
                y=alt.Y("State:N", title="State", sort="-x"),
                tooltip=[
                    alt.Tooltip("State:N", title="State"),
                    alt.Tooltip("Average Price (₹ lakh):Q", title="Average price", format=".2f"),
                ],
            )
            .properties(height=max(350, number_of_states * 25))
        )
        st.altair_chart(state_chart, use_container_width=True)
    else:
        st.info("No valid state and price data were available to chart.")


if {"Property_Type", "Price_in_Lakhs"}.issubset(df.columns):
    st.subheader("Average Price by Property Type")

    type_data = df[["Property_Type", "Price_in_Lakhs"]].copy()
    type_data["Price_in_Lakhs"] = pd.to_numeric(
        type_data["Price_in_Lakhs"], errors="coerce"
    )
    type_summary = (
        type_data.dropna(subset=["Property_Type", "Price_in_Lakhs"])
        .groupby("Property_Type", as_index=False)["Price_in_Lakhs"]
        .mean()
        .rename(columns={"Price_in_Lakhs": "Average Price (₹ lakh)"})
        .sort_values("Average Price (₹ lakh)", ascending=False)
    )

    if not type_summary.empty:
        type_chart = (
            alt.Chart(type_summary)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Average Price (₹ lakh):Q",
                    title="Average Price (₹ lakh)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=".1f"),
                ),
                y=alt.Y("Property_Type:N", title="Property Type", sort="-x"),
                tooltip=[
                    alt.Tooltip("Property_Type:N", title="Property type"),
                    alt.Tooltip("Average Price (₹ lakh):Q", title="Average price", format=".2f"),
                ],
            )
            .properties(height=max(220, len(type_summary) * 55))
        )
        st.altair_chart(type_chart, use_container_width=True)
    else:
        st.info("No valid property-type data were available to chart.")


if {"BHK", "Price_in_Lakhs"}.issubset(df.columns):
    st.subheader("Average Price by BHK")

    bhk_data = df[["BHK", "Price_in_Lakhs"]].copy()
    bhk_data["Price_in_Lakhs"] = pd.to_numeric(
        bhk_data["Price_in_Lakhs"], errors="coerce"
    )
    bhk_summary = (
        bhk_data.dropna(subset=["BHK", "Price_in_Lakhs"])
        .groupby("BHK", as_index=False)["Price_in_Lakhs"]
        .mean()
        .rename(columns={"Price_in_Lakhs": "Average Price (₹ lakh)"})
        .sort_values("BHK")
    )
    bhk_summary["BHK Label"] = bhk_summary["BHK"].astype(str) + " BHK"

    if not bhk_summary.empty:
        bhk_chart = (
            alt.Chart(bhk_summary)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Average Price (₹ lakh):Q",
                    title="Average Price (₹ lakh)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=".1f"),
                ),
                y=alt.Y("BHK Label:N", title="Bedrooms", sort=None),
                tooltip=[
                    alt.Tooltip("BHK Label:N", title="BHK"),
                    alt.Tooltip("Average Price (₹ lakh):Q", title="Average price", format=".2f"),
                ],
            )
            .properties(height=max(250, len(bhk_summary) * 55))
        )
        st.altair_chart(bhk_chart, use_container_width=True)
    else:
        st.info("No valid BHK and price data were available to chart.")


# --------------------------------------------------
# Project disclaimer
# --------------------------------------------------

st.divider()
st.caption(
    "Educational project only. Model outputs are experimental estimates, "
    "not guaranteed returns or professional financial advice."
)
