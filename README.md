# Real Estate Investment Advisor: Predicting Property Profitability & Future Value

## 1. Project Overview

The Real Estate Investment Advisor is a machine learning project that explores property investment indicators and estimates potential property values over a five-year period. It combines data preprocessing, exploratory data analysis (EDA), classification, regression, experiment tracking with MLflow, and an interactive Streamlit application.

**Important:** The outputs are exploratory estimates based on constructed target variables. They should not be interpreted as reliable financial advice or guarantees of investment profitability.

## 2. Key Features

- Housing dataset preprocessing and data-quality investigation.
- Exploratory analysis of property prices, sizes, locations, property types, BHK configurations, furnishing, ownership, parking, amenities, and transport accessibility.
- Classification model for the `Good_Investment` target.
- Regression model for the `Future_Price_5Y` target.
- Evaluation using accuracy, precision, recall, F1-score, confusion matrix, MAE, RMSE, and R².
- MLflow experiment tracking and model evaluation logging.
- Streamlit interface for property inputs, predictions, model performance, and market insights.
- CSV summaries, figures, and written EDA findings.

## 3. Dataset

The project uses a housing dataset with 250,000 records.

Expected local files:

- `data/raw/india_housing_prices.csv`
- `data/processed/india_housing_prices_preprocessed.csv`
- `data/processed/india_housing_prices_targets.csv`

The dataset's original provenance and the precise definitions of some fields still require confirmation. These files are excluded from Git because of their size. Obtain the dataset through the project's authorized source and place the files at the paths above before running the application.

## 4. Methodology

### Data Preparation

- Inspected the dataset and checked missing values and duplicate records.
- Investigated inconsistent floor information and redundant property-age fields.
- Created target variables for classification and five-year price scenarios.
- Generated processed data for analysis and modeling.

### Exploratory Data Analysis

The EDA script generates charts and CSV summaries under `reports/` and `reports/figures/`. The analysis covers price distributions, property characteristics, geographic groupings, numerical relationships, amenities, furnishing, ownership, parking, and accessibility.

See `EDA_Findings.md` for the detailed findings.

### Machine Learning

Two saved pipelines are used:

- **Classification:** predicts `Good_Investment`.
- **Regression:** estimates `Future_Price_5Y`.

The pipelines include numeric and categorical preprocessing. Saved models are expected at:

- `models/investment_classifier.pkl`
- `models/price_regressor.pkl`

These model files are small enough to include in the repository. The application can load them without retraining.

## 5. Current Evaluation Results

Results from the current saved models:

| Metric | Classification | Regression |
|---|---:|---:|
| Accuracy | 0.4984 | — |
| F1-score | Approximately 0.50 | — |
| MAE | — | 180.82 lakh |
| RMSE | — | 208.83 lakh |
| R² | — | -0.0048 |

The classification model performs close to chance on the current target. The regression model performs slightly worse than a simple mean-prediction baseline. These results indicate that the current features and constructed targets do not provide useful predictive performance.

The results are reported honestly; high predictive accuracy has not been established.

## 6. Important Data Limitations

1. `Good_Investment` is constructed using the dataset's median property price; it does not represent observed investment returns.
2. `Future_Price_5Y` is calculated using an assumed annual appreciation rate of 8%: `Price_in_Lakhs × (1.08 ** 5)`. It is a scenario calculation, not an observed future sale price.
3. The Pearson correlation between property size and price is approximately -0.0025.
4. In 116,304 records (46.52%), `Floor_No` exceeds `Total_Floors`. These records require investigation rather than automatic correction or deletion.
5. `Year_Built + Age_of_Property = 2025` for every record, making these fields redundant.
6. The original `Price_per_SqFt` field may not represent the conventional INR-per-square-foot measure.
7. Fields mentioned in the project requirements, including `Crime_Rate`, `Infrastructure_Score`, and `School_Density_Score`, are not present in the current dataset.

These limitations restrict the real-world usefulness of the current predictions.

## 7. Project Structure


Real_Estate_Investment_advisor/
├── app.py
├── preprocess_data.py
├── create_targets.py
├── eda_analysis.py
├── train_models.py
├── log_existing_models.py
├── check_data.py
├── check_saved_models.py
├── investigate_data_quality.py
├── inspect_data.py
├── verify_preprocessed.py
├── verify_targets.py
├── requirements.txt
├── README.md
├── EDA_Findings.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   └── figures/
├── notebooks/
├── src/
├── mlflow.db
└── mlruns/


Local datasets, Python virtual environments, and MLflow tracking artifacts are excluded from Git through `.gitignore`.

## 8. Setup and Run

Use Python 3.11 for the existing MLflow environment.

### Create and activate an environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that `requirements.txt` contains the dependencies needed by the application, preprocessing, model evaluation, and MLflow scripts.

### Run the Streamlit application

After placing the required dataset files and saved models at their expected paths:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL, typically `http://localhost:8501`.

### Run EDA again

```powershell
python eda_analysis.py
```

This regenerates the EDA reports and charts.

### View MLflow experiments

If MLflow tracking data has been generated locally, start the tracking UI with the appropriate backend configuration used by the project. For the existing local SQLite setup, the tracking database is `mlflow.db`, and the tracking artifacts are stored under `mlruns/`.

The previously configured MLflow UI was available at `http://127.0.0.1:5000`.

## 9. Reproducibility Notes

- The saved models can be loaded without retraining.
- The dataset must be obtained separately because it is excluded from Git.
- MLflow history is local and is not included in the repository.
- Model metrics may vary if models are retrained with different dependencies, data, or random seeds.
- Confirm the contents of `requirements.txt` before sharing the repository.

## 10. Disclaimer

This project is for educational and exploratory purposes. Its investment labels and future-price estimates are derived from assumptions in the dataset. Predictions do not establish real-world profitability, guarantee future appreciation, or constitute financial advice.
