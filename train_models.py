
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "india_housing_prices_targets.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

CLASSIFIER_PATH = MODEL_DIR / "investment_classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "price_regressor.pkl"
RANDOM_FOREST_PATH = MODEL_DIR / "random_forest_regressor.pkl"

TEST_SIZE = 0.20
RANDOM_STATE = 42

TARGET_CLASSIFICATION = "Good_Investment"
TARGET_REGRESSION = "Future_Price_5Y"

EXCLUDED_COLUMNS = [
    "ID",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Calculated_Price_per_SqFt",
    "Price_per_SqFt_Mismatch",
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
]


# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        EXCLUDED_COLUMNS
        + ["State", "City", "Property_Type"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df[TARGET_CLASSIFICATION].isnull().any():
        raise ValueError(
            "Good_Investment contains missing values."
        )

    if df[TARGET_REGRESSION].isnull().any():
        raise ValueError(
            "Future_Price_5Y contains missing values."
        )

    print("\nDataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")

    print("\nMissing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))

    print("\nClassification target distribution:")
    print(df[TARGET_CLASSIFICATION].value_counts().sort_index())

    return df


# --------------------------------------------------
# 3. Prepare features and split the dataset
# --------------------------------------------------

def prepare_data(df):
    X = df.drop(columns=EXCLUDED_COLUMNS)
    y_classification = df[TARGET_CLASSIFICATION]
    y_regression = df[TARGET_REGRESSION]

    (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
    ) = train_test_split(
        X,
        y_classification,
        y_regression,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_classification,
    )

    print("\nTrain-test split completed.")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Input features: {X.shape[1]}")

    return (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
    )


# --------------------------------------------------
# 4. Create preprocessing pipeline
# --------------------------------------------------

def create_preprocessor(X_train):
    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object", "category", "str", "string"]
        )
        .columns
        .tolist()
    )

    numeric_columns = (
        X_train
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ])

    print("\nPreprocessing configured.")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    return preprocessor


# --------------------------------------------------
# 5. Evaluation helpers
# --------------------------------------------------

def evaluate_classification(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nCLASSIFICATION RESULTS")
    print("-" * 45)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return predictions


def evaluate_regression(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name} RESULTS")
    print("-" * 45)
    print(f"MAE:  {mae:.2f} lakh")
    print(f"RMSE: {rmse:.2f} lakh")
    print(f"R2:   {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# --------------------------------------------------
# 6. Train or load the classification model
# --------------------------------------------------

def get_classifier(
    X_train,
    y_train,
    preprocessor,
    load_models,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if load_models:
        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(
                "Saved classifier not found. "
                "Run training without --load-models first."
            )

        print("\nLoading saved classification model...")
        return joblib.load(CLASSIFIER_PATH)

    print("\nTraining Logistic Regression...")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, CLASSIFIER_PATH)

    print(f"Classifier saved: {CLASSIFIER_PATH}")

    return model


# --------------------------------------------------
# 7. Train or load the regression model
# --------------------------------------------------

def get_regressor(
    X_train,
    y_train,
    preprocessor,
    load_models,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if load_models:
        if not REGRESSOR_PATH.exists():
            raise FileNotFoundError(
                "Saved regressor not found. "
                "Run training without --load-models first."
            )

        print("\nLoading saved regression model...")
        return joblib.load(REGRESSOR_PATH)

    print("\nTraining Linear Regression...")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, REGRESSOR_PATH)

    print(f"Regressor saved: {REGRESSOR_PATH}")

    return model


# --------------------------------------------------
# 8. Optional Random Forest experiment
# --------------------------------------------------

def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
):
    print("\nTraining Random Forest Regressor...")
    print("This optional experiment may take several minutes.")

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=30,
                max_depth=8,
                min_samples_leaf=10,
                max_features=0.5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    metrics = evaluate_regression(
        model,
        X_test,
        y_test,
        "Random Forest Regressor",
    )

    joblib.dump(model, RANDOM_FOREST_PATH)

    print(f"Random Forest saved: {RANDOM_FOREST_PATH}")

    return metrics


# --------------------------------------------------
# 9. Save evaluation reports
# --------------------------------------------------

def save_reports(classification_model, regression_model,
                 X_test, y_class_test, y_reg_test,
                 regression_metrics):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    class_predictions = classification_model.predict(X_test)

    report = classification_report(
        y_class_test,
        class_predictions,
        zero_division=0,
        output_dict=True,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    pd.DataFrame(
        confusion_matrix(y_class_test, class_predictions),
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )

    pd.DataFrame(regression_metrics).to_csv(
        REPORT_DIR / "regression_metrics.csv",
        index=False,
    )

    print("\nEvaluation reports saved in reports/.")


# --------------------------------------------------
# 10. Main program
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate real estate ML models."
    )

    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Load saved models instead of retraining them.",
    )

    parser.add_argument(
        "--train-random-forest",
        action="store_true",
        help="Run the optional, slower Random Forest experiment.",
    )

    args = parser.parse_args()

    df = load_data()

    (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
    ) = prepare_data(df)

    # Separate preprocessors prevent fitted preprocessing
    # state from being shared between different model pipelines.
    classifier_preprocessor = create_preprocessor(X_train)
    regressor_preprocessor = create_preprocessor(X_train)

    classification_model = get_classifier(
        X_train,
        y_class_train,
        classifier_preprocessor,
        args.load_models,
    )

    evaluate_classification(
        classification_model,
        X_test,
        y_class_test,
    )

    regression_metrics = []

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_reg_train)

    regression_metrics.append(
        evaluate_regression(
            baseline,
            X_test,
            y_reg_test,
            "Mean Baseline",
        )
    )

    regression_model = get_regressor(
        X_train,
        y_reg_train,
        regressor_preprocessor,
        args.load_models,
    )

    regression_metrics.append(
        evaluate_regression(
            regression_model,
            X_test,
            y_reg_test,
            "Linear Regression",
        )
    )

    if args.train_random_forest:
        rf_preprocessor = create_preprocessor(X_train)

        rf_metrics = train_random_forest(
            X_train,
            y_reg_train,
            X_test,
            y_reg_test,
            rf_preprocessor,
        )

        regression_metrics.append(rf_metrics)

    save_reports(
        classification_model,
        regression_model,
        X_test,
        y_class_test,
        y_reg_test,
        regression_metrics,
    )

    print("\nAll requested tasks completed.")
    print("Saved classifier:", CLASSIFIER_PATH)
    print("Saved regressor:", REGRESSOR_PATH)


if __name__ == "__main__":
    main()
