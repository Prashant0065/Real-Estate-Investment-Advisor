
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "india_housing_prices_targets.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

CLASSIFIER_PATH = MODEL_DIR / "investment_classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "price_regressor.pkl"

TRACKING_URI = f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "Real_Estate_Investment_Advisor"

RANDOM_STATE = 42
TEST_SIZE = 0.20

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


def main():
    for path in [DATA_PATH, CLASSIFIER_PATH, REGRESSOR_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=EXCLUDED_COLUMNS)
    y_class = df[TARGET_CLASSIFICATION]
    y_reg = df[TARGET_REGRESSION]

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
        train_test_split(
            X,
            y_class,
            y_reg,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_class,
        )
    )

    print("Loading saved models without retraining...")
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Classification evaluation
    class_predictions = classifier.predict(X_test)

    class_report = classification_report(
        y_class_test,
        class_predictions,
        output_dict=True,
        zero_division=0,
    )

    class_accuracy = accuracy_score(y_class_test, class_predictions)
    class_precision = class_report["weighted avg"]["precision"]
    class_recall = class_report["weighted avg"]["recall"]
    class_f1 = class_report["weighted avg"]["f1-score"]

    class_report_path = REPORT_DIR / "classification_report.csv"
    confusion_path = REPORT_DIR / "confusion_matrix.csv"

    pd.DataFrame(class_report).transpose().to_csv(class_report_path)

    pd.DataFrame(
        confusion_matrix(y_class_test, class_predictions),
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(confusion_path)

    with mlflow.start_run(run_name="Saved_Logistic_Regression_Classifier"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("evaluation_mode", "existing_model_no_retraining")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X.shape[1])

        mlflow.log_metric("accuracy", float(class_accuracy))
        mlflow.log_metric("weighted_precision", float(class_precision))
        mlflow.log_metric("weighted_recall", float(class_recall))
        mlflow.log_metric("weighted_f1_score", float(class_f1))

        mlflow.log_artifact(str(class_report_path))
        mlflow.log_artifact(str(confusion_path))
        signature = infer_signature(X_test, class_predictions)
        mlflow.sklearn.log_model(
            sk_model=classifier,
            name="investment_classifier",
            signature=signature,
            input_example=X_test.head(5),
            skops_trusted_types=["numpy.dtype"],
        )


    print(f"Classification accuracy: {class_accuracy:.4f}")

    # Regression evaluation
    reg_predictions = regressor.predict(X_test)

    reg_mae = mean_absolute_error(y_reg_test, reg_predictions)
    reg_rmse = np.sqrt(mean_squared_error(y_reg_test, reg_predictions))
    reg_r2 = r2_score(y_reg_test, reg_predictions)

    regression_report_path = REPORT_DIR / "existing_model_regression_metrics.csv"

    pd.DataFrame(
        [{
            "Model": "Linear Regression",
            "MAE": reg_mae,
            "RMSE": reg_rmse,
            "R2": reg_r2,
        }]
    ).to_csv(regression_report_path, index=False)

    with mlflow.start_run(run_name="Saved_Linear_Regression_Regressor"):
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("evaluation_mode", "existing_model_no_retraining")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X.shape[1])

        mlflow.log_metric("mae_lakhs", float(reg_mae))
        mlflow.log_metric("rmse_lakhs", float(reg_rmse))
        mlflow.log_metric("r2_score", float(reg_r2))

        mlflow.log_artifact(str(regression_report_path))
        signature = infer_signature(X_test, reg_predictions)
        mlflow.sklearn.log_model(
            sk_model=regressor,
            name="price_regressor",
            signature=signature,
            input_example=X_test.head(5),
            skops_trusted_types=["numpy.dtype"],
        )

    print(f"Regression MAE:  {reg_mae:.2f} lakh")
    print(f"Regression RMSE: {reg_rmse:.2f} lakh")
    print(f"Regression R2:   {reg_r2:.4f}")

    print("\nSUCCESS: Both evaluation runs have been logged to MLflow.")
    print(f"Tracking database: {PROJECT_DIR / 'mlflow.db'}")


if __name__ == "__main__":
    main()
