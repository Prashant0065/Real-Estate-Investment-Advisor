from pathlib import Path
import joblib

model_dir = Path("models")

classifier_path = model_dir / "investment_classifier.pkl"
regressor_path = model_dir / "price_regressor.pkl"

classifier = joblib.load(classifier_path)
regressor = joblib.load(regressor_path)

print("Saved models loaded successfully!")
print("Classifier type:", type(classifier).__name__)
print("Regressor type:", type(regressor).__name__)
print("Classifier steps:", list(classifier.named_steps.keys()))
print("Regressor steps:", list(regressor.named_steps.keys()))
