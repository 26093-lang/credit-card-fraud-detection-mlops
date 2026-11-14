import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
MODEL_FILE = BASE_DIR / "models" / "fraud_pipeline_v1.joblib"
DATA_FILE = BASE_DIR / "data" / "creditcard.csv"

model = joblib.load(MODEL_FILE)
data = pd.read_csv(DATA_FILE)

# Test 1: Model loading
def test_model_loads():
    assert model is not None
    print("PASS: Model loaded successfully")

# Test 2: Dataset loading
def test_dataset_loads():
    assert not data.empty
    assert "Class" in data.columns
    print("PASS: Dataset loaded successfully")

# Test 3: Prediction output
def test_prediction_output():
    features = data.drop(columns=["Class"]).iloc[[12]]
    prediction = model.predict(features)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
    print("PASS: Model prediction is valid")

# Test 4: Fraud probability
def test_fraud_probability():
    features = data.drop(columns=["Class"]).iloc[[12]]
    probability = model.predict_proba(features)
    assert probability.shape == (1, 2)
    assert 0 <= probability[0][1] <= 1
    print("PASS: Fraud probability is valid")

# Test 5: Imbalanced dataset
def test_imbalanced_dataset():
    counts = data["Class"].value_counts()
    assert counts[0] > counts[1]
    print("PASS: Class imbalance verified")

# Run all tests
if __name__ == "__main__":
    test_model_loads()
    test_dataset_loads()
    test_prediction_output()
    test_fraud_probability()
    test_imbalanced_dataset()

    print("\nALL TESTS COMPLETED SUCCESSFULLY!")