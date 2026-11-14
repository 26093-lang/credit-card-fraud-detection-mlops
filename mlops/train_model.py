# ============================================================
# CREDIT CARD FRAUD DETECTION - MLOPS TRAINING PIPELINE
# ============================================================

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

import hashlib
import json

from pathlib import Path
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "v1"

MODEL_PATH = MODEL_DIR / f"fraud_pipeline_{MODEL_VERSION}.joblib"

METADATA_PATH = MODEL_DIR / f"model_metadata_{MODEL_VERSION}.json"

SCORE_PATH = REPORT_DIR / f"prediction_scores_{MODEL_VERSION}.csv"

EXPERIMENT_NAME = "Credit_Card_Fraud_Detection"

# SQLite database for MLflow tracking
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


# ============================================================
# 2. DATASET VERSION / FINGERPRINT
# ============================================================

def calculate_dataset_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():

    print("\nLoading credit card dataset...")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    if "Class" not in data.columns:

        raise ValueError(
            "Dataset must contain a Class column."
        )

    if data["Class"].isna().any():

        raise ValueError(
            "Dataset contains missing target labels."
        )

    if set(data["Class"].unique()) != {0, 1}:

        raise ValueError(
            "Expected binary labels: 0 and 1."
        )

    if data.isna().any().any():

        raise ValueError(
            "Dataset contains missing values. "
            "Handle them before training."
        )

    dataset_hash = calculate_dataset_hash(DATA_PATH)

    print("\nDataset loaded successfully!")

    print("Dataset shape:", data.shape)

    print("\nClass distribution:")

    print(data["Class"].value_counts())

    fraud_percentage = (
        data["Class"].eq(1).mean() * 100
    )

    print("\nFraud percentage:")
    print(f"{fraud_percentage:.4f}%")

    return data, dataset_hash


# ============================================================
# 4. PREPROCESSING AND TRAIN-TEST SPLIT
# ============================================================

def prepare_data(data):

    X = data.drop(columns=["Class"])

    y = data["Class"]

    # Scale Time and Amount.
    # Preserve all other features.

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "scale",
                StandardScaler(),
                ["Time", "Amount"]
            )

        ],

        remainder="passthrough"

    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# 5. CREATE RANDOM FOREST PIPELINE
# ============================================================

def create_model(preprocessor):

    model = Pipeline(

        steps=[

            (
                "preprocessing",
                preprocessor
            ),

            (
                "classifier",

                RandomForestClassifier(

                    n_estimators=200,

                    max_depth=None,

                    class_weight="balanced_subsample",

                    random_state=42,

                    n_jobs=-1

                )
            )

        ]

    )

    return model


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(

        y_test,
        predictions,
        zero_division=0

    )

    recall = recall_score(

        y_test,
        predictions,
        zero_division=0

    )

    f1 = f1_score(

        y_test,
        predictions,
        zero_division=0

    )

    roc_auc = roc_auc_score(

        y_test,
        probabilities

    )

    pr_auc = average_precision_score(

        y_test,
        probabilities

    )

    metrics = {

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc),

        "pr_auc": float(pr_auc)

    }

    print("\n========== MODEL EVALUATION ==========")

    for name, value in metrics.items():

        print(f"{name}: {value:.4f}")

    return metrics, predictions, probabilities


# ============================================================
# 7. SAVE PREDICTION SCORE DISTRIBUTION
# ============================================================

def save_prediction_scores(
    y_test,
    predictions,
    probabilities
):

    score_data = pd.DataFrame({

        "actual_class": y_test.to_numpy(),

        "predicted_class": predictions,

        "fraud_probability": probabilities

    })

    score_data.to_csv(

        SCORE_PATH,

        index=False

    )

    print("\nPrediction score distribution saved:")

    print(SCORE_PATH)

    print("\nPrediction score summary:")

    print(
        score_data["fraud_probability"].describe()
    )

    return score_data


# ============================================================
# 8. MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("\n====================================")

    print("CREDIT CARD FRAUD DETECTION - MLOPS")

    print("====================================")

    # Load dataset

    data, dataset_hash = load_dataset()

    # Prepare data

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(data)

    print("\nTraining samples:", len(X_train))

    print("Testing samples:", len(X_test))

    # Create pipeline

    model = create_model(preprocessor)

    print("\nTraining Random Forest model...")

    model.fit(X_train, y_train)

    print("\nModel training completed!")

    # Evaluate model

    (
        metrics,
        predictions,
        probabilities
    ) = evaluate_model(

        model,
        X_test,
        y_test

    )

    # Save prediction score distribution

    score_data = save_prediction_scores(

        y_test,
        predictions,
        probabilities

    )

    # Save complete preprocessing + model pipeline

    joblib.dump(

        model,
        MODEL_PATH

    )

    print("\nComplete model pipeline saved:")

    print(MODEL_PATH)

    # Create model metadata

    metadata = {

        "model_name": "Random Forest",

        "model_version": MODEL_VERSION,

        "dataset_file": DATA_PATH.name,

        "dataset_sha256": dataset_hash,

        "training_samples": int(len(X_train)),

        "testing_samples": int(len(X_test)),

        "fraud_percentage": float(
            data["Class"].eq(1).mean() * 100
        ),

        "metrics": metrics,

        "created_at": datetime.now(
            timezone.utc
        ).isoformat()

    }

    with open(METADATA_PATH, "w") as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nModel metadata saved!")

    # ========================================================
    # 9. MLFLOW SQLITE TRACKING CONFIGURATION
    # ========================================================

    print("\nConnecting to MLflow SQLite database...")

    tracking_uri = (
        "sqlite:///"
        + MLFLOW_DB_PATH.resolve().as_posix()
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ========================================================
    # 10. LOG EXPERIMENT
    # ========================================================

    with mlflow.start_run():

        # Log parameters

        mlflow.log_param(
            "model_name",
            "Random Forest"
        )

        mlflow.log_param(
            "model_version",
            MODEL_VERSION
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "class_weight",
            "balanced_subsample"
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "dataset_file",
            DATA_PATH.name
        )

        # Dataset fingerprint

        mlflow.log_param(
            "dataset_sha256",
            dataset_hash
        )

        # Log evaluation metrics

        mlflow.log_metrics(metrics)

        # Add experiment tags

        mlflow.set_tag(
            "model_version",
            MODEL_VERSION
        )

        mlflow.set_tag(
            "dataset_hash",
            dataset_hash
        )

        mlflow.set_tag(
            "experiment_type",
            "Fraud Detection"
        )

        # Log complete model pipeline

        mlflow.sklearn.log_model(
            sk_model=model,
            name="fraud_detection_pipeline",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Log model metadata

        mlflow.log_artifact(

            str(METADATA_PATH),

            artifact_path="model_metadata"

        )

        # Log model file

        mlflow.log_artifact(

            str(MODEL_PATH),

            artifact_path="versioned_models"

        )

        # Log prediction score distribution

        mlflow.log_artifact(

            str(SCORE_PATH),

            artifact_path="prediction_scores"

        )

        print("\nMLflow experiment logged successfully!")

    # ========================================================
    # 11. FINAL RESULT
    # ========================================================

    print("\n====================================")

    print("MLOPS TRAINING COMPLETED SUCCESSFULLY")

    print("====================================")

    print("\nGenerated files:")

    print("1. Versioned model pipeline")

    print("2. Model metadata")

    print("3. Prediction score distribution")

    print("4. MLflow SQLite tracking database")

    print("\nAll five evaluation metrics recorded.")

    print("\n====================================")


# ============================================================
# 12. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()