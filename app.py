import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "fraud_pipeline_v1.joblib"
DATA_FILE = BASE_DIR / "data" / "creditcard.csv"
RECORD_FILE = BASE_DIR / "prediction_records.csv"

MODEL_VERSION = "v1"

# ==========================================
# LOGGING CONFIGURATION
# ==========================================

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "This application uses a versioned Random Forest "
    "machine learning pipeline to detect fraudulent transactions."
)

st.caption(f"Active Model Version: {MODEL_VERSION}")

st.divider()

# ==========================================
# LOAD MODEL
# ==========================================

if not MODEL_FILE.exists():
    st.error(f"Model pipeline not found: {MODEL_FILE}")
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
    st.success(f"Model pipeline {MODEL_VERSION} loaded successfully.")

except Exception as error:
    logger.exception("Model loading failed")
    st.error(f"Model loading error: {error}")
    st.stop()

# ==========================================
# LOAD DATASET
# ==========================================

if not DATA_FILE.exists():
    st.error("Dataset not found in the data folder.")
    st.stop()

try:
    data = pd.read_csv(DATA_FILE)

except Exception as error:
    logger.exception("Dataset loading failed")
    st.error(f"Dataset loading error: {error}")
    st.stop()

feature_columns = [
    column for column in data.columns
    if column != "Class"
]

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Transaction Prediction",
        "Class Distribution",
        "Model Performance",
        "Prediction Records",
        "Model Monitoring"
    ]
)

# ==========================================
# TRANSACTION PREDICTION
# ==========================================

if page == "Transaction Prediction":

    st.header("Transaction Prediction")

    st.write("Enter the transaction details below.")

    st.subheader("Sample Transaction")

    sample_index = st.number_input(
        "Dataset row number",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1
    )

    sample_transaction = data.iloc[int(sample_index)]

    input_values = {}

    st.write("Transaction Features")

    columns = st.columns(3)

    for i, feature in enumerate(feature_columns):

        with columns[i % 3]:

            default_value = float(sample_transaction[feature])

            input_values[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.6f"
            )

    if st.button("🔍 Predict Transaction", type="primary"):

        try:
            input_data = pd.DataFrame(
                [input_values],
                columns=feature_columns
            )

            prediction = int(model.predict(input_data)[0])

            probability = float(
                model.predict_proba(input_data)[0][1]
            )

            st.divider()

            if prediction == 1:
                st.error("🚨 FRAUDULENT TRANSACTION")
            else:
                st.success("✅ NORMAL TRANSACTION")

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            # Save prediction record

            record = input_data.copy()

            record["Prediction"] = (
                "Fraudulent" if prediction == 1 else "Normal"
            )

            record["Fraud_Probability"] = probability

            record["Model_Version"] = MODEL_VERSION

            record["Timestamp"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            if RECORD_FILE.exists():

                old_records = pd.read_csv(RECORD_FILE)

                updated_records = pd.concat(
                    [old_records, record],
                    ignore_index=True
                )

            else:
                updated_records = record

            updated_records.to_csv(
                RECORD_FILE,
                index=False
            )

            logger.info(
                "Prediction completed | Model: %s | Result: %s | Fraud probability: %.6f",
                MODEL_VERSION,
                record["Prediction"].iloc[0],
                probability
            )

            st.success(
                f"Prediction saved successfully using model {MODEL_VERSION}."
            )

        except Exception as error:

            logger.exception("Prediction failed")
            st.error(f"Prediction error: {error}")

# ==========================================
# CLASS DISTRIBUTION
# ==========================================

elif page == "Class Distribution":

    st.header("Class Distribution")

    class_counts = data["Class"].value_counts()

    class_labels = ["Normal", "Fraudulent"]

    values = [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(class_labels, values)

    ax.set_title("Credit Card Transaction Class Distribution")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Number of Transactions")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Class Counts")

    result = pd.DataFrame({
        "Class": class_labels,
        "Count": values
    })

    st.dataframe(
        result,
        use_container_width=True
    )

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header("Random Forest Model Performance")

    st.write(f"Active Model Version: {MODEL_VERSION}")

    image_files = {
        "Confusion Matrix": "random_forest_confusion_matrix.png",
        "ROC Curve": "random_forest_roc_curve.png",
        "Feature Importance": "random_forest_feature_importance.png"
    }

    for title, filename in image_files.items():

        st.subheader(title)

        image_path = BASE_DIR / filename

        if image_path.exists():

            st.image(
                str(image_path),
                use_container_width=True
            )

        else:
            st.warning(f"{filename} not found.")

# ==========================================
# PREDICTION RECORDS
# ==========================================

elif page == "Prediction Records":

    st.header("Prediction Records")

    if RECORD_FILE.exists():

        records = pd.read_csv(RECORD_FILE)

        st.write(f"Total predictions: {len(records)}")

        st.dataframe(
            records,
            use_container_width=True
        )

        st.download_button(
            "Download Prediction Records",
            records.to_csv(index=False),
            "prediction_records.csv",
            "text/csv"
        )

    else:

        st.info("No prediction records available yet.")

# ==========================================
# MODEL MONITORING
# ==========================================

elif page == "Model Monitoring":

    st.header("Model Monitoring")

    st.write(f"Active Model Version: {MODEL_VERSION}")

    if RECORD_FILE.exists():

        records = pd.read_csv(RECORD_FILE)

        total = len(records)

        fraud_count = (
            records["Prediction"] == "Fraudulent"
        ).sum()

        normal_count = (
            records["Prediction"] == "Normal"
        ).sum()

        fraud_rate = (
            fraud_count / total * 100
            if total > 0 else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", total)

        col2.metric("Fraud Predictions", int(fraud_count))

        col3.metric(
            "Predicted Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

        st.subheader("Prediction Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            ["Normal", "Fraudulent"],
            [normal_count, fraud_count]
        )

        ax.set_ylabel("Number of Predictions")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Fraud Probability Distribution")

        if "Fraud_Probability" in records.columns:

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.hist(
                records["Fraud_Probability"],
                bins=20
            )

            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Number of Predictions")
            ax.set_title("Fraud Probability Distribution")

            st.pyplot(fig)
            plt.close(fig)

    else:

        st.info(
            "Make some predictions first to generate monitoring data."
        )