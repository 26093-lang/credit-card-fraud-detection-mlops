# Credit Card Fraud Detection – MLOps Project

## 1. Project Overview

This project implements a Machine Learning-based Credit Card Fraud Detection System using a highly imbalanced transaction dataset.

The application predicts whether a credit card transaction is Normal or Fraudulent. It also demonstrates MLOps practices such as model training, experiment tracking, model versioning, monitoring, automated testing, and Docker deployment.

## 2. Objectives

- Detect fraudulent credit card transactions.
- Handle highly imbalanced datasets.
- Train and evaluate machine learning models.
- Track experiments using MLflow.
- Maintain versioned machine learning models.
- Monitor prediction results.
- Deploy the application using Docker.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- MLflow
- Joblib
- Docker
- Git and GitHub

## 4. Dataset

Dataset: Credit Card Fraud Detection Dataset

Total transactions: 284,807

Normal transactions: 284,315

Fraudulent transactions: 492

Fraud percentage: 0.1727%

The dataset contains 30 input features and one target column named Class.

## 5. Project Features

### Machine Learning
- Data preprocessing and scaling
- Handling class imbalance
- Random Forest classification
- Model evaluation
- Confusion matrix
- ROC curve
- Feature importance

### MLOps
- MLflow experiment tracking
- Model versioning
- Model metadata
- Prediction score distribution
- Application logging
- Prediction monitoring

### Web Application
- Transaction prediction
- Class distribution visualization
- Model performance visualization
- Prediction history
- Model monitoring dashboard

### Deployment and Testing
- Docker image creation
- Docker container deployment
- Automated model tests

## 6. Project Structure

```text
WelcomeScreen/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_pipeline_v1.joblib
│   └── model_metadata_v1.json
│
├── mlops/
│   └── train_model.py
│
├── reports/
│   └── prediction_scores_v1.csv
│
├── logs/
│
├── app.py
├── test_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── mlflow.db
```

## 7. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 8. Run the Application

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the application:

http://localhost:8501

## 9. Run Automated Tests

```bash
python test_model.py
```

## 10. MLflow Tracking

Start the MLflow dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

http://127.0.0.1:5000

## 11. Docker Deployment

Build the Docker image:

```bash
docker build -t fraud-detection-app .
```

Run the Docker container:

```bash
docker run -d -p 8502:8501 --name fraud-detection-container fraud-detection-app
```

Open:

http://localhost:8502

## 12. Conclusion

The project demonstrates an end-to-end machine learning workflow for credit card fraud detection. It integrates machine learning, experiment tracking, model versioning, monitoring, automated testing, and containerization to demonstrate practical MLOps concepts.
```

**Important:** This is a documentation draft. The project structure is based on the files we've created; adjust any filenames if yours differ.

Once pasted and saved, tell me **Done**. Then we'll move to Git and GitHub.