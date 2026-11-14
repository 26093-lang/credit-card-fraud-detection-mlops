# Credit Card Fraud Detection
# Phase 2 - Random Forest
# Faster version for large dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("data/creditcard.csv")

print("======================================")
print("CREDIT CARD FRAUD DETECTION")
print("PHASE 2 - RANDOM FOREST")
print("======================================")

print("\nDataset Shape:", data.shape)


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = data.drop("Class", axis=1)
y = data["Class"]


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=30,
    max_depth=12,
    max_samples=0.7,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\n======================================")
print("TRAINING RANDOM FOREST")
print("======================================")

print("\nPlease wait...")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed successfully.")


# ==========================================
# PREDICTIONS
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# EVALUATION
# ==========================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n======================================")
print("RANDOM FOREST EVALUATION")
print("======================================")

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png"
)

plt.close()

print("\nConfusion matrix saved.")


# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Random Forest - ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "random_forest_roc_curve.png"
)

plt.close()

print("ROC curve saved.")


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)


plt.figure(figsize=(10, 6))

importance.head(10).plot(
    kind="bar"
)

plt.title(
    "Top 10 Important Features - Random Forest"
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig(
    "random_forest_feature_importance.png"
)

plt.close()

print("Feature importance graph saved.")


# ==========================================
# SAVE RANDOM FOREST MODEL
# ==========================================

joblib.dump(
    model,
    "random_forest_model.pkl"
)

print("\nRandom Forest model saved as:")
print("random_forest_model.pkl")


# ==========================================
# COMPLETION
# ==========================================

print("\n======================================")
print("RANDOM FOREST PHASE COMPLETED")
print("======================================")

print("\nGenerated files:")

print("1. random_forest_model.pkl")
print("2. random_forest_confusion_matrix.png")
print("3. random_forest_roc_curve.png")
print("4. random_forest_feature_importance.png")

print("\n======================================")
print("PHASE 2 RANDOM FOREST COMPLETED")
print("======================================")