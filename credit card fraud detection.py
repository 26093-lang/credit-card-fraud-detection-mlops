import pandas as pd

# Load the dataset
data = pd.read_csv("data/creditcard.csv")

# Display basic information
print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())

print("\nClass distribution:")
print(data["Class"].value_counts())

print("\nClass percentage:")
print(data["Class"].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot class distribution
plt.figure(figsize=(6, 4))

sns.countplot(x="Class", data=data)

plt.title("Class Distribution")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig("class_distribution.png")
plt.close()

# Separate input features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")
print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)
# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model training completed.")
# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

print("\nPredictions completed.")
print("Number of predictions:", len(y_pred))
# Evaluate the model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Get probability of the transaction being fraudulent
y_probability = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

print("\n--- Model Evaluation ---")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)
# Classification Report
from sklearn.metrics import classification_report, confusion_matrix

print("\n--- Classification Report ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Fraud"]
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)
# Plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig("confusion_matrix.png")
plt.close()

print("\nConfusion matrix graph saved successfully.")
# Plot ROC Curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_probability)

plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig("roc_curve.png")
plt.close()

print("\nROC curve graph saved successfully.")
# Save the trained model and scaler
import joblib

joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")