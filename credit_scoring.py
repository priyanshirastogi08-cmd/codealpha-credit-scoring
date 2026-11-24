# ==========================================
# CodeAlpha Machine Learning Internship
# Task 1: Credit Scoring Model (VS Code Script)
# ==========================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
# ------------------------------------------
# 1. LOAD DATASET FROM LOCAL CSV
# ------------------------------------------
file_path = "credit_data.csv"
try:
  df = pd.read_csv(file_path)
  print(f"Successfully loaded {file_path}")
except FileNotFoundError:
  print(
      f"Error: '{file_path}' not found. Please put your CSV file in the same"
      " folder."
  )
  exit()
# Strip any accidental leading/trailing spaces from column names
df.columns = df.columns.str.strip()
print("\n--- Dataset Overview ---")
print(f"Shape: {df.shape}")
target_column = "target"
# Check unique values in the target column before processing
print("Unique target values before mapping:", df[target_column].unique())
# ------------------------------------------
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ------------------------------------------
# Clean and map text labels safely to integers (1 and 0)
df[target_column] = (
    df[target_column].astype(str).str.strip().str.lower()
)  # normalize text
# Map text values: 'good' -> 1, 'bad' -> 0 (Also handle 1/0 if they are already numbers)
mapping = {"good": 1, "bad": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
df[target_column] = df[target_column].map(mapping)
# Check if any values failed to map
if df[target_column].isnull().any():
  print(
      "Warning: Some target values could not be mapped. Filling or dropping"
      " unmapped rows."
  )
  df = df.dropna(subset=[target_column])
# Handle missing values in the rest of the dataset
df = df.dropna()
# Separate Target (y) and Features (X)
y = df[target_column].astype(int)
X_raw = df.drop(columns=[target_column])
# Convert categorical text columns to numeric using One-Hot Encoding
X = pd.get_dummies(X_raw, drop_first=True)
# ------------------------------------------
# 3. TRAIN-TEST SPLIT & FEATURE SCALING
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Feature Scaling (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ------------------------------------------
# 4. MODEL TRAINING & EVALUATION
# ------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
}
roc_data = {}
for name, model in models.items():
  print(f"\n==============================")
  print(f"Training Model: {name}")
  print(f"==============================")
  # Use scaled data for Logistic Regression, raw data for tree-based models
  if name == "Logistic Regression":
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
  else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
  # Required Metrics Assessment
  acc = accuracy_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_prob)
  print(f"Accuracy : {acc:.4f}")
  print(f"ROC-AUC  : {roc_auc:.4f}")
  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))
  # 1. Show Confusion Matrix popup for this model
  plt.figure(figsize=(5, 4))
  sns.heatmap(
      confusion_matrix(y_test, y_pred),
      annot=True,
      fmt="d",
      cmap="Blues",
      cbar=False,
  )
  plt.title(f"Confusion Matrix - {name}")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.tight_layout()
  plt.show()
  # 2. Save ROC curve details to dictionary
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  roc_data[name] = (fpr, tpr, roc_auc)
# ------------------------------------------
# 5. FINAL ROC CURVE COMPARISON PLOT
# ------------------------------------------
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
  plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Credit Scoring Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()