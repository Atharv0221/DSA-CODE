"""
DSML (ITE01301) - Assignment 5: Linear SVM vs Decision Tree on Iris
Name: Atharv Shinde | Class: T.Y IT-A | Batch: D | Roll No: 33076

Run:  python Assignment_5_SVM_DecisionTree.py
Plots open in a window; close each window to continue.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


from sklearn.datasets import load_iris
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

header("ASSIGNMENT 5: Linear SVM vs Decision Tree on Iris")

# 1. Load and split (80% train, 20% test)
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

def plot_cm(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# 2. Linear SVM (scaled data; scaler fitted on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel="linear", C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print("================ Linear SVM Performance ================")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}\n")
print(classification_report(y_test, y_pred_svm, target_names=target_names))
plot_cm(confusion_matrix(y_test, y_pred_svm), "Linear SVM")

# 3. Decision Tree (scale-invariant, so unscaled data)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("================ Decision Tree Performance ================")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}\n")
print(classification_report(y_test, y_pred_dt, target_names=target_names))
plot_cm(confusion_matrix(y_test, y_pred_dt), "Decision Tree")

# 4. Comparative metrics analysis
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_dt = accuracy_score(y_test, y_pred_dt)
errors_svm = np.sum(y_test != y_pred_svm)
errors_dt = np.sum(y_test != y_pred_dt)
_, _, f1_svm, _ = precision_recall_fscore_support(
    y_test, y_pred_svm, average="macro")
_, _, f1_dt, _ = precision_recall_fscore_support(
    y_test, y_pred_dt, average="macro")

print("=== METRIC-BY-METRIC COMPARISON ===\n")

if acc_dt > acc_svm:
    print(f"1. Accuracy: Decision Tree is better ({acc_dt:.4f} vs {acc_svm:.4f})")
elif acc_svm > acc_dt:
    print(f"1. Accuracy: Linear SVM is better ({acc_svm:.4f} vs {acc_dt:.4f})")
else:
    print(f"1. Accuracy: Both models perform equally ({acc_svm:.4f})")

if errors_dt < errors_svm:
    print(f"2. Error Count: Decision Tree is better with fewer errors "
          f"({errors_dt} vs {errors_svm})")
elif errors_svm < errors_dt:
    print(f"2. Error Count: Linear SVM is better with fewer errors "
          f"({errors_svm} vs {errors_dt})")
else:
    print(f"2. Error Count: Both models made equal errors ({errors_svm})")

if f1_dt > f1_svm:
    print(f"3. Macro F1-Score: Decision Tree is better ({f1_dt:.4f} vs {f1_svm:.4f})")
elif f1_svm > f1_dt:
    print(f"3. Macro F1-Score: Linear SVM is better ({f1_svm:.4f} vs {f1_dt:.4f})")
else:
    print(f"3. Macro F1-Score: Both models perform equally ({f1_svm:.4f})")
