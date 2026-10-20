"""
DSML (ITE01301) - Assignment 6: Linear Regression (Boston Housing)
Name: Atharv Shinde | Class: T.Y IT-A | Batch: D | Roll No: 33076

Run:  python Assignment_6_Linear_Regression.py
Plots open in a window; close each window to continue.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BOSTON_PATH = "Boston (1).csv"   # edit to your file location


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

header("ASSIGNMENT 6: Linear Regression (Boston Housing)")

# a. Data preparation and preprocessing
data = pd.read_csv(BOSTON_PATH)
print("\nDataset Shape:")
print(data.shape)

if "Unnamed: 0" in data.columns:
    data = data.drop("Unnamed: 0", axis=1)

print("\nMissing Values:")
print(data.isnull().sum())

X = data.drop("medv", axis=1)
y = data["medv"]
feature_names = X.columns

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
print("\nMissing values after imputation:")
print(pd.DataFrame(X).isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# b. Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# c. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------------")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

result = pd.DataFrame({"Actual Price": y_test.values,
                       "Predicted Price": y_pred})
print("\nActual vs Predicted:")
print(result.head(10))

# d. Analyze and interpret results
print("\nAnalysis and Interpretation")
print("-------------------------")
print("MSE represents the average squared difference")
print("between actual and predicted house prices.")
print("\nRMSE represents the typical prediction error")
print("in the same units as the target variable.")
print("\nR2 Score represents the proportion of variation")
print("in house prices explained by the model.")

if r2 >= 0.7:
    print("\nThe model has a good fit.")
elif r2 >= 0.5:
    print("\nThe model has a moderate fit.")
else:
    print("\nThe model has a weak fit.")

# Coefficients (added, to interpret each feature)
print("\nCoefficients:")
print(pd.Series(model.coef_, index=feature_names))
print("Intercept:", model.intercept_)

# e. Graph: actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.show()
