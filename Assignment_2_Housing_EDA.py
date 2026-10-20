"""
DSML (ITE01301) - Assignment 2: EDA on Housing Price dataset
Name: Atharv Shinde | Class: T.Y IT-A | Batch: D | Roll No: 33076

Run:  python Assignment_2_Housing_EDA.py
Plots open in a window; close each window to continue.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HOUSING_PATH = "Housing.csv"   # edit to your file location


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


header("ASSIGNMENT 2: EDA on Housing Price dataset")

# a. Load and inspect
df = pd.read_csv(HOUSING_PATH)
print("=" * 60)
print("FIRST FIVE RECORDS")
print("=" * 60)
print(df.head())
print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(df.info())
print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print("Rows and Columns :", df.shape)
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# b. Summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
numeric_columns = ["price", "area"]
for column in numeric_columns:
    print("\nColumn :", column)
    print("------------------------------")
    print("Mean              :", round(df[column].mean(), 2))
    print("Median            :", round(df[column].median(), 2))
    print("Standard Deviation:", round(df[column].std(), 2))
    print("Variance          :", round(df[column].var(), 2))
print("\nComplete Statistical Summary")
print(df.describe())

# c. Distribution: histograms
for col in ["price", "area", "bedrooms"]:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of Housing {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# c. Distribution: box plots
for col in ["price", "area"]:
    plt.figure(figsize=(5, 8))
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(f"Box plot of Housing {col}")
    plt.ylabel(col)
    plt.grid(True)
    plt.show()

# d, e. Correlation and covariance (not in the submitted PDF: added)
num = df.select_dtypes(include="number")
print("\nCorrelation matrix:")
print(num.corr())
print("\nCovariance matrix:")
print(num.cov())
plt.figure(figsize=(8, 6))
sns.heatmap(num.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()
print("price vs area correlation:", df["price"].corr(df["area"]))

# f. Pivot table (added)
pivot = pd.pivot_table(df, values="price", index="bedrooms",
                       columns="furnishingstatus", aggfunc="mean")
print("\nPivot table (mean price):")
print(pivot)
