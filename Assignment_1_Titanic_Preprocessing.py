"""
DSML (ITE01301) - Assignment 1: Titanic: preprocessing, summary statistics, visualization
Name: Atharv Shinde | Class: T.Y IT-A | Batch: D | Roll No: 33076

Run:  python Assignment_1_Titanic_Preprocessing.py
Plots open in a window; close each window to continue.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TITANIC_PATH = r"Titanic.csv"   # edit to your file location
# Set True to fill the 177 missing Age values with the median before binning
FILL_MISSING_AGE = False


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


header("ASSIGNMENT 1: Titanic preprocessing and visualization")

# a. Load and inspect
df = pd.read_csv(TITANIC_PATH)
print("First 5 Records:")
print(df.head())
print("\nDataset Information:")
print(df.info())

if FILL_MISSING_AGE:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# b. Cleaning: remove Fare outliers with the IQR rule
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["Fare"] >= Q1 - 1.5 * IQR) & (df["Fare"] <= Q3 + 1.5 * IQR)]
print("Dataset after removing outliers:")
print(df.shape)

# b. Transformation: bin Age into groups
bins = [0, 12, 19, 35, 60, 100]
labels = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
print(df[["Age", "AgeGroup"]].head())

# c. Summary statistics
print("\nAdditional Summary Statistics")
for col in ["Age", "Fare"]:
    print(f"\nColumn: {col}")
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Standard Deviation:", df[col].std())
    print("Variance:", df[col].var())
    print("Range:", df[col].max() - df[col].min())

# d. Histogram of Fare
plt.figure(figsize=(6, 4))
plt.hist(df["Fare"], bins=20, color="pink", edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# d. Pie chart of passenger class (labels taken from the data itself)
pclass = df["Pclass"].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(pclass, labels=[f"Class {c}" for c in pclass.index],
        autopct="%1.1f%%", startangle=90)
plt.title("Passenger Class Distribution")
plt.show()
