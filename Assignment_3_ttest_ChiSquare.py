"""
DSML (ITE01301) - Assignment 3: t-test and Chi-square test (Student Performance dataset)
Name: Atharv Shinde | Class: T.Y IT-A | Batch: D | Roll No: 33076

Run:  python Assignment_3_ttest_ChiSquare.py
Plots open in a window; close each window to continue.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

STUDENT_PATH = "Student_Performance_Dataset.xlsx"   # edit; needs openpyxl


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


from scipy.stats import ttest_ind, chi2_contingency

header("ASSIGNMENT 3: t-test and Chi-square test")

df = pd.read_excel(STUDENT_PATH)

# A] Independent samples t-test (Welch): math scores of male vs female
male_scores = df[df["Gender"] == "Male"]["Math_Score"]
female_scores = df[df["Gender"] == "Female"]["Math_Score"]

t_stat, p_value = ttest_ind(male_scores, female_scores, equal_var=False)

print("=" * 60)
print("INDEPENDENT SAMPLE t-TEST")
print("=" * 60)
print("Mean Math Score (Male)   :", round(male_scores.mean(), 2))
print("Mean Math Score (Female) :", round(female_scores.mean(), 2))
print("\nT-statistic :", round(t_stat, 4))
print("P-value     :", round(p_value, 4))

alpha = 0.05
if p_value < alpha:
    print("\nResult: Reject the Null Hypothesis")
    print("There is a significant difference in mathematics scores.")
else:
    print("\nResult: Fail to Reject the Null Hypothesis")
    print("No significant difference in mathematics scores.")
# H0: no significant difference in mathematics scores
# H1: significant difference in mathematics scores

# B] Chi-square test of independence: parental education vs performance
print("\n" + "=" * 60)
print("CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 60)

contingency_table = pd.crosstab(df["Parental_Education"],
                                df["Performance_Level"])
print("\nContingency Table:\n")
print(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Statistic :", round(chi2, 4))
print("Degrees of Freedom   :", dof)
print("P-value              :", round(p, 4))

if p < alpha:
    print("\nResult: Reject the Null Hypothesis")
    print("Parental education and student performance are "
          "significantly associated.")
else:
    print("\nResult: Fail to Reject the Null Hypothesis")
    print("No significant association exists between parental education "
          "and student performance.")

print("\nExpected Frequencies:\n")
print(pd.DataFrame(expected, index=contingency_table.index,
                   columns=contingency_table.columns))
