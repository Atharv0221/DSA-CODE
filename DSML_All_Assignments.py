"""
DSML (ITE01301) - Lab Assignments 1, 2, 3, 5, 6, 7, 8
Name: Atharv Shinde | Class: T.Y IT-A | Batch: D | Roll No: 33076

How to run
    python DSML_All_Assignments.py          -> runs all assignments one after another
    python DSML_All_Assignments.py 5        -> runs only Assignment 5
    python DSML_All_Assignments.py 1 3 8    -> runs Assignments 1, 3 and 8

Plots open in a window; close each window to continue.
Change the file paths in the CONFIG section below to match your computer.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# CONFIG: dataset file paths (edit these)
# ----------------------------------------------------------------------------
TITANIC_PATH = r"Titanic.csv"                          # Assignment 1
HOUSING_PATH = "Housing.csv"                           # Assignment 2
STUDENT_PATH = "Student_Performance_Dataset.xlsx"      # Assignment 3 (needs openpyxl)
BOSTON_PATH = "Boston (1).csv"                         # Assignment 6
GROCERIES_PATH = "Groceries_dataset.csv"               # Assignment 8

# Assignment 1: set True to fill the 177 missing Age values with the median
# before binning (recommended; your submitted code did not do this).
FILL_MISSING_AGE = False


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# ASSIGNMENT 1: Titanic - preprocessing, summary statistics, visualization
# ============================================================================
def assignment_1():
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


# ============================================================================
# ASSIGNMENT 2: EDA on Housing Price dataset
# ============================================================================
def assignment_2():
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


# ============================================================================
# ASSIGNMENT 3: t-test and Chi-square on Student Performance dataset
# ============================================================================
def assignment_3():
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


# ============================================================================
# ASSIGNMENT 5: SVM and Decision Tree on Iris (with model comparison)
# ============================================================================
def assignment_5():
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


# ============================================================================
# ASSIGNMENT 6: Linear Regression on Boston Housing dataset
# ============================================================================
def assignment_6():
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


# ============================================================================
# ASSIGNMENT 7: K-Means clustering on the Digits dataset
# ============================================================================
def assignment_7():
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    header("ASSIGNMENT 7: K-Means clustering (Digits)")

    digits = load_digits()
    X = digits.data
    y = digits.target

    print("Dataset shape:", X.shape)
    print("Number of images:", len(X))
    print("Features per image:", X.shape[1])

    # Sample images
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap="gray")
        ax.set_title(f"Digit: {y[i]}")
        ax.axis("off")
    plt.suptitle("Sample Images from Digits Dataset")
    plt.tight_layout()
    plt.show()

    # K-Means with K = 10
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    print("\nK-Means Clustering Results")
    print("Number of clusters:", kmeans.n_clusters)
    print("Inertia:", round(kmeans.inertia_, 2))

    silhouette = silhouette_score(X, clusters)
    ari = adjusted_rand_score(y, clusters)
    print("Silhouette Score:", round(silhouette, 4))
    print("Adjusted Rand Index:", round(ari, 4))

    # Elbow method
    inertias = []
    for k in range(2, 16):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(range(2, 16), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(range(2, 16))
    plt.grid(True)
    plt.show()

    # PCA visualization of the clusters
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                          cmap="tab10", s=15)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="X", s=200,
                c="red", edgecolors="black", label="Centroids")
    plt.title("K-Means Clustering of Digits (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.show()

    # Same PCA plot coloured by the true labels
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=15)
    plt.title("Actual Digit Labels (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Actual Digit")
    plt.grid(True)
    plt.show()


# ============================================================================
# ASSIGNMENT 8: Market basket analysis (Apriori from scratch) on Groceries
# ============================================================================
def assignment_8():
    from collections import defaultdict
    from itertools import combinations

    header("ASSIGNMENT 8: Market basket analysis (Apriori)")

    # Load and preprocess: one basket = one (Member_number, Date)
    df = pd.read_csv(GROCERIES_PATH)
    transactions = (df.groupby(["Member_number", "Date"])["itemDescription"]
                      .apply(lambda x: sorted(list(set(x)))).tolist())
    N = len(transactions)
    print("Number of baskets:", N)

    # Step 1: frequent 1-itemsets
    min_support = 0.001
    min_count = min_support * N
    counts = defaultdict(int)
    for t in transactions:
        for item in t:
            counts[frozenset([item])] += 1

    frequent_itemsets = {s: c for s, c in counts.items() if c >= min_count}
    all_frequent = dict(frequent_itemsets)

    # Step 2: Apriori loop for k >= 2
    k = 2
    current = frequent_itemsets
    while current:
        # candidate generation (join step)
        candidates = set()
        items_list = list(current.keys())
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                union_set = items_list[i] | items_list[j]
                if len(union_set) == k:
                    candidates.add(union_set)

        # candidate counting
        cand_counts = defaultdict(int)
        for t in transactions:
            t_set = set(t)
            for cand in candidates:
                if cand.issubset(t_set):
                    cand_counts[cand] += 1

        current = {c: n for c, n in cand_counts.items() if n >= min_count}
        all_frequent.update(current)
        k += 1

    print("Frequent itemsets found:", len(all_frequent))

    # Step 3: rule generation
    rules = []
    for itemset, count in all_frequent.items():
        if len(itemset) > 1:
            sup_itemset = count / N
            items = list(itemset)
            for i in range(1, len(items)):
                for ant in combinations(items, i):
                    ant = frozenset(ant)
                    cons = itemset - ant
                    sup_ant = all_frequent[ant] / N
                    conf = sup_itemset / sup_ant
                    sup_cons = all_frequent[cons] / N
                    lift = conf / sup_cons
                    if conf >= 0.05:
                        rules.append({"antecedents": list(ant),
                                      "consequents": list(cons),
                                      "support": sup_itemset,
                                      "confidence": conf,
                                      "lift": lift})

    rules_df = pd.DataFrame(rules).sort_values(by="lift", ascending=False)
    print(rules_df.head(10).to_string(index=False))


# ============================================================================
# Runner
# ============================================================================
ASSIGNMENTS = {
    1: assignment_1,
    2: assignment_2,
    3: assignment_3,
    5: assignment_5,
    6: assignment_6,
    7: assignment_7,
    8: assignment_8,
}

if __name__ == "__main__":
    wanted = [int(a) for a in sys.argv[1:]] or list(ASSIGNMENTS)
    for number in wanted:
        if number not in ASSIGNMENTS:
            print(f"No assignment {number}. Available: {sorted(ASSIGNMENTS)}")
            continue
        try:
            ASSIGNMENTS[number]()
        except FileNotFoundError as err:
            print(f"\n[Assignment {number}] data file not found: {err.filename}")
            print("Edit the file paths in the CONFIG section at the top.")
