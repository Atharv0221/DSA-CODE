"""
DSML (ITE01301) - Assignment 8: Market basket analysis (Apriori from scratch) on Groceries
Name: Atharv Shinde | Class: T.Y IT-A | Batch: D | Roll No: 33076

Run:  python Assignment_8_Apriori_Groceries.py
Plots open in a window; close each window to continue.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

GROCERIES_PATH = "Groceries_dataset.csv"   # edit to your file location


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


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
