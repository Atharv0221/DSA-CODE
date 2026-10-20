"""
DSML (ITE01301) - Assignment 7: K-Means clustering on the Digits dataset
Name: Atharv Shinde | Class: T.Y IT-A | Batch: D | Roll No: 33076

Run:  python Assignment_7_KMeans_Digits.py
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
