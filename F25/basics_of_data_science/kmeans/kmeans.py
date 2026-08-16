#!/usr/bin/env python3
"""
K-Means from scratch (dynamic, no sklearn)
------------------------------------------
- Randomized cluster centers and sample sizes
- Dynamic number of clusters
- K-means++ initialization, from scratch
- Combined 2x1 visualization: training and test sets in one figure
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List


class KMeansScratch:
    def __init__(
        self,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = "k-means++",
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def fit(self, X: np.ndarray):
        X = self._validate_X(X)
        rng_master = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centroids, best_labels, best_n_iter = None, None, None

        '''
        DEFINE YOUR FITTING ALGORITHM
        '''

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X)
        labels, _ = self._assign_labels(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def _validate_X(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
        return X

    def _check_fitted(self):
        if self.cluster_centers_ is None:
            raise RuntimeError("This KMeansScratch instance is not fitted yet.")

    def _init_centroids(self, X, rng):
        if self.init == "random":
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[idx].copy()

        # k-means++ init
        centroids = np.empty((self.n_clusters, X.shape[1]))
        centroids[0] = X[rng.integers(0, X.shape[0])]
        closest_sq_dist = self._squared_distances_to_centroids(X, centroids[:1]).min(axis=1)
        for i in range(1, self.n_clusters):
            probs = closest_sq_dist / closest_sq_dist.sum()
            next_idx = rng.choice(X.shape[0], p=probs)
            centroids[i] = X[next_idx]
            new_dists = self._squared_distances_to_centroids(X, centroids[i:i+1]).min(axis=1)
            closest_sq_dist = np.minimum(closest_sq_dist, new_dists)
        return centroids

    def _squared_distances_to_centroids(self, X, centroids):
        X_norm = (X ** 2).sum(axis=1, keepdims=True)
        C_norm = (centroids ** 2).sum(axis=1, keepdims=True).T
        cross = X @ centroids.T
        return np.maximum(X_norm + C_norm - 2.0 * cross, 0.0)

    def _assign_labels(self, X, centroids):
        '''
        DEFINE YOUR ALGORITHM
        '''

        return labels, min_d2

    def _recompute_centroids(self, X, labels, old_centroids):
        '''
        DEFINE YOUR ALGORITHM
        '''

        return new_centroids


def make_blobs(
    n_samples_per_cluster: List[int],
    centers: np.ndarray,
    cluster_std: Union[List[float], float] = 1.0,
    random_state: Optional[int] = None,
):
    rng = np.random.default_rng(random_state)
    centers = np.asarray(centers, dtype=float)
    n_clusters, n_features = centers.shape

    if isinstance(cluster_std, (list, tuple, np.ndarray)):
        stds = np.asarray(cluster_std, dtype=float)
    else:
        stds = np.full(n_clusters, float(cluster_std))

    X_list, y_list = [], []
    for i in range(n_clusters):
        X_i = rng.normal(loc=centers[i], scale=stds[i], size=(n_samples_per_cluster[i], n_features))
        X_list.append(X_i)
        y_list.append(np.full(n_samples_per_cluster[i], i, dtype=int))

    return np.vstack(X_list), np.concatenate(y_list)


def plot_train_and_test(X_train, y_train, X_test, y_test_pred, km):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

    # Training data with true labels
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=18, alpha=0.8, edgecolor="none")
    ax1.set_title("Training Data (True Labels)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    # Test data with predicted labels
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, s=18, alpha=0.8, edgecolor="none")
    ax2.set_title("Test Data (Predicted Labels)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    plt.tight_layout()
    plt.show()



def main():
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)

    # Random number of clusters and centers
    n_clusters = 5  # change this to any number
    centers = rng.uniform(low=-10, high=10, size=(n_clusters, 2))
    print("Random cluster centers:\n", centers)

    # Dynamic sample sizes
    samples_per_train_cluster = 250
    n_per_cluster_train = [samples_per_train_cluster] * n_clusters

    samples_per_test_cluster = 180
    variation = 10
    n_per_cluster_test = [samples_per_test_cluster + i * variation for i in range(n_clusters)]

    # Random cluster spreads
    stds_train = rng.uniform(0.5, 1.0, size=n_clusters)
    stds_test = rng.uniform(0, 2, size=n_clusters)

    # Generate training and test data
    X_train, y_train_true = make_blobs(n_per_cluster_train, centers, cluster_std=stds_train, random_state=rng_seed)
    X_test, _ = make_blobs(n_per_cluster_test, centers, cluster_std=stds_test, random_state=rng_seed + 1)

    # Fit model
    km = KMeansScratch(n_clusters=n_clusters, n_init=10, max_iter=200, tol=1e-4, init="k-means++", random_state=rng_seed)
    km.fit(X_train)
    y_test_pred = km.predict(X_test)

    print(f"Fitted inertia: {km.inertia_:.4f}")
    print(f"Cluster centers:\n{km.cluster_centers_}")

    # Combined visualization
    plot_train_and_test(X_train, y_train_true, X_test, y_test_pred, km)


if __name__ == "__main__":
    main()
