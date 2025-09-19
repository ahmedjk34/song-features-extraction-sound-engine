import numpy as np
import matplotlib.pyplot as plt

def compute_inertia(X, centroids, labels):
    """Compute inertia (sum of squared distances to nearest centroid)."""
    inertia = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        if cluster_points.size > 0:
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            inertia += np.sum(distances ** 2)
    return inertia

def kmeans(X, n_clusters=5, max_iters=100, tol=1e-4, n_init=10):
    """Perform K-Means clustering on data X."""
    best_inertia = float('inf')
    best_centroids = None
    best_labels = None

    for init_no in range(n_init):
        # Randomly initialize centroids
        random_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
        centroids = X[random_indices]

        for iteration in range(max_iters):
            # Assignment step
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update step
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(n_clusters)
            ])

            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            if centroid_shift < tol:
                break

            centroids = new_centroids

        inertia = compute_inertia(X, centroids, labels)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels, best_inertia

