import numpy as np
import matplotlib.pyplot as plt

def plot_dendrogram(linkage_matrix, N, labels=None, title="Dendrogram"):
    """
    Visualize the dendrogram using matplotlib.
    """
    import scipy.cluster.hierarchy as sch  # Only for plotting!
    sch.dendrogram(linkage_matrix, labels=labels, count_sort='ascending')
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.show()

def plot_cluster_assignments(X, flat_labels, title="Hierarchical Clustering"):
    """
    Visualize flat cluster assignments (2D PCA projection).
    """
    from sklearn.decomposition import PCA
    if X.shape[1] > 2:
        X_2d = PCA(n_components=2).fit_transform(X)
    else:
        X_2d = X
    plt.figure(figsize=(8,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=flat_labels, cmap="tab10", s=25)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

def cluster_size_metrics(flat_labels):
    """Compute cluster sizes and print."""
    unique, counts = np.unique(flat_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} points")
    return dict(zip(unique, counts))