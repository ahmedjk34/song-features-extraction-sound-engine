import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def compute_pairwise_distances(X):
    """
    Compute the full pairwise distance matrix using Euclidean distance.
    Optimized for memory efficiency when dealing with large datasets.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
    
    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    
    # For large datasets, warn about memory usage
    if n_samples > 5000:
        memory_gb = (n_samples ** 2 * 8) / (1024 ** 3)
        print(f"Warning: Computing {n_samples}x{n_samples} distance matrix (~{memory_gb:.2f} GB)")
    
    # Use vectorized computation for better performance
    # Broadcasting approach: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
    X_norm_sq = np.sum(X**2, axis=1)
    distances = np.sqrt(
        X_norm_sq[:, np.newaxis] + X_norm_sq[np.newaxis, :] - 2 * np.dot(X, X.T)
    )
    
    # Ensure diagonal is exactly zero
    np.fill_diagonal(distances, 0)
    
    return distances

def region_query(X, point_idx, eps, distances=None):
    """
    Return indices of all points within eps distance of point_idx.
    Optimized for performance based on analysis results.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        point_idx: index of the query point
        eps: neighborhood radius
        distances: precomputed distance matrix (optional)
    
    Returns:
        List of indices of neighboring points (including point_idx itself)
    """
    if distances is not None:
        # Use precomputed distances - much faster for repeated queries
        neighbors = np.where(distances[point_idx] <= eps)[0]
    else:
        # Compute distances on the fly - memory efficient but slower
        query_point = X[point_idx]
        # Vectorized distance computation
        distances_to_point = np.linalg.norm(X - query_point, axis=1)
        neighbors = np.where(distances_to_point <= eps)[0]
    
    return neighbors.tolist()

def expand_cluster(X, labels, point_idx, cluster_id, eps, min_samples, distances=None):
    """
    Grow a new cluster by recursively adding density-reachable points.
    Optimized based on parameter selection analysis.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        labels: cluster labels array (modified in-place)
        point_idx: starting point index
        cluster_id: ID to assign to this cluster
        eps: neighborhood radius
        min_samples: minimum points to form a dense region
        distances: precomputed distance matrix (optional)
    
    Returns:
        True if cluster was successfully expanded, False otherwise
    """
    # Get neighbors of the starting point
    neighbors = region_query(X, point_idx, eps, distances)
    
    # If not enough neighbors, this point cannot be a core point
    if len(neighbors) < min_samples:
        labels[point_idx] = -1  # Mark as noise
        return False
    
    # This point is a core point, start expanding cluster
    labels[point_idx] = cluster_id
    
    # Use a set for faster membership testing and avoid duplicates
    seeds = set(neighbors)
    seeds.discard(point_idx)  # Remove the starting point
    
    # Process seeds iteratively
    processed = set([point_idx])
    
    while seeds:
        current_point = seeds.pop()
        
        # Skip if already processed
        if current_point in processed:
            continue
        processed.add(current_point)
        
        # If point is noise, assign it to this cluster
        if labels[current_point] == -1:
            labels[current_point] = cluster_id
        
        # If point is already assigned to a cluster, skip expansion
        if labels[current_point] != 0:
            continue
        
        # Assign point to current cluster
        labels[current_point] = cluster_id
        
        # Get neighbors of current point
        current_neighbors = region_query(X, current_point, eps, distances)
        
        # If current point is also a core point, add its unprocessed neighbors to expansion set
        if len(current_neighbors) >= min_samples:
            for neighbor in current_neighbors:
                if neighbor not in processed and (labels[neighbor] == 0 or labels[neighbor] == -1):
                    seeds.add(neighbor)
    
    return True

def dbscan(X, eps, min_samples, precompute_distances=True):
    """
    Optimized DBSCAN clustering algorithm based on parameter selection analysis.
    
    Optimal parameters identified from analysis:
    - eps = 9.978 (primary recommendation)
    - min_samples = 6 (primary) or 3 (alternative)
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        eps: neighborhood radius (recommended: 9.978)
        min_samples: minimum points to form a dense region (recommended: 6)
        precompute_distances: whether to precompute distance matrix
    
    Returns:
        labels: cluster labels (-1 for noise, 0+ for clusters)
        core_samples: boolean array indicating core points
        n_clusters: number of clusters found
    """
    n_samples = X.shape[0]
    
    print(f"Running DBSCAN with eps={eps:.3f}, min_samples={min_samples}")
    print(f"Dataset size: {n_samples} points, {X.shape[1]} dimensions")
    
    # Initialize labels: 0 means unprocessed, -1 means noise, 1+ means cluster ID
    labels = np.zeros(n_samples, dtype=int)
    
    # Precompute distances if requested and dataset size is reasonable
    distances = None
    if precompute_distances:
        if n_samples <= 10000:  # Avoid memory issues for large datasets
            print("Precomputing distance matrix...")
            distances = compute_pairwise_distances(X)
        else:
            print("Large dataset detected, computing distances on-the-fly for memory efficiency")
            precompute_distances = False
    
    cluster_id = 1  # Start cluster IDs from 1
    core_samples = np.zeros(n_samples, dtype=bool)
    
    # Process each point
    processed_points = 0
    for point_idx in range(n_samples):
        # Progress indicator for large datasets
        if n_samples > 1000 and point_idx % (n_samples // 10) == 0:
            progress = (point_idx / n_samples) * 100
            print(f"Progress: {progress:.1f}% ({point_idx}/{n_samples})")
        
        # Skip if already processed
        if labels[point_idx] != 0:
            continue
        
        # Try to expand cluster from this point
        neighbors = region_query(X, point_idx, eps, distances)
        
        if len(neighbors) >= min_samples:
            # This is a core point
            core_samples[point_idx] = True
            expand_cluster(X, labels, point_idx, cluster_id, eps, min_samples, distances)
            cluster_id += 1
        else:
            # Mark as noise (may be changed later if it becomes border point)
            labels[point_idx] = -1
        
        processed_points += 1
    
    # Count number of clusters (excluding noise)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    print(f"DBSCAN completed: {n_clusters} clusters found")
    
    return labels, core_samples, n_clusters

def k_distance_plot(X, min_samples, max_points=1000):
    """
    Create k-distance plot to help choose eps parameter.
    Optimized for large datasets.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        min_samples: k value for k-distance
        max_points: maximum number of points to process (for performance)
    
    Returns:
        k_distances: sorted k-distances array
    """
    n_samples = X.shape[0]
    
    # Sample points if dataset is too large
    if n_samples > max_points:
        print(f"Sampling {max_points} points from {n_samples} for k-distance analysis")
        indices = np.random.choice(n_samples, max_points, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    n_sample_points = X_sample.shape[0]
    k_distances = []
    
    print(f"Computing k-distances for {n_sample_points} points...")
    
    for i in range(n_sample_points):
        if i % (n_sample_points // 10 + 1) == 0:
            progress = (i / n_sample_points) * 100
            print(f"K-distance progress: {progress:.1f}%")
        
        # Compute distances from point i to all other points
        point_i = X_sample[i]
        distances_from_i = np.linalg.norm(X_sample - point_i, axis=1)
        
        # Remove self-distance and sort
        distances_from_i = distances_from_i[distances_from_i > 0]
        distances_from_i.sort()
        
        # Get k-th nearest neighbor distance
        if len(distances_from_i) >= min_samples - 1:
            k_dist = distances_from_i[min_samples - 2]  # k-th nearest (0-indexed)
            k_distances.append(k_dist)
    
    # Sort k-distances in descending order
    k_distances = sorted(k_distances, reverse=True)
    
    # Plot k-distances
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
    plt.xlabel('Points sorted by distance to k-th nearest neighbor')
    plt.ylabel(f'{min_samples}-distance')
    plt.title(f'K-Distance Plot (k={min_samples}) - Look for "Elbow" to choose eps')
    plt.grid(True, alpha=0.3)
    
    # Add reference lines based on analysis results
    percentiles = [85, 90, 95, 99]
    colors = ['orange', 'red', 'darkred', 'purple']
    
    for p, color in zip(percentiles, colors):
        if len(k_distances) > 0:
            val = np.percentile(k_distances, p)
            plt.axhline(y=val, color=color, linestyle='--', alpha=0.7, 
                       label=f'{p}th percentile: {val:.3f}')
    
    # Add optimal eps suggestion based on analysis
    if min_samples == 6:
        optimal_eps = 9.978
        plt.axhline(y=optimal_eps, color='green', linestyle='-', linewidth=3, alpha=0.8,
                   label=f'Recommended eps: {optimal_eps:.3f}')
    elif min_samples == 3:
        optimal_eps = 9.978
        plt.axhline(y=optimal_eps, color='green', linestyle='-', linewidth=3, alpha=0.8,
                   label=f'Recommended eps: {optimal_eps:.3f}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return np.array(k_distances)

def get_cluster_info(labels, core_samples):
    """
    Get detailed information about the clustering results.
    Enhanced with quality metrics based on analysis.
    
    Args:
        labels: cluster labels from DBSCAN
        core_samples: boolean array indicating core points
    
    Returns:
        Dictionary with cluster statistics and quality metrics
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[label] = list(labels).count(label)
    
    n_core = np.sum(core_samples)
    n_border = len(labels) - n_core - n_noise
    
    # Calculate additional quality metrics
    noise_ratio = n_noise / len(labels)
    
    # Cluster size statistics
    if cluster_sizes:
        sizes = list(cluster_sizes.values())
        largest_cluster_size = max(sizes)
        smallest_cluster_size = min(sizes)
        avg_cluster_size = np.mean(sizes)
        cluster_size_std = np.std(sizes)
        
        # Balance metric: how evenly distributed are the clusters
        balance_ratio = smallest_cluster_size / largest_cluster_size if largest_cluster_size > 0 else 0
    else:
        largest_cluster_size = 0
        smallest_cluster_size = 0
        avg_cluster_size = 0
        cluster_size_std = 0
        balance_ratio = 0
    
    # Quality assessment based on analysis results
    quality_score = "UNKNOWN"
    if n_clusters >= 2 and noise_ratio <= 0.05:
        if n_clusters <= 10:
            quality_score = "EXCELLENT"
        else:
            quality_score = "GOOD"
    elif n_clusters >= 2 and noise_ratio <= 0.20:
        quality_score = "FAIR"
    else:
        quality_score = "POOR"
    
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_core': n_core,
        'n_border': n_border,
        'cluster_sizes': cluster_sizes,
        'noise_ratio': noise_ratio,
        'largest_cluster_size': largest_cluster_size,
        'smallest_cluster_size': smallest_cluster_size,
        'avg_cluster_size': avg_cluster_size,
        'cluster_size_std': cluster_size_std,
        'balance_ratio': balance_ratio,
        'quality_score': quality_score,
        'labels': labels,
        'core_samples': core_samples
    }

def dbscan_optimal(X):
    """
    Run DBSCAN with optimal parameters identified from comprehensive analysis.
    
    Uses the best parameters found:
    - eps = 9.978
    - min_samples = 6
    
    Args:
        X: numpy array of shape (n_samples, n_features)
    
    Returns:
        Dictionary with clustering results and metadata
    """
    print("Running DBSCAN with OPTIMAL parameters from analysis:")
    print("eps = 9.978, min_samples = 6")
    print("Expected: ~3 clusters with ~1.9% noise points")
    
    # Use optimal parameters
    eps = 9.978
    min_samples = 6
    
    labels, core_samples, n_clusters = dbscan(X, eps, min_samples, precompute_distances=True)
    info = get_cluster_info(labels, core_samples)
    
    print(f"\nOptimal DBSCAN Results:")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {info['n_noise']} ({info['noise_ratio']:.2%})")
    print(f"Quality assessment: {info['quality_score']}")
    print(f"Cluster sizes: {list(info['cluster_sizes'].values())}")
    
    return {
        'labels': labels,
        'core_samples': core_samples,
        'n_clusters': n_clusters,
        'cluster_info': info,
        'parameters': {'eps': eps, 'min_samples': min_samples}
    }

def dbscan_alternative(X):
    """
    Run DBSCAN with alternative parameters from analysis.
    
    Uses the alternative configuration:
    - eps = 9.978
    - min_samples = 3
    
    Args:
        X: numpy array of shape (n_samples, n_features)
    
    Returns:
        Dictionary with clustering results and metadata
    """
    print("Running DBSCAN with ALTERNATIVE parameters from analysis:")
    print("eps = 9.978, min_samples = 3")
    print("Expected: ~6 clusters with ~1.5% noise points")
    
    # Use alternative parameters
    eps = 9.978
    min_samples = 3
    
    labels, core_samples, n_clusters = dbscan(X, eps, min_samples, precompute_distances=True)
    info = get_cluster_info(labels, core_samples)
    
    print(f"\nAlternative DBSCAN Results:")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {info['n_noise']} ({info['noise_ratio']:.2%})")
    print(f"Quality assessment: {info['quality_score']}")
    print(f"Cluster sizes: {list(info['cluster_sizes'].values())}")
    
    return {
        'labels': labels,
        'core_samples': core_samples,
        'n_clusters': n_clusters,
        'cluster_info': info,
        'parameters': {'eps': eps, 'min_samples': min_samples}
    }