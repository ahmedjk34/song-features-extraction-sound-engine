import numpy as np

def compute_pairwise_distance(X, metric="euclidean"):
    """Compute the full pairwise distance matrix for X (N x N)."""
    N = X.shape[0]
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if metric == "euclidean":
                dist = np.linalg.norm(X[i] - X[j])
            elif metric == "cosine":
                dist = 1 - np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-10)
            else:
                raise ValueError("Unsupported metric")
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix



def initialize_clusters(N):
    """Each point is its own cluster initially. Clusters is a list of lists of indices."""
    return [[i] for i in range(N)]

def compute_linkage_distance(cluster_a, cluster_b, dist_matrix, linkage):
    """Compute the linkage distance between two clusters, given the distance matrix."""
    indices_a = np.array(cluster_a)
    indices_b = np.array(cluster_b)
    dists = dist_matrix[np.ix_(indices_a, indices_b)]
    if linkage == "single":
        return np.min(dists)
    elif linkage == "complete":
        return np.max(dists)
    elif linkage == "average":
        return np.mean(dists)
    else:
        raise ValueError("Unsupported linkage type")


def find_closest_clusters(clusters, dist_matrix, linkage):
    """Return indices of the two closest clusters to merge."""
    min_dist = np.inf
    pair = (None, None)
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            dist = compute_linkage_distance(clusters[i], clusters[j], dist_matrix, linkage)
            if dist < min_dist:
                min_dist = dist
                pair = (i, j)
    return pair, min_dist

def merge_clusters(clusters, idx_a, idx_b):
    """Merge clusters at idx_a and idx_b, replacing them with their union."""
    merged = clusters[idx_a] + clusters[idx_b]
    new_clusters = [clusters[i] for i in range(len(clusters)) if i not in (idx_a, idx_b)]
    new_clusters.append(merged)
    return new_clusters

def update_distance_matrix(dist_matrix, clusters, idx_a, idx_b, linkage):
    """Update the distance matrix after merging clusters at idx_a, idx_b."""
    # Remove rows/cols for merged clusters, add new row/col for new cluster
    N = len(clusters)
    new_dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if i == N-1 or j == N-1:  # new cluster
                # Get cluster lists
                cluster_i = clusters[i]
                cluster_j = clusters[j]
                dist = compute_linkage_distance(cluster_i, cluster_j, dist_matrix, linkage)
                new_dist_matrix[i, j] = new_dist_matrix[j, i] = dist
            else:
                # Copy old distances (both clusters survived)
                new_dist_matrix[i, j] = new_dist_matrix[j, i] = dist_matrix[i, j]
    return new_dist_matrix

def hierarchical_clustering(X, linkage="average", metric="euclidean"):
    N = X.shape[0]
    if N <= 1:
        return np.array([])
    
    dist_matrix = compute_pairwise_distance(X, metric)
    clusters = [[i] for i in range(N)]
    linkage_matrix = []
    cluster_map = {i: i for i in range(N)}  # Track cluster ID mapping
    next_cluster_id = N
    
    for step in range(N - 1):
        # Find closest clusters
        min_dist = np.inf
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = compute_linkage_distance(clusters[i], clusters[j], dist_matrix, linkage)
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j
        
        cluster_a = clusters[merge_i]
        cluster_b = clusters[merge_j]
        
        # Get proper cluster IDs
        id_a = cluster_a[0] if len(cluster_a) == 1 else cluster_map[tuple(sorted(cluster_a))]
        id_b = cluster_b[0] if len(cluster_b) == 1 else cluster_map[tuple(sorted(cluster_b))]
        
        merged = cluster_a + cluster_b
        cluster_map[tuple(sorted(merged))] = next_cluster_id
        
        linkage_matrix.append([id_a, id_b, min_dist, len(merged)])
        
        # Update clusters list
        if merge_i > merge_j:
            clusters.pop(merge_i)
            clusters.pop(merge_j)
        else:
            clusters.pop(merge_j)
            clusters.pop(merge_i)
        clusters.append(merged)
        
        next_cluster_id += 1
    
    return np.array(linkage_matrix)