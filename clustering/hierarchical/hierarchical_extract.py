import numpy as np

def extract_clusters(linkage_matrix, num_clusters, N=None):
    if N is None:
        N = linkage_matrix.shape[0] + 1
    
    if num_clusters >= N:
        return np.arange(N)
    
    # Union-Find implementation
    parent = list(range(N))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Apply merges until we have desired number of clusters
    merges_to_apply = N - num_clusters
    
    for i in range(min(merges_to_apply, len(linkage_matrix))):
        cluster_a = int(linkage_matrix[i][0])
        cluster_b = int(linkage_matrix[i][1])
        
        # Only merge if both are original points
        if cluster_a < N and cluster_b < N:
            union(cluster_a, cluster_b)
    
    # Create final labels
    cluster_ids = {}
    labels = np.zeros(N, dtype=int)
    next_id = 0
    
    for i in range(N):
        root = find(i)
        if root not in cluster_ids:
            cluster_ids[root] = next_id
            next_id += 1
        labels[i] = cluster_ids[root]
    
    return labels

def assign_hierarchical_labels(linkage_matrix, levels, N=None):
    """
    For each level (number of clusters), assign a cluster label to each point.
    Returns: dict {level: labels array}
    """
    result = {}
    for num_clusters in levels:
        result[num_clusters] = extract_clusters(linkage_matrix, num_clusters, N)
    return result


def get_merge_height_for_point(linkage_matrix, point_idx, target_num_clusters, N):
    """
    Get the height (distance) at which a point joined its cluster at target_num_clusters level.
    """
    if target_num_clusters >= N:
        return 0.0  # Point never merged
    
    merges_to_apply = N - target_num_clusters
    
    # Find when this point first gets merged into its final cluster
    parent = list(range(N))
    merge_heights = [0.0] * N
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y, height):
        px, py = find(x), find(y)
        if px != py:
            # Update merge height for all points in the smaller cluster
            if px > py:
                px, py = py, px
            parent[px] = py
            merge_heights[px] = max(merge_heights[px], height)
    
    for i in range(min(merges_to_apply, len(linkage_matrix))):
        cluster_a = int(linkage_matrix[i][0])
        cluster_b = int(linkage_matrix[i][1])
        height = float(linkage_matrix[i][2])
        
        if cluster_a < N and cluster_b < N:
            union(cluster_a, cluster_b, height)
            # Update heights for points that just got merged
            if find(point_idx) == find(cluster_a) or find(point_idx) == find(cluster_b):
                merge_heights[point_idx] = height
    
    return merge_heights[point_idx]

# MUCH SIMPLER WORKING VERSION:
# def extract_clusters_simple(linkage_matrix, num_clusters, N):
#     """Simple version that actually works."""
#     from scipy.cluster.hierarchy import fcluster
#     # Convert our format to scipy format if needed, then use fcluster
#     # But since we want custom implementation:
    
#     if num_clusters >= N:
#         return np.arange(N)
    
#     # Union-Find approach
#     parent = list(range(N))
    
#     def find(x):
#         if parent[x] != x:
#             parent[x] = find(parent[x])
#         return parent[x]
    
#     def union(x, y):
#         px, py = find(x), find(y)
#         if px != py:
#             parent[px] = py
    
#     # Apply first (N - num_clusters) merges
#     merges_to_apply = N - num_clusters
    
#     for i in range(min(merges_to_apply, len(linkage_matrix))):
#         cluster_a = int(linkage_matrix[i][0])
#         cluster_b = int(linkage_matrix[i][1])
        
#         # Only merge original points (< N)
#         if cluster_a < N and cluster_b < N:
#             union(cluster_a, cluster_b)
    
#     # Get final cluster assignments
#     clusters = {}
#     cluster_id = 0
#     labels = np.zeros(N, dtype=int)
    
#     for i in range(N):
#         root = find(i)
#         if root not in clusters:
#             clusters[root] = cluster_id
#             cluster_id += 1
#         labels[i] = clusters[root]
    
#     return labels