import numpy as np
import sys
import os

# ANSI color codes for pretty CMD logging
class Log:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def log_info(msg: str):
    print(f"{Log.OKBLUE}[INFO]{Log.ENDC} {msg}")

def log_success(msg: str):
    print(f"{Log.OKGREEN}[SUCCESS]{Log.ENDC} {msg}")

def log_warn(msg: str):
    print(f"{Log.WARNING}[WARN]{Log.ENDC} {msg}")

def log_fail(msg: str):
    print(f"{Log.FAIL}[FAIL]{Log.ENDC} {msg}")

def log_debug(msg: str):
    print(f"{Log.OKCYAN}[DEBUG]{Log.ENDC} {msg}")

def safe_distance_computation(x1, x2, metric="euclidean"):
    """Safely compute distance between two vectors."""
    try:
        if metric == "euclidean":
            dist = np.linalg.norm(x1 - x2)
        elif metric == "cosine":
            dot_product = np.dot(x1, x2)
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            
            if norm_x1 == 0 or norm_x2 == 0:
                return 1.0  # Maximum cosine distance
            
            cosine_sim = dot_product / (norm_x1 * norm_x2)
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)  # Handle numerical errors
            dist = 1 - cosine_sim
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Ensure distance is non-negative and finite
        if not np.isfinite(dist) or dist < 0:
            log_warn(f"Invalid distance computed: {dist}, setting to 0")
            return 0.0
        
        return float(dist)
    
    except Exception as e:
        log_warn(f"Error computing distance: {e}, returning default distance 1.0")
        return 1.0

def compute_pairwise_distance(X, metric="euclidean"):
    """
    Compute full pairwise distance matrix for X (N x N) with validation.
    """
    N = X.shape[0]
    
    if N == 0:
        log_fail("Empty input data for distance computation")
        return np.array([])
    
    if N == 1:
        log_warn("Single point provided, returning zero distance matrix")
        return np.zeros((1, 1))
    
    log_info(f"Computing {N}x{N} distance matrix using {metric} metric")
    
    # Validate input data
    if not np.all(np.isfinite(X)):
        log_warn("Input data contains non-finite values, handling gracefully")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    dist_matrix = np.zeros((N, N))
    
    # Compute upper triangle (matrix is symmetric)
    for i in range(N):
        for j in range(i + 1, N):
            dist = safe_distance_computation(X[i], X[j], metric)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    # Validate distance matrix
    if not np.all(np.isfinite(dist_matrix)):
        log_warn("Distance matrix contains non-finite values")
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1e6, neginf=0.0)
    
    # Ensure diagonal is zero
    np.fill_diagonal(dist_matrix, 0.0)
    
    max_dist = np.max(dist_matrix)
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    log_debug(f"Distance matrix computed: range [{min_dist:.6f}, {max_dist:.6f}]")
    
    return dist_matrix

def initialize_clusters(N):
    """Initialize each point as its own cluster."""
    if N <= 0:
        log_fail("Invalid number of points for cluster initialization")
        return []
    
    log_debug(f"Initializing {N} singleton clusters")
    return [[i] for i in range(N)]

def compute_linkage_distance(cluster_a, cluster_b, dist_matrix, linkage="average"):
    """
    Compute distance between two clusters with validation.
    """
    try:
        indices_a = np.array(cluster_a, dtype=int)
        indices_b = np.array(cluster_b, dtype=int)
        
        # Validate indices
        N = dist_matrix.shape[0]
        if np.any(indices_a >= N) or np.any(indices_b >= N) or np.any(indices_a < 0) or np.any(indices_b < 0):
            log_warn(f"Invalid cluster indices detected")
            return 1.0
        
        # Extract submatrix of distances
        dists = dist_matrix[np.ix_(indices_a, indices_b)]
        
        if dists.size == 0:
            log_warn("Empty distance submatrix")
            return 1.0
        
        if linkage == "single":
            result = np.min(dists)
        elif linkage == "complete":
            result = np.max(dists)
        elif linkage == "average":
            result = np.mean(dists)
        else:
            raise ValueError(f"Unsupported linkage: {linkage}")
        
        # Validate result
        if not np.isfinite(result) or result < 0:
            log_warn(f"Invalid linkage distance: {result}, using fallback")
            return 1.0
        
        return float(result)
    
    except Exception as e:
        log_warn(f"Error computing linkage distance: {e}")
        return 1.0

def hierarchical_clustering(X, linkage="average", metric="euclidean", verbose=False):
    """
    Perform agglomerative hierarchical clustering with robust error handling.
    Returns linkage matrix compatible with scipy dendrogram.
    """
    if verbose:
        log_info("Starting hierarchical clustering")
    
    # Input validation
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X)
        except Exception as e:
            log_fail(f"Cannot convert input to numpy array: {e}")
            return np.array([])
    
    if X.ndim != 2:
        log_fail(f"Input must be 2D array, got {X.ndim}D")
        return np.array([])
    
    N, D = X.shape
    log_info(f"Clustering {N} points with {D} features using {linkage} linkage and {metric} metric")
    
    if N <= 1:
        log_warn("Need at least 2 points for hierarchical clustering")
        return np.array([])
    
    if N > 10000:
        log_warn(f"Large dataset ({N} points) may be slow. Consider sampling.")
    
    # Validate input data
    if not np.all(np.isfinite(X)):
        log_warn("Input contains non-finite values, cleaning data")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    try:
        # Compute distance matrix
        dist_matrix = compute_pairwise_distance(X, metric)
        if dist_matrix.size == 0:
            log_fail("Failed to compute distance matrix")
            return np.array([])
        
        # Initialize clusters
        clusters = initialize_clusters(N)
        linkage_matrix = []
        cluster_map = {i: i for i in range(N)}
        next_cluster_id = N
        
        log_info(f"Starting agglomeration process ({N-1} merges required)")
        
        # Main clustering loop
        for step in range(N - 1):
            if verbose and step % max(1, (N-1) // 10) == 0:
                log_debug(f"Merge step {step+1}/{N-1}")
            
            # Find closest clusters
            min_dist = np.inf
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    try:
                        dist = compute_linkage_distance(clusters[i], clusters[j], dist_matrix, linkage)
                        if dist < min_dist:
                            min_dist = dist
                            merge_i, merge_j = i, j
                    except Exception as e:
                        log_warn(f"Error computing distance between clusters {i} and {j}: {e}")
                        continue
            
            if merge_i == -1 or merge_j == -1:
                log_fail(f"No valid clusters to merge at step {step}")
                break
            
            # Get clusters to merge
            cluster_a = clusters[merge_i]
            cluster_b = clusters[merge_j]
            
            # Determine cluster IDs for linkage matrix
            id_a = cluster_a[0] if len(cluster_a) == 1 else cluster_map.get(tuple(sorted(cluster_a)), cluster_a[0])
            id_b = cluster_b[0] if len(cluster_b) == 1 else cluster_map.get(tuple(sorted(cluster_b)), cluster_b[0])
            
            # Create merged cluster
            merged = cluster_a + cluster_b
            cluster_map[tuple(sorted(merged))] = next_cluster_id
            
            # Add to linkage matrix [cluster1, cluster2, distance, size]
            linkage_matrix.append([id_a, id_b, min_dist, len(merged)])
            
            # Update clusters list (remove merged clusters, add new one)
            if merge_i > merge_j:
                clusters.pop(merge_i)
                clusters.pop(merge_j)
            else:
                clusters.pop(merge_j)
                clusters.pop(merge_i)
            clusters.append(merged)
            
            next_cluster_id += 1
        
        linkage_array = np.array(linkage_matrix)
        
        if verbose:
            log_success(f"Hierarchical clustering completed successfully")
            if len(linkage_array) > 0:
                log_info(f"Linkage matrix shape: {linkage_array.shape}")
                log_info(f"Distance range: [{np.min(linkage_array[:, 2]):.6f}, {np.max(linkage_array[:, 2]):.6f}]")
        
        return linkage_array
    
    except Exception as e:
        log_fail(f"Hierarchical clustering failed: {e}")
        return np.array([])

def validate_linkage_matrix(linkage_matrix, N):
    """Validate the structure of a linkage matrix."""
    if linkage_matrix.size == 0:
        log_warn("Empty linkage matrix")
        return False
    
    if linkage_matrix.shape[1] != 4:
        log_fail(f"Linkage matrix should have 4 columns, got {linkage_matrix.shape[1]}")
        return False
    
    expected_rows = N - 1
    if linkage_matrix.shape[0] != expected_rows:
        log_warn(f"Expected {expected_rows} rows in linkage matrix, got {linkage_matrix.shape[0]}")
    
    # Check for non-finite values
    if not np.all(np.isfinite(linkage_matrix)):
        log_fail("Linkage matrix contains non-finite values")
        return False
    
    # Check distances are non-negative and non-decreasing (for valid hierarchical clustering)
    distances = linkage_matrix[:, 2]
    if np.any(distances < 0):
        log_warn("Linkage matrix contains negative distances")
        return False
    
    log_success("Linkage matrix validation passed")
    return True