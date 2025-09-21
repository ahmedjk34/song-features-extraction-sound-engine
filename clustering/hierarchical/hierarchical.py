import numpy as np
import heapq
from scipy.spatial.distance import pdist, squareform
from numba import jit, prange
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

@jit(nopython=True)
def euclidean_distance_numba(x1, x2):
    """Numba-optimized euclidean distance."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

@jit(nopython=True, parallel=True)
def compute_distance_matrix_numba(X):
    """Numba-optimized distance matrix computation."""
    n = X.shape[0]
    distances = np.zeros((n, n))
    
    for i in prange(n):
        for j in prange(i + 1, n):
            dist = euclidean_distance_numba(X[i], X[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances

def compute_pairwise_distance_optimized(X, metric="euclidean"):
    """
    Optimized distance matrix computation using scipy and numba.
    """
    N = X.shape[0]
    
    if N == 0:
        log_fail("Empty input data for distance computation")
        return np.array([])
    
    if N == 1:
        return np.zeros((1, 1))
    
    log_info(f"Computing {N}x{N} distance matrix using optimized {metric}")
    
    # Validate input data
    if not np.all(np.isfinite(X)):
        log_warn("Input data contains non-finite values, cleaning")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    try:
        if metric == "euclidean":
            # Method 1: Use scipy's pdist (fastest for most cases)
            if N < 5000:
                condensed_dist = pdist(X, metric='euclidean')
                dist_matrix = squareform(condensed_dist)
            else:
                # Method 2: Use numba for very large matrices
                dist_matrix = compute_distance_matrix_numba(X.astype(np.float64))
        else:
            # Fallback for other metrics
            condensed_dist = pdist(X, metric=metric)
            dist_matrix = squareform(condensed_dist)
        
        max_dist = np.max(dist_matrix)
        min_dist = np.min(dist_matrix[dist_matrix > 0])
        log_debug(f"Distance matrix computed: range [{min_dist:.6f}, {max_dist:.6f}]")
        
        return dist_matrix
        
    except Exception as e:
        log_fail(f"Optimized distance computation failed: {e}")
        # Fallback to basic method
        return compute_basic_distance_matrix(X, metric)

def compute_basic_distance_matrix(X, metric):
    """Fallback basic distance matrix computation."""
    N = X.shape[0]
    dist_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            if metric == "euclidean":
                dist = np.linalg.norm(X[i] - X[j])
            else:
                dist = 1.0
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    return dist_matrix

class OptimizedClusterTracker:
    """
    Efficient cluster tracking using disjoint set with optimizations.
    """
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n)
        self.cluster_members = {i: [i] for i in range(n)}  # Track actual members
        self.active_clusters = set(range(n))
        self.n = n
    
    def find(self, x):
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank with member tracking."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        # Merge cluster members
        self.cluster_members[root_x].extend(self.cluster_members[root_y])
        del self.cluster_members[root_y]
        self.active_clusters.discard(root_y)
        
        return True
    
    def get_cluster_members(self, cluster_id):
        """Get members of a cluster."""
        root = self.find(cluster_id)
        return self.cluster_members.get(root, [])
    
    def get_all_clusters(self):
        """Get all active clusters."""
        return {root: members for root, members in self.cluster_members.items() 
                if root in self.active_clusters}

@jit(nopython=True)
def compute_linkage_distance_numba(cluster_a_indices, cluster_b_indices, dist_matrix, linkage_type):
    """Numba-optimized linkage distance computation."""
    if linkage_type == 0:  # single
        min_dist = np.inf
        for i in cluster_a_indices:
            for j in cluster_b_indices:
                if dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
        return min_dist
    elif linkage_type == 1:  # complete
        max_dist = 0.0
        for i in cluster_a_indices:
            for j in cluster_b_indices:
                if dist_matrix[i, j] > max_dist:
                    max_dist = dist_matrix[i, j]
        return max_dist
    else:  # average
        total = 0.0
        count = 0
        for i in cluster_a_indices:
            for j in cluster_b_indices:
                total += dist_matrix[i, j]
                count += 1
        return total / count if count > 0 else 0.0

class PriorityQueue:
    """
    Optimized priority queue for cluster distances.
    """
    def __init__(self):
        self.heap = []
        self.counter = 0
    
    def push(self, distance, cluster_i, cluster_j):
        """Add a cluster pair with distance to the queue."""
        heapq.heappush(self.heap, (distance, self.counter, cluster_i, cluster_j))
        self.counter += 1
    
    def pop_valid(self, valid_clusters):
        """Pop the next valid cluster pair."""
        while self.heap:
            distance, _, cluster_i, cluster_j = heapq.heappop(self.heap)
            if cluster_i in valid_clusters and cluster_j in valid_clusters:
                return distance, cluster_i, cluster_j
        return None, None, None

def hierarchical_clustering_optimized(X, linkage="average", metric="euclidean", verbose=False):
    """
    Heavily optimized hierarchical clustering implementation.
    """
    if verbose:
        log_info("Starting OPTIMIZED hierarchical clustering")
    
    # Input validation
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X, dtype=np.float64)
        except Exception as e:
            log_fail(f"Cannot convert input to numpy array: {e}")
            return np.array([])
    
    X = X.astype(np.float64)  # Ensure float64 for numba
    N, D = X.shape
    
    if N <= 1:
        log_warn("Need at least 2 points for hierarchical clustering")
        return np.array([])
    
    log_info(f"Clustering {N} points with {D} features using optimized {linkage} linkage")
    
    if N > 20000:
        log_warn(f"Very large dataset ({N} points). Consider using sampling or sklearn.")
    
    try:
        # Step 1: Compute optimized distance matrix
        dist_matrix = compute_pairwise_distance_optimized(X, metric)
        if dist_matrix.size == 0:
            log_fail("Failed to compute distance matrix")
            return np.array([])
        
        # Step 2: Initialize optimized cluster tracker
        cluster_tracker = OptimizedClusterTracker(N)
        
        # Step 3: Initialize priority queue with all pairwise distances
        pq = PriorityQueue()
        linkage_type = {"single": 0, "complete": 1, "average": 2}[linkage]
        
        if verbose:
            log_info("Building initial priority queue...")
        
        # Pre-compute all initial distances and add to priority queue
        for i in range(N):
            for j in range(i + 1, N):
                dist = dist_matrix[i, j]
                pq.push(dist, i, j)
        
        linkage_matrix = []
        next_cluster_id = N
        
        if verbose:
            log_info(f"Starting optimized agglomeration ({N-1} merges required)")
        
        # Step 4: Main clustering loop with priority queue
        for step in range(N - 1):
            if verbose and step % max(1, (N-1) // 20) == 0:
                log_debug(f"Merge step {step+1}/{N-1}")
            
            # Find the next valid minimum distance pair
            min_dist, cluster_i, cluster_j = pq.pop_valid(cluster_tracker.active_clusters)
            
            if cluster_i is None:
                # Rebuild priority queue if needed
                if verbose:
                    log_debug("Rebuilding priority queue...")
                pq = PriorityQueue()
                active_clusters = list(cluster_tracker.active_clusters)
                
                for i in range(len(active_clusters)):
                    for j in range(i + 1, len(active_clusters)):
                        c1, c2 = active_clusters[i], active_clusters[j]
                        members_1 = np.array(cluster_tracker.get_cluster_members(c1))
                        members_2 = np.array(cluster_tracker.get_cluster_members(c2))
                        
                        dist = compute_linkage_distance_numba(
                            members_1, members_2, dist_matrix, linkage_type
                        )
                        pq.push(dist, c1, c2)
                
                min_dist, cluster_i, cluster_j = pq.pop_valid(cluster_tracker.active_clusters)
                
                if cluster_i is None:
                    log_fail(f"No valid clusters to merge at step {step}")
                    break
            
            # Get cluster information
            members_i = cluster_tracker.get_cluster_members(cluster_i)
            members_j = cluster_tracker.get_cluster_members(cluster_j)
            
            # Merge clusters
            cluster_tracker.union(cluster_i, cluster_j)
            merged_size = len(members_i) + len(members_j)
            
            # Add to linkage matrix
            linkage_matrix.append([cluster_i, cluster_j, min_dist, merged_size])
            
            # Update priority queue with new distances
            remaining_clusters = [c for c in cluster_tracker.active_clusters 
                                if c != cluster_i and c != cluster_j]
            
            # Find the new cluster root
            new_root = cluster_tracker.find(cluster_i)
            
            # Add distances from new merged cluster to all remaining clusters
            for other_cluster in remaining_clusters:
                other_members = np.array(cluster_tracker.get_cluster_members(other_cluster))
                merged_members = np.array(cluster_tracker.get_cluster_members(new_root))
                
                dist = compute_linkage_distance_numba(
                    merged_members, other_members, dist_matrix, linkage_type
                )
                pq.push(dist, new_root, other_cluster)
        
        linkage_array = np.array(linkage_matrix)
        
        if verbose:
            log_success("Optimized hierarchical clustering completed successfully")
            if len(linkage_array) > 0:
                log_info(f"Linkage matrix shape: {linkage_array.shape}")
                log_info(f"Distance range: [{np.min(linkage_array[:, 2]):.6f}, {np.max(linkage_array[:, 2]):.6f}]")
        
        return linkage_array
    
    except Exception as e:
        log_fail(f"Optimized hierarchical clustering failed: {e}")
        return np.array([])

# Additional memory-efficient version for very large datasets
def hierarchical_clustering_memory_efficient(X, linkage="average", metric="euclidean", 
                                           chunk_size=1000, verbose=False):
    """
    Memory-efficient version that processes distance matrix in chunks.
    """
    if verbose:
        log_info("Starting MEMORY-EFFICIENT hierarchical clustering")
    
    N = X.shape[0]
    
    if N <= chunk_size:
        return hierarchical_clustering_optimized(X, linkage, metric, verbose)
    
    # For very large datasets, use a different approach
    log_info(f"Using chunked processing for {N} points (chunk_size={chunk_size})")
    
    # This is a placeholder - you'd implement a more sophisticated chunked algorithm
    # For now, sample the data
    from sklearn.utils import resample
    
    log_warn(f"Dataset too large ({N}), sampling {chunk_size} points")
    X_sample, indices = resample(X, np.arange(N), n_samples=chunk_size, 
                                random_state=42, replace=False)
    
    return hierarchical_clustering_optimized(X_sample, linkage, metric, verbose)

