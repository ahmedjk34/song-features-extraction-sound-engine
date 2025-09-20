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

class UnionFind:
    """Union-Find data structure with path compression and validation."""
    
    def __init__(self, n):
        if n <= 0:
            raise ValueError("UnionFind requires positive size")
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n
    
    def find(self, x):
        """Find with path compression and bounds checking."""
        if x < 0 or x >= self.n:
            log_warn(f"Invalid index {x} for UnionFind of size {self.n}")
            return x % self.n  # Fallback
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank with validation."""
        if x < 0 or x >= self.n or y < 0 or y >= self.n:
            log_warn(f"Invalid indices for union: {x}, {y}")
            return False
        
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True
    
    def get_components(self):
        """Get all connected components."""
        components = {}
        for i in range(self.n):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        return components

def extract_clusters(linkage_matrix, num_clusters, N=None, verbose=False):
    """
    Extract flat cluster labels for given number of clusters with robust validation.
    """
    if verbose:
        log_info(f"Extracting {num_clusters} clusters from linkage matrix")
    
    # Input validation
    if linkage_matrix.size == 0:
        log_fail("Empty linkage matrix provided")
        return np.array([])
    
    if linkage_matrix.ndim != 2 or linkage_matrix.shape[1] != 4:
        log_fail(f"Invalid linkage matrix shape: {linkage_matrix.shape}")
        return np.array([])
    
    # Determine N if not provided
    if N is None:
        N = linkage_matrix.shape[0] + 1
        log_debug(f"Inferred N={N} from linkage matrix")
    
    if N <= 0:
        log_fail(f"Invalid number of points: {N}")
        return np.array([])
    
    if num_clusters <= 0:
        log_fail(f"Invalid number of clusters: {num_clusters}")
        return np.arange(N)  # Each point in its own cluster
    
    if num_clusters >= N:
        log_info(f"Requested {num_clusters} clusters >= {N} points, returning individual clusters")
        return np.arange(N)
    
    try:
        # Initialize Union-Find structure
        uf = UnionFind(N)
        
        # Apply merges to get desired number of clusters
        merges_to_apply = N - num_clusters
        max_merges = min(merges_to_apply, len(linkage_matrix))
        
        if verbose:
            log_info(f"Applying {max_merges} merges out of {len(linkage_matrix)} available")
        
        merges_applied = 0
        for i in range(max_merges):
            try:
                cluster_a = int(linkage_matrix[i][0])
                cluster_b = int(linkage_matrix[i][1])
                
                # Only merge original points (indices < N)
                if cluster_a < N and cluster_b < N:
                    if uf.union(cluster_a, cluster_b):
                        merges_applied += 1
                    
            except (ValueError, IndexError) as e:
                log_warn(f"Error processing merge {i}: {e}")
                continue
        
        if verbose:
            log_debug(f"Successfully applied {merges_applied} merges")
        
        # Get final cluster assignments
        components = uf.get_components()
        labels = np.zeros(N, dtype=int)
        
        cluster_id = 0
        for root, members in components.items():
            for member in members:
                labels[member] = cluster_id
            cluster_id += 1
        
        # Validate results
        actual_clusters = len(np.unique(labels))
        if actual_clusters != num_clusters:
            log_warn(f"Expected {num_clusters} clusters, got {actual_clusters}")
        
        if verbose:
            log_success(f"Extracted {actual_clusters} clusters successfully")
            cluster_sizes = np.bincount(labels)
            log_debug(f"Cluster sizes: {cluster_sizes}")
        
        return labels
    
    except Exception as e:
        log_fail(f"Cluster extraction failed: {e}")
        return np.arange(N)  # Fallback: each point in its own cluster

def assign_hierarchical_labels(linkage_matrix, levels, N=None, verbose=False):
    """
    Return a dict mapping each number of clusters to labels array.
    """
    if verbose:
        log_info(f"Assigning hierarchical labels for levels: {levels}")
    
    if not levels:
        log_warn("No levels provided")
        return {}
    
    try:
        results = {}
        for level in sorted(levels):
            if level <= 0:
                log_warn(f"Skipping invalid level: {level}")
                continue
                
            labels = extract_clusters(linkage_matrix, level, N, verbose=False)
            if labels.size > 0:
                results[level] = labels
                if verbose:
                    log_debug(f"Level {level}: {len(np.unique(labels))} clusters")
            else:
                log_warn(f"Failed to extract clusters for level {level}")
        
        if verbose:
            log_success(f"Successfully assigned labels for {len(results)} levels")
        
        return results
    
    except Exception as e:
        log_fail(f"Hierarchical label assignment failed: {e}")
        return {}

def get_merge_height_for_point(linkage_matrix, point_idx, target_num_clusters, N, verbose=False):
    """
    Get the height (distance) at which a point joined its cluster at target_num_clusters level.
    """
    if verbose:
        log_debug(f"Getting merge height for point {point_idx} at {target_num_clusters} clusters")
    
    # Input validation
    if linkage_matrix.size == 0:
        log_warn("Empty linkage matrix")
        return 0.0
    
    if point_idx < 0 or point_idx >= N:
        log_warn(f"Invalid point index {point_idx} for N={N}")
        return 0.0
    
    if target_num_clusters >= N:
        if verbose:
            log_debug("Point never merged at this level")
        return 0.0
    
    try:
        merges_to_apply = N - target_num_clusters
        max_merges = min(merges_to_apply, len(linkage_matrix))
        
        # Track merge history for this point
        uf = UnionFind(N)
        merge_heights = [0.0] * N
        
        for i in range(max_merges):
            try:
                cluster_a = int(linkage_matrix[i][0])
                cluster_b = int(linkage_matrix[i][1])
                height = float(linkage_matrix[i][2])
                
                if cluster_a < N and cluster_b < N:
                    # Check if our point is affected by this merge
                    root_a = uf.find(cluster_a)
                    root_b = uf.find(cluster_b)
                    point_root = uf.find(point_idx)
                    
                    # If our point is in either cluster being merged, update its height
                    if point_root == root_a or point_root == root_b:
                        merge_heights[point_idx] = height
                    
                    # Perform the union
                    uf.union(cluster_a, cluster_b)
                    
            except (ValueError, IndexError) as e:
                log_warn(f"Error processing merge {i} for height calculation: {e}")
                continue
        
        result = merge_heights[point_idx]
        
        # Validation
        if not np.isfinite(result) or result < 0:
            log_warn(f"Invalid merge height {result}, returning 0.0")
            return 0.0
        
        if verbose:
            log_debug(f"Point {point_idx} merge height: {result:.6f}")
        
        return float(result)
    
    except Exception as e:
        log_fail(f"Error calculating merge height: {e}")
        return 0.0

def compute_cluster_statistics(labels, linkage_matrix=None, verbose=False):
    """
    Compute comprehensive statistics about cluster assignments.
    """
    if verbose:
        log_info("Computing cluster statistics")
    
    try:
        stats = {}
        
        # Basic statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        stats['num_clusters'] = len(unique_labels)
        stats['cluster_sizes'] = dict(zip(unique_labels.astype(int), counts.astype(int)))
        stats['total_points'] = len(labels)
        
        # Size statistics
        stats['min_cluster_size'] = int(np.min(counts))
        stats['max_cluster_size'] = int(np.max(counts))
        stats['mean_cluster_size'] = float(np.mean(counts))
        stats['std_cluster_size'] = float(np.std(counts))
        
        # Balance metrics
        expected_size = len(labels) / len(unique_labels)
        balance_score = 1.0 - (np.std(counts) / expected_size) if expected_size > 0 else 0.0
        stats['balance_score'] = float(np.clip(balance_score, 0.0, 1.0))
        
        if verbose:
            log_success("Cluster statistics computed successfully")
            log_info(f"Clusters: {stats['num_clusters']}, Balance: {stats['balance_score']:.3f}")
        
        return stats
    
    except Exception as e:
        log_fail(f"Error computing cluster statistics: {e}")
        return {}

def validate_cluster_labels(labels, expected_range=None, verbose=False):
    """
    Validate cluster label assignments.
    """
    if verbose:
        log_info("Validating cluster labels")
    
    try:
        if len(labels) == 0:
            log_warn("Empty label array")
            return False
        
        # Check for valid label types
        if not np.issubdtype(labels.dtype, np.integer):
            log_warn(f"Labels should be integers, got {labels.dtype}")
        
        # Check for negative labels
        if np.any(labels < 0):
            log_warn("Found negative cluster labels")
            return False
        
        # Check label continuity (should be 0, 1, 2, ...)
        unique_labels = np.unique(labels)
        expected_labels = np.arange(len(unique_labels))
        if not np.array_equal(np.sort(unique_labels), expected_labels):
            log_warn("Cluster labels are not continuous")
        
        # Check expected range if provided
        if expected_range is not None:
            min_label, max_label = expected_range
            if np.min(labels) < min_label or np.max(labels) > max_label:
                log_warn(f"Labels outside expected range [{min_label}, {max_label}]")
                return False
        
        if verbose:
            log_success(f"Label validation passed for {len(labels)} points")
        
        return True
    
    except Exception as e:
        log_fail(f"Label validation failed: {e}")
        return False