import asyncio
import json
import numpy as np
from clustering_util import prepare_data_pipeline
from libsql_client import create_client
import os

from k_means.k_means import kmeans
from gmm.gmm import gmm_em

# UPDATED IMPORTS - Use optimized hierarchical clustering
from hierarchical.hierarchical import hierarchical_clustering_optimized
from hierarchical.hierarchical_extract import extract_clusters, get_merge_height_for_point

#
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances

DB_URL = os.getenv("TURSO_DATABASE_URL")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

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

async def insert_cluster_results(client, results):
    if not results:
        log_warn("No clustering results to insert.")
        return

    query = """
    INSERT INTO song_clusters (
        song_id, algorithm,
        kmeans_cluster_id, kmeans_distance,
        gmm_cluster_id, gmm_probabilities,
        hier_level1_id, hier_level2_id, hier_distance,
        dbscan_cluster_id, dbscan_is_core,
        confidence
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(song_id, algorithm) DO UPDATE SET
        kmeans_cluster_id=excluded.kmeans_cluster_id,
        kmeans_distance=excluded.kmeans_distance,
        gmm_cluster_id=excluded.gmm_cluster_id,
        gmm_probabilities=excluded.gmm_probabilities,
        hier_level1_id=excluded.hier_level1_id,
        hier_level2_id=excluded.hier_level2_id,
        hier_distance=excluded.hier_distance,
        dbscan_cluster_id=excluded.dbscan_cluster_id,
        dbscan_is_core=excluded.dbscan_is_core,
        confidence=excluded.confidence
    """
    params = [
        (
            r.get('song_id'),
            r.get('algorithm'),
            r.get('kmeans_cluster_id'),
            r.get('kmeans_distance'),
            r.get('gmm_cluster_id'),
            json.dumps(r.get('gmm_probabilities')) if r.get('gmm_probabilities') else None,
            r.get('hier_level1_id'),
            r.get('hier_level2_id'),
            r.get('hier_distance'),
            r.get('dbscan_cluster_id'),
            r.get('dbscan_is_core'),
            r.get('confidence')
        )
        for r in results
    ]

    for param_set in params:
        await client.execute(query, param_set)    
    
    log_success(f"Inserted/updated {len(results)} cluster results.")

async def perform_clustering(reduced_vectors, verified_songs, method="hierarchical"):
    """
    Perform clustering using one of the 4 algorithms.
    """
    results = []
    X = np.array(reduced_vectors)
    
    if method == "kmeans":
        log_info("Running K-Means clustering...")
        k = 5  # chosen via the elbow method
        centroids, labels, inertia = kmeans(X, n_clusters=k)
        for i, (song, label) in enumerate(zip(verified_songs, labels)):
            reduced_vector = X[i]
            dist = np.linalg.norm(reduced_vector - centroids[label])
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                max_dist_in_cluster = np.max(np.linalg.norm(cluster_points - centroids[label], axis=1))
                confidence = float(1 - (dist / max_dist_in_cluster)) if max_dist_in_cluster > 0 else 1.0
            else:
                confidence = 1.0
            results.append({
                "song_id": song['song_id'],
                "algorithm": "kmeans",
                "kmeans_cluster_id": int(label),
                "kmeans_distance": float(dist),
                "gmm_cluster_id": None,
                "gmm_probabilities": None,
                "hier_level1_id": None,
                "hier_level2_id": None,
                "hier_distance": None,
                "dbscan_cluster_id": None,
                "dbscan_is_core": None,
                "confidence": confidence
            })
            
    elif method == "gmm":
        log_info("Running GMM clustering...")
        
        try:
            means, covariances, weights, responsibilities, log_likelihood, labels = gmm_em(
                X, n_clusters=8, max_iters=100, tol=1e-6, seed=42, verbose=True
            )
            log_info(f"GMM converged with log-likelihood: {log_likelihood:.4f}")
            
            for song, label, responsibility in zip(verified_songs, labels, responsibilities):
                probs = responsibility.tolist()
                results.append({
                    "song_id": song['song_id'],
                    "algorithm": "gmm",
                    "kmeans_cluster_id": None,
                    "kmeans_distance": None,
                    "gmm_cluster_id": int(label),
                    "gmm_probabilities": probs,
                    "hier_level1_id": None,
                    "hier_level2_id": None,
                    "hier_distance": None,
                    "dbscan_cluster_id": None,
                    "dbscan_is_core": None,
                    "confidence": float(np.max(responsibility))
                })
                
        except Exception as e:
            log_fail(f"GMM clustering failed: {str(e)}")
            return []
            
    elif method == "hierarchical":
        log_info("Running OPTIMIZED scikit-learn Hierarchical clustering...")
        
        # Use PCA-transformed features directly (no additional preprocessing)
        X_features = X
        log_info(f"Using full dataset for optimized clustering (shape: {X_features.shape})")
        log_info(f"Feature range: [{X_features.min():.3f}, {X_features.max():.3f}], std: {X_features.std():.3f}")
        
        # Pre-compute COSINE distance matrix (your optimal metric for 50D audio features)
        log_info("Pre-computing cosine distance matrix for optimal clustering...")
        distance_matrix = pairwise_distances(X_features, metric='cosine')
        log_info(f"Cosine distance matrix computed: {distance_matrix.shape}")
        
        # Use your ACTUAL optimal parameters from comprehensive evaluation
        optimal_k_values = {
            'broad_level': 2,    # Keep for hierarchy
            'medium_level': 3,   # Your optimal k=3 
            'fine_level': 3      # Change from 8 to 3 (your optimal performs much better)
        }
        
        log_info(f"Using OPTIMAL parameters from evaluation: {optimal_k_values}")
        log_info("Winner: k=3, linkage=average, metric=cosine (Silhouette: 0.1587, Quality: FAIR)")
        
        # Fit all clustering levels with OPTIMAL settings
        clusterings = {}
        
        for level_name, n_clusters in optimal_k_values.items():
            actual_k = min(n_clusters, X_features.shape[0] - 1)
            
            if actual_k < 2:
                log_warn(f"Insufficient data for {level_name} clustering (k={actual_k})")
                continue
                
            try:
                log_info(f"Fitting {level_name} with k={actual_k} using optimal cosine distance...")
                
                # OPTIMAL AgglomerativeClustering settings from your evaluation
                clustering = AgglomerativeClustering(
                    n_clusters=actual_k,
                    linkage='average',               # Your optimal linkage (not complete!)
                    metric='precomputed',            # Use precomputed cosine distance matrix
                    compute_full_tree=True,          # Compute full dendrogram for accuracy
                    compute_distances=True           # Store distances for analysis
                )
                
                # Fit using precomputed COSINE distance matrix for optimal results
                labels = clustering.fit_predict(distance_matrix)
                
                # Calculate comprehensive quality metrics using COSINE metric
                if len(np.unique(labels)) > 1:
                    # Use COSINE distance for silhouette (matches clustering metric)
                    silhouette = silhouette_score(X_features, labels, metric='cosine')
                    calinski_harabasz = calinski_harabasz_score(X_features, labels)
                    davies_bouldin = davies_bouldin_score(X_features, labels)
                    
                    # Additional accuracy metrics
                    inertia = 0.0
                    cluster_sizes = []
                    for cluster_id in np.unique(labels):
                        cluster_mask = (labels == cluster_id)
                        cluster_points = X_features[cluster_mask]
                        if len(cluster_points) > 1:
                            cluster_center = np.mean(cluster_points, axis=0)
                            # Use cosine-based inertia calculation
                            cosine_distances = pairwise_distances(cluster_points, cluster_center.reshape(1, -1), metric='cosine').flatten()
                            cluster_inertia = np.sum(cosine_distances)
                            inertia += cluster_inertia
                        cluster_sizes.append(np.sum(cluster_mask))
                    
                    balance = min(cluster_sizes) / max(cluster_sizes) if cluster_sizes else 0.0
                    
                    log_info(f"{level_name} (k={actual_k}) OPTIMAL METRICS:")
                    log_info(f"  Silhouette (cosine): {silhouette:.4f}")
                    log_info(f"  Calinski-Harabasz: {calinski_harabasz:.2f}")
                    log_info(f"  Davies-Bouldin: {davies_bouldin:.4f}")
                    log_info(f"  Balance ratio: {balance:.4f}")
                    log_info(f"  Cluster sizes: {cluster_sizes}")
                    
                else:
                    silhouette = 0.0
                    calinski_harabasz = 0.0
                    davies_bouldin = float('inf')
                    balance = 0.0
                    
                clusterings[level_name] = {
                    'model': clustering,
                    'labels': labels,
                    'n_clusters': actual_k,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski_harabasz,
                    'davies_bouldin': davies_bouldin,
                    'balance': balance,
                    'distances': clustering.distances_ if hasattr(clustering, 'distances_') else None
                }
                
            except Exception as e:
                log_fail(f"Failed to create {level_name} clustering with k={actual_k}: {e}")
                continue
        
        if not clusterings:
            log_fail("All hierarchical clustering attempts failed")
            return []
        
        # Use the optimal levels from your evaluation
        broad_clustering = clusterings.get('broad_level')
        fine_clustering = clusterings.get('fine_level')  # Now using k=3 instead of k=8
        
        if not broad_clustering or not fine_clustering:
            log_fail("Could not create required clustering levels")
            return []
        
        broad_labels = broad_clustering['labels']
        fine_labels = fine_clustering['labels']
        
        log_success("OPTIMAL hierarchical clustering completed:")
        log_info(f"  Broad level (k=2): silhouette={broad_clustering['silhouette']:.4f}")
        log_info(f"  Fine level (k=3): silhouette={fine_clustering['silhouette']:.4f}")
        
        # Build results with COSINE-based confidence calculation
        for idx, song in enumerate(verified_songs):
            try:
                # Calculate per-point silhouette using COSINE distance
                fine_label = fine_labels[idx]
                point_features = X_features[idx].reshape(1, -1)
                
                # Calculate COSINE distances to all points in same cluster
                same_cluster_mask = (fine_labels == fine_label) & (np.arange(len(fine_labels)) != idx)
                if np.any(same_cluster_mask):
                    same_cluster_points = X_features[same_cluster_mask]
                    same_cluster_distances = pairwise_distances(point_features, same_cluster_points, metric='cosine').flatten()
                    avg_same_cluster_dist = np.mean(same_cluster_distances)
                else:
                    avg_same_cluster_dist = 0.0
                
                # Calculate COSINE distances to nearest other cluster
                other_clusters = np.unique(fine_labels[fine_labels != fine_label])
                min_other_cluster_dist = float('inf')
                
                for other_cluster in other_clusters:
                    other_cluster_mask = (fine_labels == other_cluster)
                    other_cluster_points = X_features[other_cluster_mask]
                    other_cluster_distances = pairwise_distances(point_features, other_cluster_points, metric='cosine').flatten()
                    avg_other_cluster_dist = np.mean(other_cluster_distances)
                    min_other_cluster_dist = min(min_other_cluster_dist, avg_other_cluster_dist)
                
                # Calculate point-specific silhouette score with COSINE
                if min_other_cluster_dist != float('inf') and max(avg_same_cluster_dist, min_other_cluster_dist) > 0:
                    point_silhouette = (min_other_cluster_dist - avg_same_cluster_dist) / max(min_other_cluster_dist, avg_same_cluster_dist)
                else:
                    point_silhouette = 0.0
                
                # Convert silhouette to confidence (shift and scale to [0.1, 1.0])
                confidence = (point_silhouette + 1.0) / 2.0  # Map [-1,1] to [0,1]
                confidence = max(0.1, min(1.0, confidence))
                
                # COSINE-based distance calculation
                hier_distance = avg_same_cluster_dist if avg_same_cluster_dist > 0 else 0.1
                
                results.append({
                    "song_id": song['song_id'],
                    "algorithm": "hierarchical",
                    "kmeans_cluster_id": None,
                    "kmeans_distance": None,
                    "gmm_cluster_id": None,
                    "gmm_probabilities": None,
                    "hier_level1_id": int(broad_labels[idx]),
                    "hier_level2_id": int(fine_labels[idx]),
                    "hier_distance": float(hier_distance),
                    "dbscan_cluster_id": None,
                    "dbscan_is_core": None,
                    "confidence": float(confidence)
                })
                
            except Exception as e:
                log_warn(f"Error processing song {idx}: {e}")
                # Fallback with reasonable defaults
                results.append({
                    "song_id": song['song_id'],
                    "algorithm": "hierarchical",
                    "kmeans_cluster_id": None,
                    "kmeans_distance": None,
                    "gmm_cluster_id": None,
                    "gmm_probabilities": None,
                    "hier_level1_id": int(broad_labels[idx]) if idx < len(broad_labels) else 0,
                    "hier_level2_id": int(fine_labels[idx]) if idx < len(fine_labels) else 0,
                    "hier_distance": 0.5,
                    "dbscan_cluster_id": None,
                    "dbscan_is_core": None,
                    "confidence": 0.5
                })
        
        log_success(f"OPTIMAL hierarchical clustering completed. Processed {len(results)} songs with cosine-based confidence calculation.")
        log_info(f"Applied winner configuration: k=3, average linkage, cosine metric (Quality: FAIR)") 
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        log_info("Running DBSCAN clustering...")
        dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)
        for song, label in zip(verified_songs, dbscan.labels_):
            is_core = label != -1 and len(dbscan.core_sample_indices_) > 0
            results.append({
                "song_id": song['song_id'],
                "algorithm": "dbscan",
                "kmeans_cluster_id": None,
                "kmeans_distance": None,
                "gmm_cluster_id": None,
                "gmm_probabilities": None,
                "hier_level1_id": None,
                "hier_level2_id": None,
                "hier_distance": None,
                "dbscan_cluster_id": int(label),
                "dbscan_is_core": bool(is_core),
                "confidence": 1.0 if label != -1 else 0.0
            })
    else:
        log_warn(f"Unknown clustering method: {method}")
        
    return results

async def main():
    """Run all clustering methods."""
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return
    
    methods = ["kmeans", "gmm", "hierarchical", "dbscan"]
    
    if not DB_URL or not AUTH_TOKEN:
        log_fail("Database credentials missing!")
        return
    
    async with create_client(DB_URL, auth_token=AUTH_TOKEN) as client:
        for method in methods:
            log_info(f"\n=== Running {method.upper()} clustering ===")
            try:
                cluster_results = await perform_clustering(reduced_vectors, verified_songs, method=method)
                if cluster_results:
                    await insert_cluster_results(client, cluster_results)
                    log_success(f"{method.upper()} clustering completed and saved.")
                else:
                    log_warn(f"No results from {method.upper()} clustering.")
            except Exception as e:
                log_fail(f"Error in {method.upper()} clustering: {str(e)}")
                continue

async def main_single_method(method="hierarchical"):
    """Run clustering with a single method."""
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return
    
    log_info(f"Running {method.upper()} clustering only...")
    cluster_results = await perform_clustering(reduced_vectors, verified_songs, method=method)
    
    if not DB_URL or not AUTH_TOKEN:
        log_fail("Database credentials missing!")
        return
    
    async with create_client(DB_URL, auth_token=AUTH_TOKEN) as client:
        await insert_cluster_results(client, cluster_results)

if __name__ == "__main__":
    # Choose which version to run:
    # asyncio.run(main())  # Run all methods
    asyncio.run(main_single_method("hierarchical"))  # Run optimized hierarchical only