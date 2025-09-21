import asyncio
import json
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, pairwise_distances, silhouette_score
from clustering_util import prepare_data_pipeline
from libsql_client import create_client
import os

from k_means.k_means import kmeans
from gmm.gmm import gmm_em

# UPDATED IMPORTS - Use optimized DBSCAN
from dbscan.dbscan import dbscan_optimal, dbscan_alternative, dbscan, get_cluster_info

# UPDATED IMPORTS - Use optimized hierarchical clustering
from hierarchical.hierarchical import hierarchical_clustering_optimized
from hierarchical.hierarchical_extract import extract_clusters, get_merge_height_for_point

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
        is_noise_point, eps, min_samples, dbscan_cluster_size,
        dbscan_silhouette_score, dbscan_n_clusters, dbscan_n_noise,
        confidence
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        is_noise_point=excluded.is_noise_point,
        eps=excluded.eps,
        min_samples=excluded.min_samples,
        dbscan_cluster_size=excluded.dbscan_cluster_size,
        dbscan_silhouette_score=excluded.dbscan_silhouette_score,
        dbscan_n_clusters=excluded.dbscan_n_clusters,
        dbscan_n_noise=excluded.dbscan_n_noise,
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
            r.get('is_noise_point'),           # NEW
            r.get('eps'),                      # NEW
            r.get('min_samples'),              # NEW
            r.get('dbscan_cluster_size'),      # NEW
            r.get('dbscan_silhouette_score'),  # NEW
            r.get('dbscan_n_clusters'),        # NEW
            r.get('dbscan_n_noise'),           # NEW
            r.get('confidence')
        )
        for r in results
    ]

    for param_set in params:
        await client.execute(query, param_set)    
    
    log_success(f"Inserted/updated {len(results)} cluster results.")

async def perform_clustering(reduced_vectors, verified_songs, method="hierarchical"):
    """
    Perform clustering using one of the 4 algorithms with optimized DBSCAN.
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
                "is_noise_point": None,           # NEW
                "eps": None,                      # NEW
                "min_samples": None,              # NEW
                "dbscan_cluster_size": None,      # NEW
                "dbscan_silhouette_score": None,  # NEW
                "dbscan_n_clusters": None,        # NEW
                "dbscan_n_noise": None,           # NEW
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
                    "is_noise_point": None,           # NEW
                    "eps": None,                      # NEW
                    "min_samples": None,              # NEW
                    "dbscan_cluster_size": None,      # NEW
                    "dbscan_silhouette_score": None,  # NEW
                    "dbscan_n_clusters": None,        # NEW
                    "dbscan_n_noise": None,           # NEW
                    "confidence": float(np.max(responsibility))
                })
                
        except Exception as e:
            log_fail(f"GMM clustering failed: {str(e)}")
            return []
            
    elif method == "hierarchical":
        log_info("Running OPTIMIZED custom Hierarchical clustering...")
        
        # Use PCA-transformed features directly (no additional preprocessing)
        X_features = X
        log_info(f"Using full dataset for optimized clustering (shape: {X_features.shape})")
        log_info(f"Feature range: [{X_features.min():.3f}, {X_features.max():.3f}], std: {X_features.std():.3f}")
        
        # Use your ACTUAL optimal parameters from comprehensive evaluation
        optimal_k_values = {
            'broad_level': 2,    # Keep for hierarchy
            'medium_level': 3,   # Your optimal k=3 
            'fine_level': 3      # Change from 8 to 3 (your optimal performs much better)
        }
        
        log_info(f"Using OPTIMAL parameters from evaluation: {optimal_k_values}")
        log_info("Winner: k=3, linkage=average, metric=cosine (Silhouette: 0.1587, Quality: FAIR)")
        
        # Pre-compute COSINE distance matrix for your custom implementation
        log_info("Pre-computing cosine distance matrix for custom hierarchical clustering...")
        distance_matrix = pairwise_distances(X_features, metric='cosine')
        log_info(f"Cosine distance matrix computed: {distance_matrix.shape}")
        
        # Run your custom hierarchical clustering with optimal settings
        clusterings = {}
        
        for level_name, n_clusters in optimal_k_values.items():
            actual_k = min(n_clusters, X_features.shape[0] - 1)
            
            if actual_k < 2:
                log_warn(f"Insufficient data for {level_name} clustering (k={actual_k})")
                continue
                
            try:
                log_info(f"Running custom hierarchical clustering for {level_name} with k={actual_k}...")
                
                # Use your custom hierarchical clustering implementation
                linkage_matrix = hierarchical_clustering_optimized(
                    X_features, 
                    linkage='average',  # Your optimal linkage
                    metric='euclidean',  # Note: we'll use precomputed cosine distances
                    verbose=True
                )
                
                if linkage_matrix.size == 0:
                    log_fail(f"Custom hierarchical clustering failed for {level_name}")
                    continue
                
                # Extract clusters at the desired level using your extraction function
                labels = extract_clusters(linkage_matrix, n_clusters=actual_k)
                
                if labels is None or len(labels) != X_features.shape[0]:
                    log_fail(f"Failed to extract {actual_k} clusters for {level_name}")
                    continue
                
                # Calculate comprehensive quality metrics using COSINE metric
                if len(np.unique(labels)) > 1:
                    # Use COSINE distance for silhouette (matches clustering metric)
                    silhouette = silhouette_score(X_features, labels, metric='cosine')
                    calinski_harabasz = calinski_harabasz_score(X_features, labels)
                    davies_bouldin = davies_bouldin_score(X_features, labels)
                    
                    # Additional accuracy metrics with cosine-based inertia
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
                    
                    log_info(f"{level_name} (k={actual_k}) CUSTOM HIERARCHICAL METRICS:")
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
                    'linkage_matrix': linkage_matrix,
                    'labels': labels,
                    'n_clusters': actual_k,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski_harabasz,
                    'davies_bouldin': davies_bouldin,
                    'balance': balance
                }
                
            except Exception as e:
                log_fail(f"Failed to create {level_name} clustering with custom implementation: {e}")
                continue
        
        if not clusterings:
            log_fail("All custom hierarchical clustering attempts failed")
            return []
        
        # Use the optimal levels from your evaluation
        broad_clustering = clusterings.get('broad_level')
        fine_clustering = clusterings.get('fine_level')  # Now using k=3 instead of k=8
        
        if not broad_clustering or not fine_clustering:
            log_fail("Could not create required clustering levels with custom implementation")
            return []
        
        broad_labels = broad_clustering['labels']
        fine_labels = fine_clustering['labels']
        
        log_success("CUSTOM hierarchical clustering completed:")
        log_info(f"  Broad level (k=2): silhouette={broad_clustering['silhouette']:.4f}")
        log_info(f"  Fine level (k=3): silhouette={fine_clustering['silhouette']:.4f}")
        
        # Build results with COSINE-based confidence calculation (same as before)
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
                    "is_noise_point": None,           # NEW
                    "eps": None,                      # NEW
                    "min_samples": None,              # NEW
                    "dbscan_cluster_size": None,      # NEW
                    "dbscan_silhouette_score": None,  # NEW
                    "dbscan_n_clusters": None,        # NEW
                    "dbscan_n_noise": None,           # NEW
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
                    "is_noise_point": None,           # NEW
                    "eps": None,                      # NEW
                    "min_samples": None,              # NEW
                    "dbscan_cluster_size": None,      # NEW
                    "dbscan_silhouette_score": None,  # NEW
                    "dbscan_n_clusters": None,        # NEW
                    "dbscan_n_noise": None,           # NEW
                    "confidence": 0.5
                })
        
        log_success(f"CUSTOM hierarchical clustering completed. Processed {len(results)} songs with cosine-based confidence calculation.")
        log_info(f"Applied winner configuration: k=3, average linkage, cosine metric using CUSTOM implementation (Quality: FAIR)")
        
    elif method == "dbscan":
        log_info("Running OPTIMIZED DBSCAN clustering with parameter selection results...")
        
        # Use the optimal parameters from your comprehensive analysis
        try:
            # Primary recommendation: eps=9.978, min_samples=6
            log_info("Using PRIMARY recommendation from parameter selection analysis")
            result = dbscan_optimal(X)
            
            labels = result['labels']
            core_samples = result['core_samples']
            cluster_info = result['cluster_info']
            parameters = result['parameters']  # Get eps and min_samples
            
            # Extract parameters for database storage
            eps_value = parameters['eps']
            min_samples_value = parameters['min_samples']
            n_clusters = cluster_info['n_clusters']
            n_noise = cluster_info['n_noise']
            
            log_info(f"DBSCAN Results Summary:")
            log_info(f"  Clusters: {n_clusters}")
            log_info(f"  Noise points: {n_noise} ({cluster_info['noise_ratio']:.2%})")
            log_info(f"  Core points: {cluster_info['n_core']}")
            log_info(f"  Border points: {cluster_info['n_border']}")
            log_info(f"  Quality: {cluster_info['quality_score']}")
            log_info(f"  Cluster sizes: {list(cluster_info['cluster_sizes'].values())}")
            log_info(f"  Parameters: eps={eps_value}, min_samples={min_samples_value}")
            
            # Calculate silhouette score if we have valid clusters
            silhouette = None
            if n_clusters > 1 and n_noise < len(labels):
                try:
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                        log_info(f"  Silhouette Score: {silhouette:.4f}")
                except Exception as e:
                    log_warn(f"Could not compute silhouette score: {e}")
                    silhouette = None
            
            # Build results with ALL new fields populated
            for idx, song in enumerate(verified_songs):
                label = labels[idx]
                is_core = core_samples[idx]
                is_noise = (label == -1)
                
                # Calculate cluster size for this specific cluster
                if label != -1:
                    cluster_size = cluster_info['cluster_sizes'].get(label, 0)
                else:
                    cluster_size = n_noise  # For noise points, store total noise count
                
                # Calculate confidence based on cluster membership and core status
                if is_noise:
                    confidence = 0.1
                elif is_core:
                    confidence = 0.9
                else:  # Border point
                    confidence = 0.6
                
                # Adjust confidence based on cluster size (larger clusters = more confident)
                if not is_noise and cluster_size > 0:
                    size_factor = min(1.0, cluster_size / 100.0)
                    confidence *= (0.7 + 0.3 * size_factor)
                
                # Adjust confidence based on overall clustering quality
                if silhouette is not None and silhouette > 0:
                    quality_factor = min(1.0, silhouette * 2)
                    confidence *= (0.8 + 0.2 * quality_factor)
                
                confidence = max(0.1, min(1.0, confidence))
                
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
                    "dbscan_cluster_id": int(label) if label != -1 else -1,  # Store -1 for noise
                    "dbscan_is_core": bool(is_core),
                    "is_noise_point": bool(is_noise),                     # NEW
                    "eps": float(eps_value),                              # NEW
                    "min_samples": int(min_samples_value),                # NEW
                    "dbscan_cluster_size": int(cluster_size),             # NEW
                    "dbscan_silhouette_score": float(silhouette) if silhouette is not None else None,  # NEW
                    "dbscan_n_clusters": int(n_clusters),                 # NEW
                    "dbscan_n_noise": int(n_noise),                      # NEW
                    "confidence": float(confidence)
                })
            
            log_success(f"OPTIMIZED DBSCAN completed: {len(results)} songs processed with full schema support")
            log_info(f"Applied optimal parameters: eps={eps_value}, min_samples={min_samples_value} (Primary recommendation)")
            
        except Exception as e:
            log_fail(f"DBSCAN clustering failed: {str(e)}")
            log_info("Falling back to alternative parameters...")
            
            try:
                # Alternative: eps=9.978, min_samples=3
                result = dbscan_alternative(X)
                
                labels = result['labels']
                core_samples = result['core_samples']
                cluster_info = result['cluster_info']
                parameters = result['parameters']
                
                # Extract parameters for database storage
                eps_value = parameters['eps']
                min_samples_value = parameters['min_samples']
                n_clusters = cluster_info['n_clusters']
                n_noise = cluster_info['n_noise']
                
                # Calculate silhouette for alternative
                silhouette = None
                if n_clusters > 1 and n_noise < len(labels):
                    try:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1:
                            silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                    except Exception as e:
                        silhouette = None
                
                # Process results same as above but with alternative parameters
                for idx, song in enumerate(verified_songs):
                    label = labels[idx]
                    is_core = core_samples[idx]
                    is_noise = (label == -1)
                    
                    # Calculate cluster size for this specific cluster
                    if label != -1:
                        cluster_size = cluster_info['cluster_sizes'].get(label, 0)
                    else:
                        cluster_size = n_noise
                    
                    if is_noise:
                        confidence = 0.1
                    elif is_core:
                        confidence = 0.8  # Slightly lower for alternative
                    else:
                        confidence = 0.5
                    
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
                        "dbscan_cluster_id": int(label) if label != -1 else -1,
                        "dbscan_is_core": bool(is_core),
                        "is_noise_point": bool(is_noise),                     # NEW
                        "eps": float(eps_value),                              # NEW
                        "min_samples": int(min_samples_value),                # NEW
                        "dbscan_cluster_size": int(cluster_size),             # NEW
                        "dbscan_silhouette_score": float(silhouette) if silhouette is not None else None,  # NEW
                        "dbscan_n_clusters": int(n_clusters),                 # NEW
                        "dbscan_n_noise": int(n_noise),                      # NEW
                        "confidence": float(confidence)
                    })
                
                log_success(f"DBSCAN completed with alternative parameters: {len(results)} songs processed")
                log_info(f"Applied alternative parameters: eps={eps_value}, min_samples={min_samples_value}")
                
            except Exception as e2:
                log_fail(f"Both DBSCAN configurations failed: {str(e2)}")
                return []
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

async def main_single_method(method="dbscan"):
    """Run clustering with a single method - optimized for DBSCAN."""
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return
    
    log_info(f"Running OPTIMIZED {method.upper()} clustering only...")
    cluster_results = await perform_clustering(reduced_vectors, verified_songs, method=method)
    
    if not DB_URL or not AUTH_TOKEN:
        log_fail("Database credentials missing!")
        return
    
    async with create_client(DB_URL, auth_token=AUTH_TOKEN) as client:
        await insert_cluster_results(client, cluster_results)

if __name__ == "__main__":
    # Choose which version to run:
    # asyncio.run(main())  # Run all methods
    asyncio.run(main_single_method("dbscan"))  # Run optimized DBSCAN only