import asyncio
import json
import numpy as np
from clustering_util import prepare_data_pipeline
from libsql_client import create_client
import os

from k_means.k_means import kmeans

# UPDATED IMPORT - Use the fixed GMM implementation
from gmm.gmm import gmm_em

from hierarchical.hierarchical import hierarchical_clustering
from hierarchical.hierarchical_extract import extract_clusters , get_merge_height_for_point

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
        k = 6  # chosen via the elbow method
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
        log_info("Running FIXED GMM clustering...")
        
        try:
            means, covariances, weights, responsibilities, log_likelihood, labels = gmm_em(
                X, n_clusters=8, max_iters=100, tol=1e-6, seed=42, verbose=True
            )
            log_info(f"GMM converged with log-likelihood: {log_likelihood:.4f}")
            
            # Build results using the fixed return values
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
        log_info("Running custom Hierarchical clustering...")
        
        # Use your custom implementation
        linkage_matrix = hierarchical_clustering(X, linkage="average", metric="euclidean")
        
        # Choose two levels for broad and fine clusters
        broad_level = 3
        fine_level = 8
        
        # Get cluster assignments for both levels
        hier_level1_labels = extract_clusters(linkage_matrix, broad_level, N=X.shape[0])
        hier_level2_labels = extract_clusters(linkage_matrix, fine_level, N=X.shape[0])
        
        # Build results with merge heights
        for idx, song in enumerate(verified_songs):
            # Get the merge distance at which this sample joined its fine cluster
            merge_height = get_merge_height_for_point(linkage_matrix, idx, fine_level, N=X.shape[0])
            
            # Calculate confidence based on merge height (lower height = more confident)
            # Normalize by max height in linkage matrix
            max_height = np.max(linkage_matrix[:, 2]) if len(linkage_matrix) > 0 else 1.0
            confidence = float(1.0 - (merge_height / max_height)) if max_height > 0 else 1.0
            confidence = max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
            
            results.append({
                "song_id": song['song_id'],
                "algorithm": "hierarchical",
                "kmeans_cluster_id": None,
                "kmeans_distance": None,
                "gmm_cluster_id": None,
                "gmm_probabilities": None,
                "hier_level1_id": int(hier_level1_labels[idx]),
                "hier_level2_id": int(hier_level2_labels[idx]),
                "hier_distance": float(merge_height),
                "dbscan_cluster_id": None,
                "dbscan_is_core": None,
                "confidence": confidence
            })
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
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return
    
    # UPDATED: Test all methods sequentially
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

# ALTERNATIVE: Run single method
async def main_single_method(method="gmm"):
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
    asyncio.run(main_single_method("gmm"))  # Run only GMM