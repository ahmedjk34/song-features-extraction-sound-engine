import asyncio
import json
import numpy as np
from clustering_util import prepare_data_pipeline
from libsql_client import create_client
import os
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from k_means.k_means import kmeans
from gmm.gmm import gmm_em, assign_labels


DB_URL = os.getenv("TURSO_DATABASE_URL")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")


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



async def insert_cluster_results(client, results):
    """
    results: list of dicts with keys matching your song_clusters table
    """
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


async def perform_clustering(reduced_vectors, verified_songs, method="kmeans"):
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
            # Use the reduced vector instead of original feature_vector
            reduced_vector = X[i]
            dist = np.linalg.norm(reduced_vector - centroids[label])
            
            # Calculate confidence based on relative distance within cluster
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
        log_info("Running custom GMM clustering...")
        # Use your custom GMM implementation
        means, covariances, weights, responsibilities, log_likelihood = gmm_em(
            X, n_clusters=4, max_iters=100, tol=1e-4, seed=42, verbose=True
        )
        labels = assign_labels(responsibilities)
        
        log_info(f"GMM converged with log-likelihood: {log_likelihood:.4f}")
        
        for song, label, responsibility in zip(verified_songs, labels, responsibilities):
            # Get the probabilities for this data point across all clusters
            probs = responsibility.tolist()
            
            results.append({
                "song_id": song['song_id'],
                "algorithm": "gmm",  # Changed to distinguish from sklearn version
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

    elif method == "hierarchical":
        log_info("Running Hierarchical clustering...")
        hier = AgglomerativeClustering(n_clusters=8).fit(X)
        for song, label in zip(verified_songs, hier.labels_):
            results.append({
                "song_id": song['song_id'],
                "algorithm": "hierarchical",
                "kmeans_cluster_id": None,
                "kmeans_distance": None,
                "gmm_cluster_id": None,
                "gmm_probabilities": None,
                "hier_level1_id": int(label // 4),  # Example: broad cluster
                "hier_level2_id": int(label),      # fine cluster
                "hier_distance": float(0),          # placeholder (requires linkage matrix)
                "dbscan_cluster_id": None,
                "dbscan_is_core": None,
                "confidence": 1.0
            })

    elif method == "dbscan":
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

    # Choose clustering method here
    method = "gmm"  # "kmeans", "gmm", "hierarchical", "dbscan"
    cluster_results = await perform_clustering(reduced_vectors, verified_songs, method=method)

    if not DB_URL or not AUTH_TOKEN:
        log_fail("Database credentials missing!")
        return

    async with create_client(DB_URL, auth_token=AUTH_TOKEN) as client:
        await insert_cluster_results(client, cluster_results)


if __name__ == "__main__":
    asyncio.run(main())