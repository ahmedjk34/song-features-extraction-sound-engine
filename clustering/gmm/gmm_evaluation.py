import os
import asyncio
import numpy as np
from dotenv import load_dotenv
from libsql_client import create_client
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt

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

def parse_feature_vector(vector_str):
    """Parse a feature vector from a string to a numpy array."""
    if isinstance(vector_str, (list, np.ndarray)):
        return np.array(vector_str)
    if vector_str is None:
        return None
    try:
        s = vector_str.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        values = [float(x) for x in s.split(",") if x.strip()]
        return np.array(values)
    except Exception as e:
        log_warn(f"Failed to parse feature vector: {vector_str} ({e})")
        return None

async def get_gmm_clusters_from_db():
    """Fetch GMM clustered songs from the database - simplified approach like K-means."""
    load_dotenv()
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    if not url or not auth_token:
        log_fail("Missing database credentials! Check your .env file.")
        return []

    log_info("Connecting to database...")

    async with create_client(url, auth_token=auth_token) as client:
        log_success("Database connection established.")

        # First check if GMM data exists
        check_query = """
            SELECT COUNT(*) as count
            FROM song_clusters 
            WHERE algorithm = 'gmm'
        """
        log_info("Checking for GMM data...")
        
        try:
            check_result = await client.execute(check_query)
            count = check_result.rows[0][0] if check_result.rows else 0
            log_info(f"Found {count} GMM cluster entries in database.")
            
            if count == 0:
                log_warn("No GMM clustered songs found in the database.")
                return []
        except Exception as e:
            log_fail(f"Error checking for GMM data: {e}")
            return []

        query = """
            SELECT s.song_id, s.feature_vector, c.gmm_cluster_id, c.gmm_probabilities
            FROM songs s
            JOIN song_clusters c ON s.song_id = c.song_id
            WHERE c.algorithm = 'gmm'
        """
        log_info("Fetching GMM clustered songs from database...")
        
        try:
            result = await client.execute(query)
            if not result.rows:
                log_warn("No GMM clustered songs found in the database.")
                return []
            columns = result.columns
            data = [{col: row[i] for i, col in enumerate(columns)} for row in result.rows]
            log_success(f"Fetched {len(data)} GMM clustered songs from the database.")
            return data
        except Exception as e:
            log_fail(f"Error executing query: {e}")
            return []

def parse_responsibilities(resp_str):
    """Parse a responsibility vector from a string to a numpy array."""
    if isinstance(resp_str, (list, np.ndarray)):
        return np.array(resp_str)
    if resp_str is None:
        return None
    try:
        s = resp_str.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        values = [float(x) for x in s.split(",") if x.strip()]
        return np.array(values)
    except Exception as e:
        log_warn(f"Failed to parse responsibilities: {resp_str} ({e})")
        return None

def plot_gmm_results(X, labels, sil_score, cluster_counts):
    """Plot cluster size distribution and a 2D PCA scatter for GMM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cluster size distribution
    clusters = list(cluster_counts.keys())
    counts = [cluster_counts[c] for c in clusters]
    axes[0].bar(clusters, counts, color="skyblue", edgecolor="black")
    axes[0].set_title("GMM Cluster Size Distribution")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Number of Songs")
    
    # 2D visualization using PCA if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(X)
        var_explained = pca.explained_variance_ratio_
        xlabel = f"PC1 ({var_explained[0]:.2%} var)"
        ylabel = f"PC2 ({var_explained[1]:.2%} var)"
    else:
        reduced = X
        xlabel = "Component 1"
        ylabel = "Component 2"
    
    scatter = axes[1].scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=15, alpha=0.7)
    axes[1].set_title(f"GMM Clusters (2D view)\nSilhouette Score: {sil_score:.4f}")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    
    # Add colorbar/legend
    legend1 = axes[1].legend(*scatter.legend_elements(), title="Cluster", loc='upper right')
    axes[1].add_artist(legend1)
    
    plt.tight_layout()
    plt.show()

def evaluate_gmm_silhouette_and_sizes(clustered_data):
    """Evaluate GMM clustering results."""
    X, labels, responsibilities_list = [], [], []
    
    log_info(f"Processing {len(clustered_data)} data entries...")
    
    for i, entry in enumerate(clustered_data):
        fv = parse_feature_vector(entry.get("feature_vector"))
        resp = parse_responsibilities(entry.get("gmm_probabilities"))  # Changed from gmm_responsibilities
        cid = entry.get("gmm_cluster_id")
        
        if fv is not None and resp is not None and cid is not None:
            X.append(fv)
            responsibilities_list.append(resp)
            labels.append(int(cid))
        else:
            log_warn(f"Entry {i} missing data: fv={fv is not None}, resp={resp is not None}, cid={cid}")
    
    if not X:
        log_fail("No valid data entries found for evaluation.")
        return float("nan"), {}
    
    X = np.array(X)
    labels = np.array(labels)
    responsibilities = np.vstack(responsibilities_list) if responsibilities_list else None
    
    log_info(f"Valid entries: {len(X)}, Feature dimensions: {X.shape[1]}")
    log_info(f"Unique clusters: {sorted(set(labels))}")
    
    # Calculate silhouette score
    if len(set(labels)) < 2:
        log_warn("Silhouette score requires at least 2 clusters.")
        sil_score = float("nan")
    else:
        sil_score = silhouette_score(X, labels)
    
    log_info(f"Silhouette Score: {sil_score:.4f}")
    
    # Cluster size analysis
    cluster_counts = Counter(labels)
    log_info("GMM Cluster Size Distribution:")
    for cid, count in sorted(cluster_counts.items()):
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {cid}: {count} songs ({percentage:.1f}%)")
    
    # Plot results
    try:
        plot_gmm_results(X, labels, sil_score, cluster_counts)
    except Exception as e:
        log_warn(f"Failed to create plots: {e}")
    
    return sil_score, cluster_counts

async def update_gmm_cluster_metrics_in_db(clustered_data, sil_score, cluster_counts):
    """Update GMM cluster metrics in the database."""
    load_dotenv()
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    async with create_client(url, auth_token=auth_token) as client:
        log_info("Updating GMM cluster metrics in the database...")
        
        for cid, size in cluster_counts.items():
            query = """
                UPDATE song_clusters
                SET gmm_cluster_size = ?, gmm_silhouette_score = ?
                WHERE algorithm = 'gmm' AND gmm_cluster_id = ?;
            """
            
            await client.execute(query, [size, float(sil_score), int(cid)])
            log_success(f"Updated GMM cluster {cid}: size={size}, silhouette={sil_score:.4f}")
        
        log_success("GMM cluster metrics update completed.")

async def main():
    """Main execution function."""
    log_info("Starting GMM evaluation...")
    
    # Fetch clustered data
    clustered_data = await get_gmm_clusters_from_db()
    if not clustered_data:
        log_fail("No GMM clustered data found. Exiting.")
        return
    
    # Evaluate clustering
    sil_score, cluster_counts = evaluate_gmm_silhouette_and_sizes(clustered_data)
    
    if not cluster_counts:
        log_fail("No valid clustering data for evaluation. Exiting.")
        return
    
    # Update database with metrics
    await update_gmm_cluster_metrics_in_db(clustered_data, sil_score, cluster_counts)
    
    log_success("GMM evaluation completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())