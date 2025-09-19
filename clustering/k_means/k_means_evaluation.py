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

async def get_kmeans_clusters_from_db():
    load_dotenv()
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    if not url or not auth_token:
        log_fail("Missing database credentials! Check your .env file.")
        return []
    log_info("Connecting to database...")
    async with create_client(url, auth_token=auth_token) as client:
        log_success("Database connection established.")
        query = """
            SELECT s.song_id, s.feature_vector, c.kmeans_cluster_id
            FROM songs s
            JOIN song_clusters c ON s.song_id = c.song_id
            WHERE c.algorithm = 'kmeans'
        """
        log_info("Fetching clustered songs from database...")
        result = await client.execute(query)
        if not result.rows:
            log_warn("No clustered songs found in the database.")
            return []
        columns = result.columns
        data = [{col: row[i] for i, col in enumerate(columns)} for row in result.rows]
        log_success(f"Fetched {len(data)} clustered songs from the database.")
        return data

def plot_results(X, labels, sil_score, cluster_counts):
    """Plot cluster size distribution and a 2D PCA scatter."""
    # ---- Bar chart: cluster sizes ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of sizes
    clusters = list(cluster_counts.keys())
    counts = [cluster_counts[c] for c in clusters]
    axes[0].bar(clusters, counts, color="skyblue", edgecolor="black")
    axes[0].set_title("Cluster Size Distribution")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Number of Songs")

    # ---- 2D scatter plot of points ----
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(X)
    else:
        reduced = X

    scatter = axes[1].scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=15)
    axes[1].set_title(f"K-Means Clusters (2D view) \nSilhouette Score: {sil_score:.4f}")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")
    legend1 = axes[1].legend(*scatter.legend_elements(), title="Cluster")
    axes[1].add_artist(legend1)

    plt.tight_layout()
    plt.show()

def evaluate_silhouette_and_sizes(clustered_data):
    X, labels = [], []
    for entry in clustered_data:
        fv = parse_feature_vector(entry.get("feature_vector"))
        cid = entry.get("kmeans_cluster_id")
        if fv is not None and cid is not None:
            X.append(fv)
            labels.append(int(cid))

    X = np.array(X)
    labels = np.array(labels)

    if len(set(labels)) < 2:
        log_warn("Silhouette score requires at least 2 clusters.")
        sil_score = float("nan")
    else:
        sil_score = silhouette_score(X, labels)

    log_info(f"Silhouette Score: {sil_score:.4f}")

    cluster_counts = Counter(labels)
    log_info("Cluster Size Distribution:")
    for cid, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cid}: {count} songs")

    # ---- Plot the results ----
    plot_results(X, labels, sil_score, cluster_counts)

    return sil_score, cluster_counts

async def main():
    clustered_data = await get_kmeans_clusters_from_db()
    if not clustered_data:
        log_fail("No k-means clustered data found.")
        return
    evaluate_silhouette_and_sizes(clustered_data)

if __name__ == "__main__":
    asyncio.run(main())
