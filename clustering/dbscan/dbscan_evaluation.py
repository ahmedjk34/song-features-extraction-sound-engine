import os
import asyncio
import numpy as np
from dotenv import load_dotenv
from libsql_client import create_client
from sklearn.metrics import silhouette_score, adjusted_rand_score
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

async def get_dbscan_clusters_from_db():
    """Fetch DBSCAN clustered songs from database."""
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
            SELECT s.song_id, s.feature_vector, c.dbscan_cluster_id, c.is_core_point, c.is_noise_point
            FROM songs s
            JOIN song_clusters c ON s.song_id = c.song_id
            WHERE c.algorithm = 'dbscan'
        """
        log_info("Fetching DBSCAN clustered songs from database...")
        result = await client.execute(query)
        if not result.rows:
            log_warn("No DBSCAN clustered songs found in the database.")
            return []
        columns = result.columns
        data = [{col: row[i] for i, col in enumerate(columns)} for row in result.rows]
        log_success(f"Fetched {len(data)} DBSCAN clustered songs from the database.")
        return data

def plot_dbscan_results(X, labels, core_samples, sil_score, cluster_counts, eps, min_samples):
    """Plot DBSCAN cluster results with multiple visualizations."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ---- 1. Cluster size distribution ----
    noise_count = cluster_counts.get(-1, 0)
    regular_clusters = {k: v for k, v in cluster_counts.items() if k != -1}
    
    clusters = list(regular_clusters.keys()) + ([-1] if noise_count > 0 else [])
    counts = [regular_clusters[c] for c in regular_clusters.keys()] + ([noise_count] if noise_count > 0 else [])
    colors = ['skyblue'] * len(regular_clusters) + (['red'] if noise_count > 0 else [])
    
    bars = axes[0, 0].bar(clusters, counts, color=colors, edgecolor="black")
    axes[0, 0].set_title("DBSCAN Cluster Size Distribution")
    axes[0, 0].set_xlabel("Cluster ID (-1 = Noise)")
    axes[0, 0].set_ylabel("Number of Songs")
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                       f'{count}', ha='center', va='bottom')
    
    # ---- 2. 2D scatter plot of clusters ----
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(X)
        xlabel, ylabel = "PCA Component 1", "PCA Component 2"
    else:
        reduced = X
        xlabel, ylabel = "Feature 1", "Feature 2"
    
    # Plot points with different markers for core/border/noise
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points in black
            class_member_mask = (labels == k)
            xy = reduced[class_member_mask]
            axes[0, 1].scatter(xy[:, 0], xy[:, 1], s=20, c='black', marker='x', 
                             alpha=0.6, label='Noise')
        else:
            # Regular cluster points
            class_member_mask = (labels == k)
            xy = reduced[class_member_mask]
            
            # Separate core and border points
            core_mask = core_samples & class_member_mask
            border_mask = class_member_mask & ~core_samples & (labels != -1)
            
            if np.any(core_mask):
                axes[0, 1].scatter(reduced[core_mask, 0], reduced[core_mask, 1],
                                 s=30, c=[col], marker='o', alpha=0.8)
            if np.any(border_mask):
                axes[0, 1].scatter(reduced[border_mask, 0], reduced[border_mask, 1],
                                 s=15, c=[col], marker='s', alpha=0.6)
    
    axes[0, 1].set_title(f'DBSCAN Clusters (eps={eps}, min_samples={min_samples})\n'
                        f'Silhouette Score: {sil_score:.4f}')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel(ylabel)
    axes[0, 1].legend()
    
    # ---- 3. Point type distribution ----
    n_core = np.sum(core_samples)
    n_noise = np.sum(labels == -1)
    n_border = len(labels) - n_core - n_noise
    
    point_types = ['Core', 'Border', 'Noise']
    point_counts = [n_core, n_border, n_noise]
    point_colors = ['green', 'orange', 'red']
    
    wedges, texts, autotexts = axes[1, 0].pie(point_counts, labels=point_types, colors=point_colors,
                                              autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title("Point Type Distribution")
    
    # ---- 4. Cluster statistics ----
    axes[1, 1].axis('off')
    
    stats_text = f"""DBSCAN Clustering Results
    
Parameters:
• eps (radius): {eps}
• min_samples: {min_samples}

Results:
• Total points: {len(labels)}
• Number of clusters: {len(regular_clusters)}
• Core points: {n_core} ({100*n_core/len(labels):.1f}%)
• Border points: {n_border} ({100*n_border/len(labels):.1f}%)
• Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)

Quality Metrics:
• Silhouette Score: {sil_score:.4f}
• Noise Ratio: {n_noise/len(labels):.3f}

Cluster Sizes:"""
    
    for cid in sorted(regular_clusters.keys()):
        stats_text += f"\n• Cluster {cid}: {regular_clusters[cid]} points"
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def evaluate_dbscan_silhouette_and_sizes(clustered_data):
    """Evaluate DBSCAN clustering quality and extract statistics."""
    X, labels, core_samples = [], [], []
    
    for entry in clustered_data:
        fv = parse_feature_vector(entry.get("feature_vector"))
        cid = entry.get("dbscan_cluster_id")
        is_core = entry.get("is_core_point", 0)
        
        if fv is not None and cid is not None:
            X.append(fv)
            labels.append(int(cid))
            core_samples.append(bool(is_core))
    
    X = np.array(X)
    labels = np.array(labels)
    core_samples = np.array(core_samples)
    
    # Calculate silhouette score (excluding noise points)
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) < 2 or len(set(labels[non_noise_mask])) < 2:
        log_warn("Silhouette score requires at least 2 clusters with non-noise points.")
        sil_score = float("nan")
    else:
        sil_score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
    
    log_info(f"Silhouette Score (non-noise): {sil_score:.4f}")
    
    # Count clusters and noise
    cluster_counts = Counter(labels)
    n_clusters = len([c for c in cluster_counts.keys() if c != -1])
    n_noise = cluster_counts.get(-1, 0)
    
    log_info(f"Number of clusters found: {n_clusters}")
    log_info(f"Number of noise points: {n_noise}")
    log_info("Cluster Size Distribution:")
    
    for cid, count in sorted(cluster_counts.items()):
        if cid == -1:
            print(f"  Noise: {count} songs")
        else:
            print(f"  Cluster {cid}: {count} songs")
    
    # Extract eps and min_samples from first entry (assuming they're the same for all)
    eps = clustered_data[0].get('eps', 'Unknown')
    min_samples = clustered_data[0].get('min_samples', 'Unknown')
    
    # Plot the results
    plot_dbscan_results(X, labels, core_samples, sil_score, cluster_counts, eps, min_samples)
    
    return sil_score, cluster_counts, n_clusters, n_noise

async def update_dbscan_metrics_in_db(clustered_data, sil_score, cluster_counts, n_clusters, n_noise):
    """Update DBSCAN cluster metrics in the database."""
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    async with create_client(url, auth_token=auth_token) as client:
        log_info("Updating DBSCAN cluster metrics in the database...")
        
        for cid, size in cluster_counts.items():
            query = """
                UPDATE song_clusters
                SET dbscan_cluster_size = ?, dbscan_silhouette_score = ?, 
                    dbscan_n_clusters = ?, dbscan_n_noise = ?
                WHERE algorithm = 'dbscan' AND dbscan_cluster_id = ?;
            """
            await client.execute(query, [size, float(sil_score) if not np.isnan(sil_score) else None, 
                                       n_clusters, n_noise, int(cid)])
            
            if cid == -1:
                log_success(f"Updated noise points: count={size}")
            else:
                log_success(f"Updated cluster {cid}: size={size}")
        
        log_success("All DBSCAN cluster metrics updated successfully.")

async def compare_with_kmeans():
    """Compare DBSCAN results with existing K-Means clustering."""
    log_info("Fetching K-Means results for comparison...")
    
    # Fetch K-Means data
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    async with create_client(url, auth_token=auth_token) as client:
        # Get songs that have both K-Means and DBSCAN labels
        query = """
            SELECT s.song_id, s.feature_vector, 
                   k.kmeans_cluster_id, d.dbscan_cluster_id
            FROM songs s
            JOIN song_clusters k ON s.song_id = k.song_id AND k.algorithm = 'kmeans'
            JOIN song_clusters d ON s.song_id = d.song_id AND d.algorithm = 'dbscan'
        """
        result = await client.execute(query)
        
        if not result.rows:
            log_warn("No songs found with both K-Means and DBSCAN labels.")
            return
        
        columns = result.columns
        comparison_data = [{col: row[i] for i, col in enumerate(columns)} for row in result.rows]
        
        log_info(f"Found {len(comparison_data)} songs with both clustering results.")
        
        # Extract labels for comparison
        kmeans_labels = []
        dbscan_labels = []
        
        for entry in comparison_data:
            kmeans_labels.append(int(entry['kmeans_cluster_id']))
            dbscan_labels.append(int(entry['dbscan_cluster_id']))
        
        # Calculate Adjusted Rand Index
        ari = adjusted_rand_score(kmeans_labels, dbscan_labels)
        log_info(f"Adjusted Rand Index (K-Means vs DBSCAN): {ari:.4f}")
        
        # Print some statistics
        kmeans_clusters = len(set(kmeans_labels))
        dbscan_clusters = len([c for c in set(dbscan_labels) if c != -1])
        dbscan_noise = dbscan_labels.count(-1)
        
        print(f"\nClustering Comparison:")
        print(f"  K-Means clusters: {kmeans_clusters}")
        print(f"  DBSCAN clusters: {dbscan_clusters}")
        print(f"  DBSCAN noise points: {dbscan_noise}")
        print(f"  Agreement (ARI): {ari:.4f}")

async def main():
    """Main evaluation pipeline for DBSCAN results."""
    clustered_data = await get_dbscan_clusters_from_db()
    if not clustered_data:
        log_fail("No DBSCAN clustered data found.")
        return
    
    # Evaluate clustering quality
    sil_score, cluster_counts, n_clusters, n_noise = evaluate_dbscan_silhouette_and_sizes(clustered_data)
    
    # Update database with computed metrics
    await update_dbscan_metrics_in_db(clustered_data, sil_score, cluster_counts, n_clusters, n_noise)
    
    # Compare with K-Means if available
    await compare_with_kmeans()

if __name__ == "__main__":
    asyncio.run(main())