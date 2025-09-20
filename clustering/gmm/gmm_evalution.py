import os
import asyncio
import numpy as np
from dotenv import load_dotenv
from libsql_client import create_client
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
# from gmm import assign_labels  # Commented out as it's not used

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
    load_dotenv()
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    if not url or not auth_token:
        log_fail("Missing database credentials! Check your .env file.")
        return []
    
    log_info("Connecting to database...")
    
    try:
        async with create_client(url, auth_token=auth_token) as client:
            log_success("Database connection established.")
            
            # First, let's check if the tables exist
            check_tables_query = """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND (name='songs' OR name='song_clusters');
            """
            
            log_info("Checking if required tables exist...")
            try:
                table_result = await client.execute(check_tables_query)
                log_debug(f"Table check response: {table_result}")
                
                if hasattr(table_result, 'rows'):
                    existing_tables = [row[0] for row in table_result.rows]
                    log_info(f"Found tables: {existing_tables}")
                    
                    if 'songs' not in existing_tables:
                        log_fail("Table 'songs' not found in database.")
                        return []
                    if 'song_clusters' not in existing_tables:
                        log_fail("Table 'song_clusters' not found in database.")
                        return []
                else:
                    log_warn("Unexpected response format when checking tables.")
                    log_debug(f"Response type: {type(table_result)}")
                    log_debug(f"Response content: {table_result}")
                
            except Exception as e:
                log_warn(f"Error checking tables: {e}")
                log_info("Proceeding with main query anyway...")
            
            # Main query
            query = """
                SELECT s.song_id, s.feature_vector, c.gmm_cluster_id, c.gmm_responsibilities
                FROM songs s
                JOIN song_clusters c ON s.song_id = c.song_id
                WHERE c.algorithm = 'gmm'
            """
            
            log_info("Fetching GMM clustered songs from database...")
            
            try:
                result = await client.execute(query)
                log_debug(f"Query result type: {type(result)}")
                log_debug(f"Query result attributes: {dir(result)}")
                
                # Handle different response formats
                if hasattr(result, 'rows') and hasattr(result, 'columns'):
                    if not result.rows:
                        log_warn("No GMM clustered songs found in the database.")
                        return []
                    
                    columns = result.columns
                    data = [{col: row[i] for i, col in enumerate(columns)} for row in result.rows]
                    log_success(f"Fetched {len(data)} GMM clustered songs from the database.")
                    return data
                    
                elif isinstance(result, dict):
                    log_debug(f"Result keys: {result.keys()}")
                    
                    # Try different possible keys
                    rows_data = None
                    columns_data = None
                    
                    if 'rows' in result:
                        rows_data = result['rows']
                        columns_data = result.get('columns', [])
                    elif 'results' in result:
                        rows_data = result['results']
                        columns_data = result.get('columns', [])
                    else:
                        log_fail(f"Unexpected result format. Available keys: {list(result.keys())}")
                        return []
                    
                    if not rows_data:
                        log_warn("No GMM clustered songs found in the database.")
                        return []
                    
                    if not columns_data:
                        # If no column info, assume default order
                        columns_data = ['song_id', 'feature_vector', 'gmm_cluster_id', 'gmm_responsibilities']
                        log_warn("No column information found, using default column order.")
                    
                    data = [{col: row[i] for i, col in enumerate(columns_data)} for row in rows_data]
                    log_success(f"Fetched {len(data)} GMM clustered songs from the database.")
                    return data
                    
                else:
                    log_fail(f"Unexpected result type: {type(result)}")
                    log_debug(f"Result content: {result}")
                    return []
                    
            except Exception as e:
                log_fail(f"Error executing query: {e}")
                log_debug(f"Exception type: {type(e)}")
                return []
                
    except Exception as e:
        log_fail(f"Database connection error: {e}")
        log_debug(f"Exception type: {type(e)}")
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
        resp = parse_responsibilities(entry.get("gmm_responsibilities"))
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
    
    try:
        async with create_client(url, auth_token=auth_token) as client:
            log_info("Updating GMM cluster metrics in the database...")
            
            for cid, size in cluster_counts.items():
                query = """
                    UPDATE song_clusters
                    SET gmm_cluster_size = ?, gmm_silhouette_score = ?
                    WHERE algorithm = 'gmm' AND gmm_cluster_id = ?;
                """
                
                try:
                    await client.execute(query, [size, float(sil_score), int(cid)])
                    log_success(f"Updated GMM cluster {cid}: size={size}, silhouette={sil_score:.4f}")
                except Exception as e:
                    log_fail(f"Failed to update cluster {cid}: {e}")
            
            log_success("GMM cluster metrics update completed.")
            
    except Exception as e:
        log_fail(f"Database connection error during update: {e}")

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
    
    # Optionally update database (uncomment the line below)
    # await update_gmm_cluster_metrics_in_db(clustered_data, sil_score, cluster_counts)
    
    log_success("GMM evaluation completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())