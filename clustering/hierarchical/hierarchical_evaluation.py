import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import scipy.cluster.hierarchy as sch
import warnings

from dotenv import load_dotenv
from libsql_client import create_client

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

def safe_pca_transform(X, n_components=2, verbose=False):
    """Safely apply PCA transformation with validation."""
    try:
        if X.shape[1] <= n_components:
            if verbose:
                log_info(f"Data already has {X.shape[1]} dimensions, no PCA needed")
            return X
        
        # Check for sufficient variance
        if np.allclose(X, X[0]):
            log_warn("Data has no variance, cannot apply PCA")
            return np.random.randn(X.shape[0], n_components) * 0.01
        
        pca = PCA(n_components=n_components, random_state=42)
        X_transformed = pca.fit_transform(X)
        
        if verbose:
            var_explained = pca.explained_variance_ratio_
            log_info(f"PCA explained variance: {var_explained}")
        
        return X_transformed, pca.explained_variance_ratio_
    
    except Exception as e:
        log_warn(f"PCA transformation failed: {e}, using random projection")
        return np.random.randn(X.shape[0], n_components) * np.std(X), np.array([0.5, 0.5])

def plot_cluster_assignments(X, flat_labels, title="Hierarchical Clustering Results", 
                           figsize=(10, 8), save_path=None, show_centroids=True):
    """
    Visualize clusters in 2D via PCA projection with enhanced features.
    """
    try:
        log_info("Creating cluster assignment plot")
        
        # Input validation
        if len(X) != len(flat_labels):
            log_fail(f"Mismatched dimensions: X={len(X)}, labels={len(flat_labels)}")
            return None
        
        if len(X) == 0:
            log_fail("Empty data provided")
            return None
        
        # Apply PCA if needed
        if X.shape[1] > 2:
            X_2d, var_explained = safe_pca_transform(X, n_components=2, verbose=True)
            xlabel = f"PC1 ({var_explained[0]:.1%} variance)"
            ylabel = f"PC2 ({var_explained[1]:.1%} variance)"
        else:
            X_2d = X
            xlabel = "Component 1"
            ylabel = "Component 2"
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_labels = np.unique(flat_labels)
        n_clusters = len(unique_labels)
        
        # Choose colormap
        if n_clusters <= 10:
            cmap = plt.cm.tab10
        else:
            cmap = plt.cm.viridis
        
        # Plot points
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=flat_labels, cmap=cmap, 
                           s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plot cluster centroids
        if show_centroids and n_clusters < 50:  # Avoid cluttering for many clusters
            for label in unique_labels:
                mask = flat_labels == label
                if np.sum(mask) > 0:
                    centroid = np.mean(X_2d[mask], axis=0)
                    ax.plot(centroid[0], centroid[1], 'kx', markersize=10, markeredgewidth=3)
                    ax.annotate(f'C{label}', (centroid[0], centroid[1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        if n_clusters > 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster ID', fontsize=12)
            cbar.set_ticks(unique_labels)
        
        # Add cluster statistics
        cluster_sizes = [np.sum(flat_labels == label) for label in unique_labels]
        stats_text = f"Clusters: {n_clusters}\nSizes: {cluster_sizes[:5]}"
        if len(cluster_sizes) > 5:
            stats_text += "..."
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
               verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log_success(f"Cluster plot saved to {save_path}")
        
        plt.show()
        log_success("Cluster assignment plot created successfully")
        return fig
    
    except Exception as e:
        log_fail(f"Failed to create cluster plot: {e}")
        return None

def cluster_size_metrics(flat_labels, verbose=False):
    """
    Compute and display comprehensive cluster size metrics.
    """
    try:
        if verbose:
            log_info("Computing cluster size metrics")
        
        unique, counts = np.unique(flat_labels, return_counts=True)
        
        # Basic metrics
        metrics = {
            'cluster_count': len(unique),
            'total_points': len(flat_labels),
            'cluster_sizes': dict(zip(unique.astype(int), counts.astype(int))),
            'min_size': int(np.min(counts)),
            'max_size': int(np.max(counts)),
            'mean_size': float(np.mean(counts)),
            'std_size': float(np.std(counts)),
            'median_size': float(np.median(counts))
        }
        
        # Balance metrics
        expected_size = len(flat_labels) / len(unique)
        coefficient_of_variation = metrics['std_size'] / metrics['mean_size'] if metrics['mean_size'] > 0 else float('inf')
        balance_score = 1.0 / (1.0 + coefficient_of_variation)
        
        metrics['expected_size'] = float(expected_size)
        metrics['coefficient_of_variation'] = float(coefficient_of_variation)
        metrics['balance_score'] = float(balance_score)
        
        # Display results
        print(f"\n{Log.HEADER}=== CLUSTER SIZE ANALYSIS ==={Log.ENDC}")
        print(f"Number of clusters: {metrics['cluster_count']}")
        print(f"Total points: {metrics['total_points']}")
        print(f"Size statistics: min={metrics['min_size']}, max={metrics['max_size']}, mean={metrics['mean_size']:.1f}, std={metrics['std_size']:.1f}")
        print(f"Balance score: {metrics['balance_score']:.3f} (1.0 = perfectly balanced)")
        print(f"\nCluster sizes:")
        
        for cluster_id, size in sorted(metrics['cluster_sizes'].items()):
            percentage = (size / metrics['total_points']) * 100
            print(f"  Cluster {cluster_id}: {size:4d} points ({percentage:5.1f}%)")
        
        if verbose:
            log_success("Cluster size metrics computed successfully")
        
        return metrics
    
    except Exception as e:
        log_fail(f"Error computing cluster size metrics: {e}")
        return {}

def evaluate_clustering_quality(X, labels, verbose=False):
    """
    Comprehensive clustering quality evaluation.
    """
    try:
        if verbose:
            log_info("Evaluating clustering quality")
        
        metrics = {}
        
        # Basic validation
        if len(X) != len(labels):
            log_fail("Mismatched data and label dimensions")
            return {}
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Silhouette analysis
        if n_clusters > 1 and n_clusters < len(X):
            try:
                silhouette_avg = silhouette_score(X, labels)
                metrics['silhouette_score'] = float(silhouette_avg)
                
                # Per-cluster silhouette scores
                from sklearn.metrics import silhouette_samples
                sample_silhouette_values = silhouette_samples(X, labels)
                cluster_silhouette_scores = {}
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_silhouette_scores[int(label)] = float(np.mean(sample_silhouette_values[cluster_mask]))
                
                metrics['cluster_silhouette_scores'] = cluster_silhouette_scores
                
            except Exception as e:
                log_warn(f"Silhouette analysis failed: {e}")
                metrics['silhouette_score'] = None
        else:
            log_warn("Cannot compute silhouette score: need 2 ≤ clusters < n_samples")
            metrics['silhouette_score'] = None
        
        # Inertia (within-cluster sum of squares)
        try:
            total_inertia = 0.0
            cluster_inertias = {}
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_points = X[cluster_mask]
                
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    inertia = np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
                    cluster_inertias[int(label)] = float(inertia)
                    total_inertia += inertia
            
            metrics['total_inertia'] = float(total_inertia)
            metrics['cluster_inertias'] = cluster_inertias
            
        except Exception as e:
            log_warn(f"Inertia computation failed: {e}")
            metrics['total_inertia'] = None
        
        # Calinski-Harabasz Index (Variance Ratio Criterion)
        try:
            from sklearn.metrics import calinski_harabasz_score
            if n_clusters > 1:
                ch_score = calinski_harabasz_score(X, labels)
                metrics['calinski_harabasz_score'] = float(ch_score)
        except Exception as e:
            log_warn(f"Calinski-Harabasz score computation failed: {e}")
            metrics['calinski_harabasz_score'] = None
        
        # Davies-Bouldin Index
        try:
            from sklearn.metrics import davies_bouldin_score
            if n_clusters > 1:
                db_score = davies_bouldin_score(X, labels)
                metrics['davies_bouldin_score'] = float(db_score)
        except Exception as e:
            log_warn(f"Davies-Bouldin score computation failed: {e}")
            metrics['davies_bouldin_score'] = None
        
        # Display results
        print(f"\n{Log.HEADER}=== CLUSTERING QUALITY EVALUATION ==={Log.ENDC}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of points: {len(X)}")
        
        if metrics.get('silhouette_score') is not None:
            score = metrics['silhouette_score']
            quality = "Excellent" if score > 0.7 else "Good" if score > 0.5 else "Fair" if score > 0.25 else "Poor"
            print(f"Silhouette Score: {score:.4f} ({quality})")
        
        if metrics.get('calinski_harabasz_score') is not None:
            print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f} (higher is better)")
        
        if metrics.get('davies_bouldin_score') is not None:
            print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (lower is better)")
        
        if metrics.get('total_inertia') is not None:
            print(f"Total Inertia: {metrics['total_inertia']:.2f}")
        
        if verbose:
            log_success("Clustering quality evaluation completed")
        
        return metrics
    
    except Exception as e:
        log_fail(f"Clustering quality evaluation failed: {e}")
        return {}

def plot_evaluation_summary(X, labels, save_path=None, figsize=(15, 10)):
    """
    Create a comprehensive evaluation summary plot.
    """
    try:
        log_info("Creating evaluation summary plot")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Cluster visualization
        if X.shape[1] > 2:
            X_2d, var_explained = safe_pca_transform(X, n_components=2, verbose=False)
            xlabel = f"PC1 ({var_explained[0]:.1%})"
            ylabel = f"PC2 ({var_explained[1]:.1%})"
        else:
            X_2d = X
            xlabel, ylabel = "Component 1", "Component 2"
        
        scatter = axes[0,0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=20, alpha=0.7)
        axes[0,0].set_title("Cluster Assignments")
        axes[0,0].set_xlabel(xlabel)
        axes[0,0].set_ylabel(ylabel)
        
        # 2. Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        axes[0,1].bar(unique, counts, color='skyblue', edgecolor='black')
        axes[0,1].set_title("Cluster Size Distribution")
        axes[0,1].set_xlabel("Cluster ID")
        axes[0,1].set_ylabel("Number of Points")
        
        # 3. Silhouette analysis
        try:
            from sklearn.metrics import silhouette_samples, silhouette_score
            if len(unique) > 1:
                silhouette_avg = silhouette_score(X, labels)
                sample_silhouette_values = silhouette_samples(X, labels)
                
                y_lower = 10
                for i, label in enumerate(unique):
                    cluster_silhouette_values = sample_silhouette_values[labels == label]
                    cluster_silhouette_values.sort()
                    
                    size_cluster_i = cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    
                    color = plt.cm.tab10(i / len(unique))
                    axes[1,0].fill_betweenx(np.arange(y_lower, y_upper),
                                          0, cluster_silhouette_values,
                                          facecolor=color, edgecolor=color, alpha=0.7)
                    
                    axes[1,0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
                    y_lower = y_upper + 10
                
                axes[1,0].axvline(x=silhouette_avg, color="red", linestyle="--", 
                                label=f'Average Score: {silhouette_avg:.3f}')
                axes[1,0].set_title("Silhouette Analysis")
                axes[1,0].set_xlabel("Silhouette Coefficient Values")
                axes[1,0].set_ylabel("Cluster Label")
                axes[1,0].legend()
        except Exception as e:
            axes[1,0].text(0.5, 0.5, f"Silhouette plot failed:\n{str(e)}", 
                          transform=axes[1,0].transAxes, ha='center', va='center')
            axes[1,0].set_title("Silhouette Analysis (Failed)")
        
        # 4. Level comparison (hierarchical-specific)
        try:
            axes[1,1].text(0.5, 0.5, "Hierarchical Clustering\nEvaluation Complete", 
                          transform=axes[1,1].transAxes, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            axes[1,1].set_title("Hierarchical Analysis")
        except Exception as e:
            axes[1,1].text(0.5, 0.5, f"Analysis failed:\n{str(e)}", 
                          transform=axes[1,1].transAxes, ha='center', va='center')
            axes[1,1].set_title("Analysis (Failed)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log_success(f"Evaluation summary saved to {save_path}")
        
        plt.show()
        log_success("Evaluation summary plot created successfully")
        return fig
    
    except Exception as e:
        log_fail(f"Failed to create evaluation summary: {e}")
        return None

def parse_feature_vector(vector_str):
    """Parse a feature vector from DB string into numpy array."""
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

async def get_hierarchical_clusters_from_db():
    """
    Fetch hierarchical clustering results from the database.
    Uses the correct column names from your schema.
    """
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
            
            # Fixed query using correct column names from your schema
            query = """
                SELECT s.song_id, s.feature_vector, c.hier_level1_id, c.hier_level2_id, c.hier_distance
                FROM songs s
                JOIN song_clusters c ON s.song_id = c.song_id
                WHERE c.algorithm = 'hierarchical'
                AND c.hier_level1_id IS NOT NULL
            """
            
            log_info("Fetching hierarchical clustered songs from database...")
            result = await client.execute(query)
            
            if not result.rows:
                log_warn("No hierarchical clustered songs found in the database.")
                return []
                
            # Convert result to dictionary format
            columns = result.columns
            data = []
            for row in result.rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row[i]
                data.append(row_dict)
            
            log_success(f"Fetched {len(data)} hierarchical clustered songs from the database.")
            return data
            
    except Exception as e:
        log_fail(f"Database error: {e}")
        return []

async def update_hierarchical_metrics_in_db(clustered_data, level1_metrics, level2_metrics):
    """Update hierarchical clustering metrics in the database."""
    load_dotenv()
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    try:
        async with create_client(url, auth_token=auth_token) as client:
            log_info("Updating hierarchical metrics in the database...")
            
            # You may need to add columns to your schema first:
            # ALTER TABLE song_clusters ADD COLUMN hier_level1_silhouette REAL;
            # ALTER TABLE song_clusters ADD COLUMN hier_level2_silhouette REAL;
            
            # Update metrics for each song
            for entry in clustered_data:
                song_id = entry.get('song_id')
                level1_id = entry.get('hier_level1_id')
                level2_id = entry.get('hier_level2_id')
                
                level1_sil = level1_metrics.get('silhouette_score')
                level2_sil = level2_metrics.get('silhouette_score')
                
                # This assumes you add silhouette columns to your schema
                update_query = """
                    UPDATE song_clusters 
                    SET hier_level1_silhouette = ?, hier_level2_silhouette = ?
                    WHERE song_id = ? AND algorithm = 'hierarchical'
                """
                
                await client.execute(update_query, [level1_sil, level2_sil, song_id])
            
            log_success("Hierarchical metrics updated successfully.")
            
    except Exception as e:
        log_fail(f"Failed to update metrics in database: {e}")

async def main():
    """Main evaluation pipeline for hierarchical clustering."""
    # Load data from DB
    clustered_data = await get_hierarchical_clusters_from_db()
    if not clustered_data:
        log_fail("No hierarchical clustered data found.")
        return
    
    log_info(f"Processing {len(clustered_data)} hierarchical clustered songs")
    
    # Prepare feature matrix and labels for both levels
    X = []
    level1_labels = []
    level2_labels = []
    
    for entry in clustered_data:
        fv = parse_feature_vector(entry.get("feature_vector"))
        l1_id = entry.get("hier_level1_id")
        l2_id = entry.get("hier_level2_id")
        
        if fv is not None and l1_id is not None:
            X.append(fv)
            level1_labels.append(int(l1_id))
            level2_labels.append(int(l2_id) if l2_id is not None else -1)
    
    X = np.array(X)
    level1_labels = np.array(level1_labels)
    level2_labels = np.array(level2_labels)
    
    if X.shape[0] == 0:
        log_fail("No valid feature vectors found.")
        return
    
    log_success(f"Successfully loaded {X.shape[0]} feature vectors with {X.shape[1]} dimensions")
    
    # Evaluate Level 1 clustering (broad clusters)
    print(f"\n{Log.HEADER}{'='*60}{Log.ENDC}")
    print(f"{Log.HEADER}EVALUATING LEVEL 1 HIERARCHICAL CLUSTERING (Broad){Log.ENDC}")
    print(f"{Log.HEADER}{'='*60}{Log.ENDC}")
    
    level1_metrics = evaluate_clustering_quality(X, level1_labels, verbose=True)
    level1_size_metrics = cluster_size_metrics(level1_labels, verbose=True)
    
    # Evaluate Level 2 clustering (fine clusters) if available
    if np.any(level2_labels != -1):
        print(f"\n{Log.HEADER}{'='*60}{Log.ENDC}")
        print(f"{Log.HEADER}EVALUATING LEVEL 2 HIERARCHICAL CLUSTERING (Fine){Log.ENDC}")
        print(f"{Log.HEADER}{'='*60}{Log.ENDC}")
        
        # Filter out entries without level 2 assignments
        level2_mask = level2_labels != -1
        if np.sum(level2_mask) > 1:
            X_level2 = X[level2_mask]
            level2_labels_filtered = level2_labels[level2_mask]
            
            level2_metrics = evaluate_clustering_quality(X_level2, level2_labels_filtered, verbose=True)
            level2_size_metrics = cluster_size_metrics(level2_labels_filtered, verbose=True)
        else:
            log_warn("Not enough Level 2 cluster assignments for evaluation")
            level2_metrics = {}
    else:
        log_info("No Level 2 cluster assignments found")
        level2_metrics = {}
    
    # Visualize results
    print(f"\n{Log.HEADER}{'='*40}{Log.ENDC}")
    print(f"{Log.HEADER}CREATING VISUALIZATIONS{Log.ENDC}")
    print(f"{Log.HEADER}{'='*40}{Log.ENDC}")
    
    # Plot Level 1 clusters
    plot_cluster_assignments(X, level1_labels, 
                           title="Hierarchical Clustering - Level 1 (Broad Clusters)")
    
    # Plot Level 2 clusters if available
    if np.any(level2_labels != -1):
        level2_mask = level2_labels != -1
        if np.sum(level2_mask) > 1:
            plot_cluster_assignments(X[level2_mask], level2_labels[level2_mask], 
                                   title="Hierarchical Clustering - Level 2 (Fine Clusters)")
    
    # Create comprehensive evaluation summary
    plot_evaluation_summary(X, level1_labels)
    
    # Update database with metrics (optional - requires schema modification)
    # await update_hierarchical_metrics_in_db(clustered_data, level1_metrics, level2_metrics)
    
    log_success("Hierarchical clustering evaluation pipeline completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())