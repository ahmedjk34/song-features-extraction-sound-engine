import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import scipy.cluster.hierarchy as sch
import warnings

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

def plot_dendrogram(linkage_matrix, labels=None, title="Hierarchical Clustering Dendrogram", 
                   max_points=100, figsize=(12, 6), save_path=None):
    """
    Visualize dendrogram with enhanced features and error handling.
    """
    try:
        log_info("Creating dendrogram plot")
        
        if linkage_matrix.size == 0:
            log_fail("Empty linkage matrix, cannot create dendrogram")
            return None
        
        # Validate linkage matrix
        if linkage_matrix.shape[1] != 4:
            log_fail(f"Invalid linkage matrix shape: {linkage_matrix.shape}")
            return None
        
        plt.figure(figsize=figsize)
        
        # Limit display for large datasets
        if linkage_matrix.shape[0] > max_points:
            log_warn(f"Large dataset ({linkage_matrix.shape[0]} merges), truncating display")
            truncate_mode = 'lastp'
            p = max_points
        else:
            truncate_mode = None
            p = None
        
        # Create dendrogram
        dendrogram_data = sch.dendrogram(
            linkage_matrix, 
            labels=labels,
            truncate_mode=truncate_mode,
            p=p,
            count_sort='ascending',
            distance_sort='ascending'
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Sample Index or (Cluster Size)", fontsize=12)
        plt.ylabel("Distance", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        max_dist = np.max(linkage_matrix[:, 2])
        min_dist = np.min(linkage_matrix[:, 2])
        plt.text(0.02, 0.98, f"Distance range: [{min_dist:.3f}, {max_dist:.3f}]", 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log_success(f"Dendrogram saved to {save_path}")
        
        plt.show()
        log_success("Dendrogram plot created successfully")
        return dendrogram_data
    
    except Exception as e:
        log_fail(f"Failed to create dendrogram: {e}")
        return None

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

def evaluate_clustering_quality(X, labels, linkage_matrix=None, verbose=False):
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

def plot_evaluation_summary(X, labels, linkage_matrix=None, save_path=None, figsize=(15, 10)):
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
        
        # 4. Dendrogram (if linkage matrix available)
        if linkage_matrix is not None and linkage_matrix.size > 0:
            try:
                sch.dendrogram(linkage_matrix, ax=axes[1,1], truncate_mode='lastp', p=30)
                axes[1,1].set_title("Dendrogram (Truncated)")
                axes[1,1].set_xlabel("Sample Index")
                axes[1,1].set_ylabel("Distance")
            except Exception as e:
                axes[1,1].text(0.5, 0.5, f"Dendrogram failed:\n{str(e)}", 
                              transform=axes[1,1].transAxes, ha='center', va='center')
                axes[1,1].set_title("Dendrogram (Failed)")
        else:
            axes[1,1].text(0.5, 0.5, "No linkage matrix\nprovided", 
                          transform=axes[1,1].transAxes, ha='center', va='center')
            axes[1,1].set_title("Dendrogram (Not Available)")
        
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