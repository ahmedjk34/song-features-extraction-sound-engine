import sys
import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Add parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import your modules
from clustering_util import prepare_data_pipeline
from hierarchical import hierarchical_clustering, log_info, log_success, log_warn, log_fail, log_debug
from hierarchical_extract import extract_clusters, compute_cluster_statistics

def evaluate_clustering_for_k(X, linkage_matrix, k, verbose=False):
    """
    Evaluate clustering quality for a specific number of clusters.
    """
    try:
        if verbose:
            log_debug(f"Evaluating k={k} clusters")
        
        # Extract cluster labels
        labels = extract_clusters(linkage_matrix, k, N=X.shape[0], verbose=False)
        
        if len(labels) == 0:
            log_warn(f"Failed to extract clusters for k={k}")
            return None
        
        actual_clusters = len(np.unique(labels))
        if actual_clusters == 1:
            log_warn(f"k={k} resulted in only 1 cluster")
            return None
        
        metrics = {}
        metrics['k'] = k
        metrics['actual_clusters'] = actual_clusters
        
        # Silhouette score
        try:
            if actual_clusters > 1 and actual_clusters < len(X):
                sil_score = silhouette_score(X, labels)
                metrics['silhouette_score'] = float(sil_score)
            else:
                metrics['silhouette_score'] = None
        except Exception as e:
            log_warn(f"Silhouette computation failed for k={k}: {e}")
            metrics['silhouette_score'] = None
        
        # Calinski-Harabasz score
        try:
            if actual_clusters > 1:
                ch_score = calinski_harabasz_score(X, labels)
                metrics['calinski_harabasz_score'] = float(ch_score)
            else:
                metrics['calinski_harabasz_score'] = None
        except Exception as e:
            log_warn(f"Calinski-Harabasz computation failed for k={k}: {e}")
            metrics['calinski_harabasz_score'] = None
        
        # Davies-Bouldin score
        try:
            if actual_clusters > 1:
                db_score = davies_bouldin_score(X, labels)
                metrics['davies_bouldin_score'] = float(db_score)
            else:
                metrics['davies_bouldin_score'] = None
        except Exception as e:
            log_warn(f"Davies-Bouldin computation failed for k={k}: {e}")
            metrics['davies_bouldin_score'] = None
        
        # Inertia (within-cluster sum of squares)
        try:
            total_inertia = 0.0
            for label in np.unique(labels):
                cluster_mask = labels == label
                cluster_points = X[cluster_mask]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    inertia = np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
                    total_inertia += inertia
            metrics['inertia'] = float(total_inertia)
        except Exception as e:
            log_warn(f"Inertia computation failed for k={k}: {e}")
            metrics['inertia'] = None
        
        # Cluster balance
        try:
            cluster_stats = compute_cluster_statistics(labels, verbose=False)
            metrics['balance_score'] = cluster_stats.get('balance_score', None)
            metrics['cluster_sizes'] = cluster_stats.get('cluster_sizes', {})
        except Exception as e:
            log_warn(f"Balance computation failed for k={k}: {e}")
            metrics['balance_score'] = None
            metrics['cluster_sizes'] = {}
        
        if verbose:
            log_debug(f"k={k}: sil={metrics.get('silhouette_score', 'N/A'):.3f}, "
                     f"ch={metrics.get('calinski_harabasz_score', 'N/A'):.1f}")
        
        return metrics
    
    except Exception as e:
        log_fail(f"Evaluation failed for k={k}: {e}")
        return None

def hierarchical_model_selection(X, k_range=None, linkage="average", metric="euclidean", verbose=True):
    """
    Perform model selection for hierarchical clustering across different numbers of clusters.
    """
    if verbose:
        log_info("Starting hierarchical clustering model selection")
    
    # Set default k_range
    if k_range is None:
        k_range = range(2, min(21, X.shape[0]))
    
    if verbose:
        log_info(f"Testing k values: {list(k_range)}")
        log_info(f"Data shape: {X.shape}")
        log_info(f"Using {linkage} linkage with {metric} metric")
    
    # Perform hierarchical clustering once
    log_info("Computing hierarchical clustering...")
    linkage_matrix = hierarchical_clustering(X, linkage=linkage, metric=metric, verbose=verbose)
    
    if linkage_matrix.size == 0:
        log_fail("Hierarchical clustering failed")
        return None
    
    log_success(f"Linkage matrix computed: {linkage_matrix.shape}")
    
    # Evaluate each k value
    results = []
    for k in k_range:
        if k >= X.shape[0]:
            log_warn(f"Skipping k={k} (>= number of data points)")
            continue
        
        metrics = evaluate_clustering_for_k(X, linkage_matrix, k, verbose=False)
        if metrics is not None:
            results.append(metrics)
            if verbose:
                sil = metrics.get('silhouette_score')
                ch = metrics.get('calinski_harabasz_score')
                db = metrics.get('davies_bouldin_score')
                print(f"k={k:2d}: Silhouette={sil:.3f if sil else 'N/A':>6}, "
                      f"CH={ch:.1f if ch else 'N/A':>6}, "
                      f"DB={db:.3f if db else 'N/A':>6}")
    
    if not results:
        log_fail("No successful evaluations")
        return None
    
    log_success(f"Evaluated {len(results)} different k values")
    
    return {
        'results': results,
        'linkage_matrix': linkage_matrix,
        'linkage_type': linkage,
        'metric': metric
    }

def find_optimal_k(results, criterion='silhouette', verbose=True):
    """
    Find optimal number of clusters based on specified criterion.
    """
    if not results or 'results' not in results:
        log_fail("No results provided for optimal k selection")
        return None
    
    data = results['results']
    valid_results = []
    
    criterion_key = {
        'silhouette': 'silhouette_score',
        'calinski_harabasz': 'calinski_harabasz_score',
        'davies_bouldin': 'davies_bouldin_score',
        'inertia': 'inertia',
        'balance': 'balance_score'
    }.get(criterion)
    
    if criterion_key is None:
        log_fail(f"Unknown criterion: {criterion}")
        return None
    
    # Filter valid results
    for result in data:
        score = result.get(criterion_key)
        if score is not None and np.isfinite(score):
            valid_results.append(result)
    
    if not valid_results:
        log_warn(f"No valid results for criterion {criterion}")
        return None
    
    # Find optimal based on criterion
    if criterion in ['silhouette', 'calinski_harabasz', 'balance']:
        # Higher is better
        optimal = max(valid_results, key=lambda x: x[criterion_key])
        direction = "maximize"
    else:
        # Lower is better (davies_bouldin, inertia)
        optimal = min(valid_results, key=lambda x: x[criterion_key])
        direction = "minimize"
    
    if verbose:
        log_info(f"Optimal k by {criterion} ({direction}): k={optimal['k']}, "
                f"score={optimal[criterion_key]:.4f}")
    
    return optimal

def plot_model_selection_results(results, save_path=None, figsize=(15, 10)):
    """
    Plot comprehensive model selection results.
    """
    try:
        log_info("Creating model selection plots")
        
        if not results or 'results' not in results:
            log_fail("No results to plot")
            return None
        
        data = results['results']
        if not data:
            log_fail("Empty results data")
            return None
        
        # Extract data
        k_values = [r['k'] for r in data]
        sil_scores = [r.get('silhouette_score') for r in data]
        ch_scores = [r.get('calinski_harabasz_score') for r in data]
        db_scores = [r.get('davies_bouldin_score') for r in data]
        inertias = [r.get('inertia') for r in data]
        balance_scores = [r.get('balance_score') for r in data]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Silhouette Score
        valid_sil = [(k, s) for k, s in zip(k_values, sil_scores) if s is not None]
        if valid_sil:
            k_sil, sil_vals = zip(*valid_sil)
            axes[0,0].plot(k_sil, sil_vals, 'bo-', linewidth=2, markersize=6)
            axes[0,0].set_title('Silhouette Score (Higher is Better)')
            axes[0,0].set_xlabel('Number of Clusters (k)')
            axes[0,0].set_ylabel('Silhouette Score')
            axes[0,0].grid(True, alpha=0.3)
            
            # Mark optimal
            best_idx = np.argmax(sil_vals)
            axes[0,0].axvline(x=k_sil[best_idx], color='red', linestyle='--', alpha=0.7)
            axes[0,0].text(k_sil[best_idx], sil_vals[best_idx], f'  k={k_sil[best_idx]}', 
                          verticalalignment='bottom')
        
        # 2. Calinski-Harabasz Score
        valid_ch = [(k, s) for k, s in zip(k_values, ch_scores) if s is not None]
        if valid_ch:
            k_ch, ch_vals = zip(*valid_ch)
            axes[0,1].plot(k_ch, ch_vals, 'go-', linewidth=2, markersize=6)
            axes[0,1].set_title('Calinski-Harabasz Score (Higher is Better)')
            axes[0,1].set_xlabel('Number of Clusters (k)')
            axes[0,1].set_ylabel('CH Score')
            axes[0,1].grid(True, alpha=0.3)
            
            # Mark optimal
            best_idx = np.argmax(ch_vals)
            axes[0,1].axvline(x=k_ch[best_idx], color='red', linestyle='--', alpha=0.7)
        
        # 3. Davies-Bouldin Score
        valid_db = [(k, s) for k, s in zip(k_values, db_scores) if s is not None]
        if valid_db:
            k_db, db_vals = zip(*valid_db)
            axes[0,2].plot(k_db, db_vals, 'ro-', linewidth=2, markersize=6)
            axes[0,2].set_title('Davies-Bouldin Score (Lower is Better)')
            axes[0,2].set_xlabel('Number of Clusters (k)')
            axes[0,2].set_ylabel('DB Score')
            axes[0,2].grid(True, alpha=0.3)
            
            # Mark optimal
            best_idx = np.argmin(db_vals)
            axes[0,2].axvline(x=k_db[best_idx], color='red', linestyle='--', alpha=0.7)
        
        # 4. Inertia (Elbow Method)
        valid_inertia = [(k, s) for k, s in zip(k_values, inertias) if s is not None]
        if valid_inertia:
            k_inertia, inertia_vals = zip(*valid_inertia)
            axes[1,0].plot(k_inertia, inertia_vals, 'mo-', linewidth=2, markersize=6)
            axes[1,0].set_title('Inertia - Elbow Method (Lower is Better)')
            axes[1,0].set_xlabel('Number of Clusters (k)')
            axes[1,0].set_ylabel('Inertia')
            axes[1,0].grid(True, alpha=0.3)
            
            # Mark optimal (minimum inertia)
            best_idx = np.argmin(inertia_vals)
            axes[1,0].axvline(x=k_inertia[best_idx], color='red', linestyle='--', alpha=0.7)
            axes[1,0].text(k_inertia[best_idx], inertia_vals[best_idx], f'  k={k_inertia[best_idx]}', 
                           verticalalignment='bottom')
        
        # 5. Balance Score
        valid_balance = [(k, s) for k, s in zip(k_values, balance_scores) if s is not None]
        if valid_balance:
            k_bal, bal_vals = zip(*valid_balance)
            axes[1,1].plot(k_bal, bal_vals, 'co-', linewidth=2, markersize=6)
            axes[1,1].set_title('Cluster Balance Score (Higher is Better)')
            axes[1,1].set_xlabel('Number of Clusters (k)')
            axes[1,1].set_ylabel('Balance Score')
            axes[1,1].grid(True, alpha=0.3)
            
            # Mark optimal
            best_idx = np.argmax(bal_vals)
            axes[1,1].axvline(x=k_bal[best_idx], color='red', linestyle='--', alpha=0.7)
            axes[1,1].text(k_bal[best_idx], bal_vals[best_idx], f'  k={k_bal[best_idx]}', 
                           verticalalignment='bottom')
        
        # 6. Empty subplot (or optionally cluster size distribution)
        axes[1,2].axis('off')  # Placeholder; could add histogram of cluster sizes
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            log_success(f"Model selection plot saved to {save_path}")
        plt.show()
    
    except Exception as e:
        log_fail(f"Plotting failed: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Load or prepare data
    X = prepare_data_pipeline(verbose=True)  # Make sure this returns a NumPy array
    
    # Run hierarchical model selection
    results = hierarchical_model_selection(X, k_range=range(2, 15), linkage="average", metric="euclidean")
    
    # Find optimal k
    optimal = find_optimal_k(results, criterion='silhouette')
    
    # Plot results
    plot_model_selection_results(results, save_path="hierarchical_model_selection.png")
