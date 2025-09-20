import sys
import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import resample

# Add parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import your modules
from clustering_util import prepare_data_pipeline
from hierarchical import log_info, log_success, log_warn, log_fail, log_debug
from hierarchical_extract import compute_cluster_statistics

def sample_data_for_hierarchical(X, max_samples=1000, random_state=42):
    """
    Sample data to make hierarchical clustering feasible.
    """
    if X.shape[0] <= max_samples:
        log_info(f"Data size {X.shape[0]} <= {max_samples}, using all data")
        return X, np.arange(X.shape[0])
    
    log_warn(f"Data too large ({X.shape[0]} points). Sampling {max_samples} points for hierarchical clustering")
    
    # Sample data
    X_sampled, indices = resample(X, np.arange(X.shape[0]), 
                                 n_samples=max_samples, 
                                 random_state=random_state, 
                                 replace=False)
    
    log_info(f"Sampled data shape: {X_sampled.shape}")
    return X_sampled, indices

def evaluate_clustering_for_k_sklearn(X, k, linkage='average', verbose=False):
    """
    Evaluate clustering quality for a specific number of clusters using sklearn.
    This is much more efficient than custom implementation.
    """
    try:
        if verbose:
            log_debug(f"Evaluating k={k} clusters")
        
        # Use sklearn's AgglomerativeClustering (much more efficient)
        clusterer = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(X)
        
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

def hierarchical_model_selection_efficient(X, k_range=None, linkage="average", 
                                         max_samples=1000, verbose=True):
    """
    Perform efficient model selection for hierarchical clustering.
    Uses sampling for large datasets and sklearn's optimized implementation.
    """
    if verbose:
        log_info("Starting efficient hierarchical clustering model selection")
    
    # Set default k_range
    if k_range is None:
        k_range = range(2, min(21, X.shape[0]))
    
    # Sample data if too large
    X_work, sample_indices = sample_data_for_hierarchical(X, max_samples=max_samples)
    
    if verbose:
        log_info(f"Testing k values: {list(k_range)}")
        log_info(f"Working data shape: {X_work.shape}")
        log_info(f"Using {linkage} linkage with euclidean metric")
    
    # Evaluate each k value
    results = []
    for k in k_range:
        if k >= X_work.shape[0]:
            log_warn(f"Skipping k={k} (>= number of data points)")
            continue
        
        metrics = evaluate_clustering_for_k_sklearn(X_work, k, linkage=linkage, verbose=False)
        if metrics is not None:
            results.append(metrics)
            if verbose:
                sil = metrics.get('silhouette_score')
                ch = metrics.get('calinski_harabasz_score')
                db = metrics.get('davies_bouldin_score')
                
                sil_str = f"{sil:.3f}" if sil is not None else "N/A"
                ch_str = f"{ch:.1f}" if ch is not None else "N/A"
                db_str = f"{db:.3f}" if db is not None else "N/A"
                
                print(f"k={k:2d}: Silhouette={sil_str:>6}, "
                      f"CH={ch_str:>8}, "
                      f"DB={db_str:>6}")
    
    if not results:
        log_fail("No successful evaluations")
        return None
    
    log_success(f"Evaluated {len(results)} different k values")
    
    return {
        'results': results,
        'linkage_type': linkage,
        'metric': 'euclidean',
        'sampled': X_work.shape[0] < X.shape[0],
        'sample_size': X_work.shape[0],
        'original_size': X.shape[0],
        'sample_indices': sample_indices
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

def get_final_clustering(X, optimal_k, linkage='average', verbose=True):
    """
    Get final clustering on full dataset using optimal k.
    """
    if verbose:
        log_info(f"Computing final clustering with k={optimal_k}")
    
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=optimal_k,
            linkage=linkage,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(X)
        
        if verbose:
            log_success(f"Final clustering completed. Found {len(np.unique(labels))} clusters")
            
            # Compute final statistics
            cluster_stats = compute_cluster_statistics(labels, verbose=True)
            
        return labels, cluster_stats
        
    except Exception as e:
        log_fail(f"Final clustering failed: {e}")
        return None, None

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
        
        # Add title indicating if sampling was used
        if results.get('sampled', False):
            fig.suptitle(f"Hierarchical Clustering Model Selection\n"
                        f"(Evaluated on {results['sample_size']} sampled points "
                        f"from {results['original_size']} total)", fontsize=14)
        else:
            fig.suptitle("Hierarchical Clustering Model Selection", fontsize=14)
        
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
        
        # 6. Summary table
        axes[1,2].axis('off')
        
        # Create summary text
        summary_text = "Optimal k by different criteria:\n\n"
        criteria = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        for criterion in criteria:
            optimal = find_optimal_k(results, criterion=criterion, verbose=False)
            if optimal:
                summary_text += f"{criterion.replace('_', ' ').title()}: k={optimal['k']}\n"
        
        axes[1,2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log_success(f"Model selection plot saved to {save_path}")
        plt.show()
        
        return fig
    
    except Exception as e:
        log_fail(f"Plotting failed: {e}")
        return None

if __name__ == "__main__":
    # Load or prepare data
    pipeline_result = asyncio.run(prepare_data_pipeline())
    
    # Check if pipeline succeeded
    if pipeline_result is None or pipeline_result[0] is None:
        log_fail("Data pipeline failed")
        sys.exit(1)
    
    # Unpack the tuple
    X, pca, verified_songs = pipeline_result
    
    log_info(f"Loaded data with shape: {X.shape}")
    log_info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Run efficient hierarchical model selection
    results = hierarchical_model_selection_efficient(
        X,
        k_range=range(2, 15),
        linkage="average",
        max_samples=1000,  # Limit to 1000 samples for efficiency
        verbose=True
    )
    
    if results is None:
        log_fail("Model selection failed")
        sys.exit(1)
    
    # Find optimal k using different criteria
    criteria = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    optimal_results = {}
    
    for criterion in criteria:
        optimal = find_optimal_k(results, criterion=criterion)
        if optimal:
            optimal_results[criterion] = optimal
            log_success(f"Best k by {criterion}: {optimal['k']}")
    
    # Use silhouette score as primary criterion
    if 'silhouette' in optimal_results:
        best_k = optimal_results['silhouette']['k']
        log_info(f"Using k={best_k} based on silhouette score for final clustering")
        
        # Get final clustering on full dataset
        final_labels, final_stats = get_final_clustering(X, best_k, linkage="average")
        
        if final_labels is not None:
            log_success("Final clustering completed successfully!")
    
    # Plot results
    plot_model_selection_results(results, save_path="hierarchical_model_selection.png")