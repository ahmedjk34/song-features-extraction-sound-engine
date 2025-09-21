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

def _assess_clustering_quality(silhouette, balance):
    """Assess clustering quality based on metrics."""
    if silhouette >= 0.5 and balance >= 0.5:
        return "EXCELLENT"
    elif silhouette >= 0.3 and balance >= 0.3:
        return "GOOD"
    elif silhouette >= 0.1 and balance >= 0.1:
        return "FAIR"
    else:
        return "POOR"

def comprehensive_hierarchical_model_selection(X, k_range=None, max_samples=1000, verbose=True):
    """
    Comprehensive model selection with detailed logging like main.py
    """
    if verbose:
        log_info("Running COMPREHENSIVE Hierarchical clustering with automatic parameter tuning...")
    
    # Sample data if needed
    X_work, sample_indices = sample_data_for_hierarchical(X, max_samples=max_samples)
    
    if verbose:
        log_info(f"Using dataset (shape: {X_work.shape})")
        log_info(f"Feature range: [{X_work.min():.3f}, {X_work.max():.3f}], std: {X_work.std():.3f}")
    
    # Set default k_range
    if k_range is None:
        k_range = range(3, min(11, X_work.shape[0]))
    
    # Test configurations with multiple distance metrics for high-dimensional data
    test_configs = []
    for k in k_range:
        for linkage in ['ward', 'complete', 'average']:
            if linkage == 'ward':
                # Ward only works with euclidean
                test_configs.append({'n_clusters': k, 'linkage': linkage, 'metric': 'euclidean'})
            else:
                # Test multiple metrics for complete/average linkage
                for metric in ['euclidean', 'manhattan', 'cosine']:
                    test_configs.append({'n_clusters': k, 'linkage': linkage, 'metric': metric})
    
    best_score = -1
    best_config = None
    best_labels = None
    best_clustering = None
    all_results = []
    
    log_info(f"🔍 TESTING {len(test_configs)} CLUSTERING CONFIGURATIONS...")
    log_info("=" * 80)
    
    for i, config in enumerate(test_configs, 1):
        try:
            actual_k = min(config['n_clusters'], X_work.shape[0] - 1)
            if actual_k < 2:
                continue
                
            config_name = f"k={actual_k}_{config['linkage']}_{config['metric']}"
            log_info(f"[{i:2d}/{len(test_configs)}] Testing: {config_name}")
            
            # Create clustering model
            clustering = AgglomerativeClustering(
                n_clusters=actual_k,
                linkage=config['linkage'],
                metric=config['metric']
            )
            labels = clustering.fit_predict(X_work)
            
            # Calculate metrics with appropriate distance metric
            if len(np.unique(labels)) > 1:
                # Use same metric for silhouette score as clustering
                metric_for_silhouette = config['metric']
                if config['metric'] == 'cosine':
                    # Cosine distance for high-dimensional features
                    silhouette = silhouette_score(X_work, labels, metric='cosine')
                elif config['metric'] == 'manhattan':
                    # Manhattan (L1) distance
                    silhouette = silhouette_score(X_work, labels, metric='manhattan')
                else:
                    # Euclidean distance
                    silhouette = silhouette_score(X_work, labels, metric='euclidean')
                
                # These metrics always use euclidean internally
                calinski_harabasz = calinski_harabasz_score(X_work, labels)
                davies_bouldin = davies_bouldin_score(X_work, labels)
                
                # Calculate cluster balance
                cluster_sizes = [np.sum(labels == j) for j in np.unique(labels)]
                balance = min(cluster_sizes) / max(cluster_sizes) if cluster_sizes else 0.0
                
                # Calculate composite score (prioritize balance and silhouette)
                composite_score = (silhouette * 0.4) + (balance * 0.6)
                
                # Detailed logging for each configuration
                log_info(f"    📊 METRICS:")
                log_info(f"      Silhouette Score: {silhouette:7.4f} {'✅' if silhouette > 0.2 else '⚠️' if silhouette > 0 else '❌'}")
                log_info(f"      Balance Ratio:    {balance:7.4f} {'✅' if balance > 0.5 else '⚠️' if balance > 0.1 else '❌'}")
                log_info(f"      Composite Score:  {composite_score:7.4f}")
                log_info(f"      Calinski-Harabasz: {calinski_harabasz:6.2f}")
                log_info(f"      Davies-Bouldin:   {davies_bouldin:7.4f}")
                log_info(f"      Cluster Sizes:    {cluster_sizes}")
                
                all_results.append({
                    'config': config,
                    'config_name': config_name,
                    'clustering': clustering,
                    'labels': labels,
                    'k': actual_k,
                    'linkage': config['linkage'],
                    'metric': config['metric'],
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski_harabasz,
                    'davies_bouldin_score': davies_bouldin,
                    'balance_score': balance,
                    'composite_score': composite_score,
                    'cluster_sizes': cluster_sizes,
                    'inertia': None  # We'll calculate this if needed
                })
                
                # Update best configuration
                if composite_score > best_score:
                    best_score = composite_score
                    best_config = config
                    best_labels = labels
                    best_clustering = clustering
                    log_info(f"    🏆 NEW BEST CONFIGURATION! (Score: {composite_score:.4f})")
                else:
                    log_info(f"    📈 Score: {composite_score:.4f} (Current best: {best_score:.4f})")
                    
            log_info("-" * 60)
                    
        except Exception as e:
            log_fail(f"    ❌ Config {config} failed: {e}")
            continue
    
    if best_labels is None:
        log_fail("❌ All hierarchical clustering configurations failed")
        return None
    
    # COMPREHENSIVE RESULTS SUMMARY
    log_info("=" * 80)
    log_success("🎯 CLUSTERING CONFIGURATION ANALYSIS COMPLETE")
    log_info("=" * 80)
    
    # Sort results by composite score
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    log_info("📊 TOP 5 CONFIGURATIONS:")
    for i, result in enumerate(all_results[:5], 1):
        status = "WINNER" if i == 1 else f"#{i}"
        log_info(f"  {status} {result['config_name']}")
        log_info(f"      Composite: {result['composite_score']:.4f} | "
                f"Silhouette: {result['silhouette_score']:.4f} | "
                f"Balance: {result['balance_score']:.4f}")
        log_info(f"      Sizes: {result['cluster_sizes']}")
    
    log_info("=" * 80)
    log_success(f"WINNING CONFIGURATION:")
    best_result = all_results[0]
    log_info(f"  Configuration: k={best_config['n_clusters']}, linkage={best_config['linkage']}, metric={best_config['metric']}")
    log_info(f"  Composite Score: {best_result['composite_score']:.4f}")
    log_info(f"  Silhouette Score: {best_result['silhouette_score']:.4f}")
    log_info(f"  Balance Ratio: {best_result['balance_score']:.4f}")
    log_info(f"  Cluster Sizes: {best_result['cluster_sizes']}")
    log_info(f"  Quality Assessment: {_assess_clustering_quality(best_result['silhouette_score'], best_result['balance_score'])}")
    log_info("=" * 80)
    
    return {
        'results': all_results,
        'best_config': best_config,
        'best_result': best_result,
        'sampled': X_work.shape[0] < X.shape[0],
        'sample_size': X_work.shape[0],
        'original_size': X.shape[0],
        'sample_indices': sample_indices
    }

def get_final_clustering_on_full_data(X, best_config, verbose=True):
    """
    Apply the best configuration to the full dataset
    """
    if verbose:
        log_info(f"🔄 Applying best configuration to full dataset: k={best_config['n_clusters']}, linkage={best_config['linkage']}")
    
    try:
        # Create clustering model with best config
        clustering = AgglomerativeClustering(
            n_clusters=best_config['n_clusters'],
            linkage=best_config['linkage'],
            metric=best_config.get('metric', 'euclidean')
        )
        
        labels = clustering.fit_predict(X)
        
        # Calculate final metrics on full data
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels, metric='euclidean')
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            cluster_sizes = [np.sum(labels == j) for j in np.unique(labels)]
            balance = min(cluster_sizes) / max(cluster_sizes) if cluster_sizes else 0.0
            
            if verbose:
                log_success("✅ Final clustering on full dataset completed!")
                log_info("📈 FINAL FULL DATASET METRICS:")
                log_info(f"  Total songs processed: {len(labels)}")
                log_info(f"  Final cluster count: {len(np.unique(labels))}")
                log_info(f"  Silhouette Score: {silhouette:.4f}")
                log_info(f"  Balance Score: {balance:.4f}")
                log_info(f"  Cluster Sizes: {cluster_sizes}")
                log_info(f"  Quality Assessment: {_assess_clustering_quality(silhouette, balance)}")
            
            return {
                'labels': labels,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'balance_score': balance,
                'cluster_sizes': cluster_sizes,
                'clustering': clustering
            }
        
    except Exception as e:
        log_fail(f"Final clustering failed: {e}")
        return None

def plot_essential_results(model_selection_results, final_results, X, save_path=None, figsize=(15, 5)):
    """
    Create essential visualization plots - ONLY the 3 most important ones
    """
    try:
        log_info("🎨 Creating essential visualization plots...")
        
        # Create subplots - 1 row, 3 columns
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("Essential Hierarchical Clustering Results", fontsize=16, fontweight='bold')
        
        data = model_selection_results['results']
        best_result = model_selection_results['best_result']
        
        # Group by linkage for better visualization
        linkages = ['ward', 'complete', 'average']
        colors = {'ward': 'blue', 'complete': 'green', 'average': 'red'}
        
        # PLOT 1: Composite Score by K and Linkage (Most Important - shows optimization)
        ax1 = axes[0]
        
        # Group data by linkage and metric for better visualization
        linkages = ['ward', 'complete', 'average']
        metrics = ['euclidean', 'manhattan', 'cosine']
        colors = {'ward': 'blue', 'complete': 'green', 'average': 'red'}
        linestyles = {'euclidean': '-', 'manhattan': '--', 'cosine': ':'}
        
        for linkage in linkages:
            for metric in metrics:
                if linkage == 'ward' and metric != 'euclidean':
                    continue  # Ward only supports euclidean
                    
                linkage_metric_data = [r for r in data if r['linkage'] == linkage and r['metric'] == metric]
                if linkage_metric_data:
                    k_vals = [r['k'] for r in linkage_metric_data]
                    comp_vals = [r['composite_score'] for r in linkage_metric_data]
                    label = f"{linkage}-{metric}" if linkage != 'ward' else linkage
                    ax1.plot(k_vals, comp_vals, 'o-', color=colors[linkage], 
                            linestyle=linestyles[metric], label=label, 
                            linewidth=2, markersize=4, alpha=0.7)
        
        ax1.set_title('Composite Score by K, Linkage & Metric', fontweight='bold')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Composite Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Mark the winner
        ax1.axvline(x=best_result['k'], color='gold', linestyle='--', linewidth=2, alpha=0.8)
        ax1.scatter([best_result['k']], [best_result['composite_score']], 
                   color='gold', s=200, marker='*', edgecolor='black', linewidth=2, zorder=5)
        ax1.text(best_result['k'], best_result['composite_score'], 
                f"  WINNER\n  k={best_result['k']}\n  {best_result['linkage']}\n  {best_result['metric']}", 
                verticalalignment='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        # PLOT 2: Final Cluster Sizes (Shows actual distribution)
        ax2 = axes[1]
        if final_results:
            cluster_sizes = final_results['cluster_sizes']
            cluster_ids = list(range(len(cluster_sizes)))
            bars = ax2.bar(cluster_ids, cluster_sizes, color='lightblue', 
                          alpha=0.7, edgecolor='darkblue', linewidth=1)
            
            ax2.set_title(f'Final Cluster Distribution (k={best_result["k"]})', fontweight='bold')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Number of Songs')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            max_height = max(cluster_sizes)
            for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
                height = bar.get_height()
                pct = (size / sum(cluster_sizes)) * 100
                ax2.text(bar.get_x() + bar.get_width()/2, height + max_height*0.01,
                        f'{size}\n({pct:.1f}%)', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
            
            # Add balance info
            balance = final_results['balance_score']
            quality = _assess_clustering_quality(final_results['silhouette_score'], balance)
            ax2.text(0.02, 0.98, f'Balance: {balance:.3f}\nQuality: {quality}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # PLOT 3: Top 5 Configurations Comparison
        ax3 = axes[2]
        top_5 = data[:5]
        config_names = [f"k={r['k']}\n{r['linkage']}\n{r['metric']}" for r in top_5]
        composite_vals = [r['composite_score'] for r in top_5]
        
        # Create color gradient - gold for winner, others in decreasing intensity
        bar_colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgray'][:len(top_5)]
        
        bars = ax3.barh(range(len(top_5)), composite_vals, color=bar_colors,
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        ax3.set_yticks(range(len(top_5)))
        ax3.set_yticklabels(config_names, fontweight='bold')
        ax3.set_xlabel('Composite Score')
        ax3.set_title('Top 5 Configurations', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels and rank badges
        max_val = max(composite_vals)
        badges = ['#1', '#2', '#3', '#4', '#5']
        for i, (bar, val) in enumerate(zip(bars, composite_vals)):
            # Score label
            ax3.text(val + max_val*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
            # Rank badge
            ax3.text(max_val*0.02, bar.get_y() + bar.get_height()/2,
                    badges[i], ha='left', va='center', fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Add sampling info if applicable
        if model_selection_results.get('sampled', False):
            fig.text(0.5, 0.02, 
                    f"Model selection on {model_selection_results['sample_size']} sampled points, "
                    f"final results on {model_selection_results['original_size']} total points",
                    ha='center', fontsize=10, style='italic')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log_success(f"📊 Essential plots saved to {save_path}")
        
        plt.show()
        
        log_success("🎨 Essential visualization completed!")
        return fig
    
    except Exception as e:
        log_fail(f"Essential plotting failed: {e}")
        return None

if __name__ == "__main__":
    # Load or prepare data
    log_info("🚀 Starting essential hierarchical clustering analysis...")
    
    pipeline_result = asyncio.run(prepare_data_pipeline())
    
    # Check if pipeline succeeded
    if pipeline_result is None or pipeline_result[0] is None:
        log_fail("Data pipeline failed")
        sys.exit(1)
    
    # Unpack the tuple
    X, pca, verified_songs = pipeline_result
    
    log_info(f"Loaded data with shape: {X.shape}")
    log_info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Run comprehensive hierarchical model selection (like main.py)
    model_selection_results = comprehensive_hierarchical_model_selection(
        X,
        k_range=range(3, 11),  # Test k=3 to k=10
        max_samples=10000,      # Limit to 1000 samples for efficiency during selection
        verbose=True
    )
    
    if model_selection_results is None:
        log_fail("Model selection failed")
        sys.exit(1)
    
    # Apply best configuration to full dataset
    final_results = get_final_clustering_on_full_data(X, model_selection_results['best_config'])
    
    if final_results is None:
        log_fail("Final clustering failed")
        sys.exit(1)
    
    # Create ESSENTIAL visualization (only 3 plots)
    fig = plot_essential_results(
        model_selection_results, 
        final_results, 
        X, 
        save_path="hierarchical_model_selection.png"
    )
    
    log_success("🎉 ESSENTIAL HIERARCHICAL CLUSTERING ANALYSIS COMPLETED!")
    log_info("=" * 80)
    log_info("📋 SUMMARY:")
    log_info(f"  • Tested {len(model_selection_results['results'])} configurations")
    log_info(f"  • Best: k={model_selection_results['best_config']['n_clusters']}, linkage={model_selection_results['best_config']['linkage']}")
    log_info(f"  • Final silhouette: {final_results['silhouette_score']:.4f}")
    log_info(f"  • Final balance: {final_results['balance_score']:.4f}")
    log_info(f"  • Quality: {_assess_clustering_quality(final_results['silhouette_score'], final_results['balance_score'])}")
    log_info("=" * 80)