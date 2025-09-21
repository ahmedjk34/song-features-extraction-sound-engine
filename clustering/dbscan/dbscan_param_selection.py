import sys
import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Add parent directory to sys.path ---
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- Imports from parent directory and local files ---
from clustering_util import prepare_data_pipeline
from dbscan import k_distance_plot, dbscan, get_cluster_info

class DBSCANParamSelector:
    """Helper class for DBSCAN parameter selection and analysis."""

    def __init__(self, X):
        self.X = X
        self.param_results = {}

    def analyze_k_distances(self, min_samples_values=[3, 4, 5, 6, 8, 10], max_points=1000):
        """Analyze k-distances for different min_samples values."""
        print("Analyzing k-distances for parameter selection...")

        # Create subplots for different min_samples values
        n_plots = len(min_samples_values)
        cols = 3
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        all_k_distances = {}

        for idx, min_samples in enumerate(min_samples_values):
            row = idx // cols
            col = idx % cols

            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]

            # Compute k-distances
            n_samples = self.X.shape[0]
            k_distances = []

            # Sample points if dataset is too large
            if n_samples > max_points:
                indices = np.random.choice(n_samples, max_points, replace=False)
                X_sample = self.X[indices]
            else:
                X_sample = self.X

            for i in range(X_sample.shape[0]):
                distances_from_i = []
                for j in range(X_sample.shape[0]):
                    if i != j:
                        dist = np.linalg.norm(X_sample[i] - X_sample[j])
                        distances_from_i.append(dist)

                distances_from_i.sort()
                if len(distances_from_i) >= min_samples - 1:
                    k_dist = distances_from_i[min_samples - 2]
                    k_distances.append(k_dist)

            k_distances = sorted(k_distances, reverse=True)
            all_k_distances[min_samples] = k_distances

            # Plot k-distances
            ax.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
            ax.set_xlabel('Points sorted by distance')
            ax.set_ylabel(f'{min_samples}-distance')
            ax.set_title(f'K-Distance Plot (k={min_samples})')
            ax.grid(True, alpha=0.3)

            # Add percentile lines
            for p in [90, 95, 99]:
                if len(k_distances) > 0:
                    val = np.percentile(k_distances, p)
                    ax.axhline(y=val, color='red', linestyle='--', alpha=0.7)
                    ax.text(len(k_distances)*0.7, val, f'{p}th: {val:.3f}',
                            fontsize=8, va='bottom')

        # Hide empty subplots
        for idx in range(len(min_samples_values), rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)

        plt.tight_layout()
        plt.show()

        return all_k_distances

    def suggest_eps_values(self, k_distances_dict):
        """Suggest eps values based on k-distance analysis."""
        suggestions = {}

        print("\n" + "="*60)
        print("EPS PARAMETER SUGGESTIONS")
        print("="*60)

        for min_samples, k_distances in k_distances_dict.items():
            # Calculate potential eps values
            knee_candidates = []

            # Method 1: Look for the steepest drop (elbow/knee)
            if len(k_distances) > 10:
                # Calculate second derivative to find inflection point
                k_dist_array = np.array(k_distances)
                diff1 = np.diff(k_dist_array)
                diff2 = np.diff(diff1)

                # Find point with maximum second derivative (steepest increase)
                if len(diff2) > 0:
                    knee_idx = np.argmax(np.abs(diff2)) + 2
                    if knee_idx < len(k_distances):
                        knee_candidates.append(k_distances[knee_idx])

            # Method 2: Use percentiles
            if len(k_distances) > 0:
                percentile_95 = np.percentile(k_distances, 95)
                percentile_90 = np.percentile(k_distances, 90)
                percentile_85 = np.percentile(k_distances, 85)
            else:
                percentile_95 = percentile_90 = percentile_85 = 0

            knee_candidates.extend([percentile_85, percentile_90, percentile_95])

            # Method 3: Use mean and standard deviation
            if len(k_distances) > 0:
                mean_k_dist = np.mean(k_distances)
                std_k_dist = np.std(k_distances)
                knee_candidates.append(mean_k_dist + 0.5 * std_k_dist)
            else:
                mean_k_dist = std_k_dist = 0

            # Select final suggestions
            knee_candidates = [k for k in knee_candidates if k > 0]
            knee_candidates.sort()

            # Remove duplicates and select diverse values
            final_suggestions = []
            for candidate in knee_candidates:
                if not final_suggestions or abs(candidate - final_suggestions[-1]) > 0.1 * candidate:
                    final_suggestions.append(candidate)

            suggestions[min_samples] = final_suggestions[:3]  # Top 3 suggestions

            print(f"\nmin_samples = {min_samples}:")
            print(f"  Suggested eps values: {[f'{eps:.3f}' for eps in final_suggestions[:3]]}")
            print(f"  85th percentile: {percentile_85:.3f}")
            print(f"  90th percentile: {percentile_90:.3f}")
            print(f"  95th percentile: {percentile_95:.3f}")

        return suggestions

    def parameter_grid_search(self, eps_range, min_samples_range, max_combinations=20):
        """Perform grid search over parameter combinations."""
        print(f"\nPerforming parameter grid search...")

        # Convert eps values to float if they're numpy types
        eps_range = [float(eps) for eps in eps_range]

        print(f"eps range: {eps_range}")
        print(f"min_samples range: {min_samples_range}")

        results = []
        combinations_tested = 0

        for min_samples in min_samples_range:
            for eps in eps_range:
                if combinations_tested >= max_combinations:
                    break

                print(f"Testing eps={eps:.3f}, min_samples={min_samples}...")

                # Run DBSCAN
                labels, core_samples, n_clusters = dbscan(self.X, eps, min_samples,
                                                          precompute_distances=False)

                # Get cluster info
                info = get_cluster_info(labels, core_samples)

                # Calculate quality metrics
                n_noise = info['n_noise']
                noise_ratio = info['noise_ratio']

                # Silhouette score (if possible)
                try:
                    if n_clusters > 1 and n_noise < len(labels):
                        from sklearn.metrics import silhouette_score
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1:
                            sil_score = silhouette_score(self.X[non_noise_mask], labels[non_noise_mask])
                        else:
                            sil_score = float('nan')
                    else:
                        sil_score = float('nan')
                except Exception:
                    sil_score = float('nan')

                # Fix: precompute formatted string for silhouette score
                sil_score_str = f"{sil_score:.3f}" if not np.isnan(sil_score) else "N/A"

                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': noise_ratio,
                    'silhouette_score': sil_score,
                    'cluster_sizes': info['cluster_sizes'],
                    'labels': labels,
                    'core_samples': core_samples
                }

                results.append(result)
                combinations_tested += 1

                print(f"  → {n_clusters} clusters, {n_noise} noise points ({noise_ratio:.2%}), silhouette: {sil_score_str}")

        self.param_results = results
        return results

    def visualize_parameter_space(self, results):
        """Visualize the parameter search results."""
        if not results:
            print("No results to visualize!")
            return

        # Extract data for plotting
        eps_values = [r['eps'] for r in results]
        min_samples_values = [r['min_samples'] for r in results]
        n_clusters = [r['n_clusters'] for r in results]
        noise_ratios = [r['noise_ratio'] for r in results]
        silhouette_scores = [r['silhouette_score'] if not np.isnan(r['silhouette_score']) else 0 for r in results]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Number of clusters vs parameters
        scatter1 = axes[0, 0].scatter(eps_values, min_samples_values, c=n_clusters,
                                     s=100, cmap='viridis', alpha=0.7)
        axes[0, 0].set_xlabel('eps')
        axes[0, 0].set_ylabel('min_samples')
        axes[0, 0].set_title('Number of Clusters Found')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Number of Clusters')

        # Add text annotations
        for i, result in enumerate(results):
            if result['n_clusters'] > 0:
                axes[0, 0].annotate(f"{result['n_clusters']}",
                                   (result['eps'], result['min_samples']),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)

        # 2. Noise ratio vs parameters
        scatter2 = axes[0, 1].scatter(eps_values, min_samples_values, c=noise_ratios,
                                     s=100, cmap='Reds', alpha=0.7)
        axes[0, 1].set_xlabel('eps')
        axes[0, 1].set_ylabel('min_samples')
        axes[0, 1].set_title('Noise Ratio')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Noise Ratio')

        # 3. Silhouette score vs parameters
        valid_sil_scores = [s for s in silhouette_scores if s != 0]
        valid_eps = [eps_values[i] for i, s in enumerate(silhouette_scores) if s != 0]
        valid_min_samples = [min_samples_values[i] for i, s in enumerate(silhouette_scores) if s != 0]

        if valid_sil_scores:
            scatter3 = axes[1, 0].scatter(valid_eps, valid_min_samples, c=valid_sil_scores,
                                         s=100, cmap='RdYlGn', alpha=0.7)
            axes[1, 0].set_xlabel('eps')
            axes[1, 0].set_ylabel('min_samples')
            axes[1, 0].set_title('Silhouette Score')
            plt.colorbar(scatter3, ax=axes[1, 0], label='Silhouette Score')
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid silhouette scores',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Silhouette Score (No Valid Data)')

        # 4. Parameter recommendations
        axes[1, 1].axis('off')

        # Find best parameters based on different criteria
        best_by_silhouette = max([r for r in results if not np.isnan(r['silhouette_score'])],
                                key=lambda x: x['silhouette_score'], default=None)

        # Best by balanced criteria (good silhouette, reasonable noise ratio)
        balanced_candidates = [r for r in results
                             if not np.isnan(r['silhouette_score'])
                             and r['noise_ratio'] < 0.3
                             and r['n_clusters'] > 1]

        best_balanced = max(balanced_candidates, key=lambda x: x['silhouette_score'], default=None)

        recommendations_text = "PARAMETER RECOMMENDATIONS\n" + "="*30 + "\n\n"

        if best_by_silhouette:
            recommendations_text += f"Best Silhouette Score:\n"
            recommendations_text += f"  eps = {best_by_silhouette['eps']:.3f}\n"
            recommendations_text += f"  min_samples = {best_by_silhouette['min_samples']}\n"
            recommendations_text += f"  → {best_by_silhouette['n_clusters']} clusters\n"
            recommendations_text += f"  → {best_by_silhouette['noise_ratio']:.2%} noise\n"
            recommendations_text += f"  → silhouette: {best_by_silhouette['silhouette_score']:.3f}\n\n"

        if best_balanced and best_balanced != best_by_silhouette:
            recommendations_text += f"Best Balanced Result:\n"
            recommendations_text += f"  eps = {best_balanced['eps']:.3f}\n"
            recommendations_text += f"  min_samples = {best_balanced['min_samples']}\n"
            recommendations_text += f"  → {best_balanced['n_clusters']} clusters\n"
            recommendations_text += f"  → {best_balanced['noise_ratio']:.2%} noise\n"
            recommendations_text += f"  → silhouette: {best_balanced['silhouette_score']:.3f}\n\n"

        # General guidelines
        recommendations_text += "GENERAL GUIDELINES:\n"
        recommendations_text += "• Lower eps → more clusters, more noise\n"
        recommendations_text += "• Higher eps → fewer clusters, less noise\n"
        recommendations_text += "• Higher min_samples → fewer clusters\n"
        recommendations_text += "• Good silhouette score > 0.5\n"
        recommendations_text += "• Reasonable noise ratio < 20-30%\n"

        axes[1, 1].text(0.05, 0.95, recommendations_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.show()

        return best_by_silhouette, best_balanced

    def get_final_recommendations(self):
        """Get final parameter recommendations based on all analysis."""
        if not self.param_results:
            print("No parameter search results available. Run parameter_grid_search first.")
            return None

        print("\n" + "="*60)
        print("FINAL PARAMETER RECOMMENDATIONS")
        print("="*60)

        # Filter out poor results
        good_results = [r for r in self.param_results
                       if r['n_clusters'] >= 2
                       and r['noise_ratio'] <= 0.5
                       and not np.isnan(r['silhouette_score'])]

        if not good_results:
            print("No good parameter combinations found. Consider:")
            print("- Expanding the search range")
            print("- Using different distance metrics")
            print("- Preprocessing the data differently")
            return None

        # Rank by silhouette score
        good_results.sort(key=lambda x: x['silhouette_score'], reverse=True)

        print("\nTop 3 Parameter Combinations:")
        print("-" * 40)

        for i, result in enumerate(good_results[:3]):
            print(f"\n{i+1}. eps={result['eps']:.3f}, min_samples={result['min_samples']}")
            print(f"   Clusters: {result['n_clusters']}")
            print(f"   Noise: {result['n_noise']} points ({result['noise_ratio']:.1%})")
            print(f"   Silhouette Score: {result['silhouette_score']:.3f}")

            # Show cluster size distribution
            cluster_sizes = list(result['cluster_sizes'].values())
            if cluster_sizes:
                print(f"   Cluster sizes: {cluster_sizes}")

        return good_results[:3]

async def main():
    """Main parameter selection pipeline."""
    # --- Use reusable data pipeline ---
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return

    print(f"Data loaded: {reduced_vectors.shape[0]} songs, {reduced_vectors.shape[1]} features")

    # Initialize parameter selector
    selector = DBSCANParamSelector(reduced_vectors)

    # Step 1: Analyze k-distances
    print("\n" + "="*60)
    print("STEP 1: K-DISTANCE ANALYSIS")
    print("="*60)

    k_distances_dict = selector.analyze_k_distances(min_samples_values=[3, 4, 5, 6, 8, 10])

    # Step 2: Get eps suggestions
    eps_suggestions = selector.suggest_eps_values(k_distances_dict)

    # Step 3: Parameter grid search
    print("\n" + "="*60)
    print("STEP 2: PARAMETER GRID SEARCH")
    print("="*60)

    #!VERY EXAPNDED, TAKES LONG TIME
    # # Use suggested eps values for grid search
    # all_eps_values = []
    # for min_samples, eps_list in eps_suggestions.items():
    #     all_eps_values.extend(eps_list)

    # # Add some additional eps values for broader search
    # if all_eps_values:
    #     min_eps, max_eps = min(all_eps_values), max(all_eps_values)
    #     additional_eps = np.linspace(min_eps * 0.5, max_eps * 1.5, 5)
    #     all_eps_values.extend(additional_eps.tolist())

    # # Remove duplicates and sort, ensure all are floats
    # eps_range = sorted(list(set([float(round(eps, 3)) for eps in all_eps_values])))
    # min_samples_range = [3, 4, 5, 6, 8, 10]

    # results = selector.parameter_grid_search(eps_range, min_samples_range, max_combinations=30)

    import numpy as np

# Assume eps_suggestions and selector are already defined

    # 1. Gather all suggested eps values
    all_eps_values = []
    for min_samples, eps_list in eps_suggestions.items():
        all_eps_values.extend(eps_list)

    # 2. Add some additional eps values for broader search
    if all_eps_values:
        min_eps, max_eps = min(all_eps_values), max(all_eps_values)
        additional_eps = np.linspace(min_eps * 0.5, max_eps * 1.5, 5)
        all_eps_values.extend(additional_eps.tolist())

    # 3. Remove duplicates, sort, and round
    eps_range = sorted(set(round(float(eps), 3) for eps in all_eps_values))

    # 4. Choose a smart subset of eps and min_samples to limit total searches to 15
    #    - For best coverage, take sqrt(15) ≈ 4 values of each (4*4 = 16, 3*5=15, etc.)

    # We'll use 3 or 4 values for min_samples, and 4 or 5 for eps depending on length
    min_samples_all = [3, 4, 5, 6, 8, 10]
    n_min_samples = 3
    n_eps = 5

    if len(eps_range) < n_eps:
        n_eps = len(eps_range)
    if len(min_samples_all) < n_min_samples:
        n_min_samples = len(min_samples_all)

    # Select evenly spaced values from each
    min_samples_range = np.linspace(min_samples_all[0], min_samples_all[-1], n_min_samples, dtype=int)
    eps_range_smart = np.linspace(eps_range[0], eps_range[-1], n_eps)
    eps_range_smart = [round(float(e), 3) for e in eps_range_smart]

    # 5. Grid search
    results = selector.parameter_grid_search(eps_range_smart, min_samples_range.tolist(), max_combinations=15)

    # Step 4: Visualize results
    print("\n" + "="*60)
    print("STEP 3: VISUALIZING RESULTS")
    print("="*60)

    best_silhouette, best_balanced = selector.visualize_parameter_space(results)

    # Step 5: Final recommendations
    final_recommendations = selector.get_final_recommendations()

    if final_recommendations:
        print(f"\nRecommended parameters for your dataset:")
        best = final_recommendations[0]
        print(f"eps = {best['eps']:.3f}")
        print(f"min_samples = {best['min_samples']}")
        print(f"Expected: ~{best['n_clusters']} clusters with ~{best['noise_ratio']:.1%} noise")

if __name__ == "__main__":
    asyncio.run(main())