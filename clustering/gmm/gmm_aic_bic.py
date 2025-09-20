import sys
import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt

# --- Add parent directory to sys.path ---
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- CORRECTED Imports ---
from clustering_util import prepare_data_pipeline  # Use your fixed version if you renamed it
from gmm import gmm_em

def compute_bic_aic(X, log_likelihood, n_clusters):
    """Compute BIC and AIC for model selection."""
    N, D = X.shape
    # Parameters: K means (K*D) + K covariances (K*D*(D+1)/2) + K-1 weights
    n_params = n_clusters * D + n_clusters * D * (D + 1) // 2 + (n_clusters - 1)
    
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(N) * n_params - 2 * log_likelihood
    
    return bic, aic

async def main():
    # --- Use reusable data pipeline ---
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return
    
    print(f"Data shape: {reduced_vectors.shape}")
    print(f"Data range: [{reduced_vectors.min():.3f}, {reduced_vectors.max():.3f}]")
    
    K_range = range(2, 11)  # Reduced range for testing
    bics, aics, logLs = [], [], []
    
    for K in K_range:
        print(f"\n=== Testing K={K} ===")
        
        try:
            # CORRECTED: Use the fixed GMM function with proper parameters
            means, covariances, weights, responsibilities, log_likelihood, labels = gmm_em(
                reduced_vectors, 
                n_clusters=K, 
                max_iters=100, 
                tol=1e-6, 
                seed=42, 
                verbose=False  # Set to True if you want to see convergence
            )
            
            # Calculate BIC and AIC
            bic, aic = compute_bic_aic(reduced_vectors, log_likelihood, K)
            
            bics.append(bic)
            aics.append(aic)
            logLs.append(log_likelihood)
            
            # Show cluster sizes
            cluster_sizes = np.bincount(labels)
            print(f"K={K}: BIC={bic:.2f}, AIC={aic:.2f}, logL={log_likelihood:.2f}")
            print(f"Cluster sizes: {cluster_sizes}")
            
        except Exception as e:
            print(f"Error for K={K}: {str(e)}")
            # Append NaN values to maintain list alignment
            bics.append(np.nan)
            aics.append(np.nan)
            logLs.append(np.nan)
            continue
    
    # --- Plot BIC and AIC curves (only for successful runs) ---
    valid_indices = ~np.isnan(bics)
    valid_K = np.array(list(K_range))[valid_indices]
    valid_bics = np.array(bics)[valid_indices]
    valid_aics = np.array(aics)[valid_indices]
    
    if len(valid_K) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_K, valid_bics, label='BIC', marker='o', linewidth=2)
        plt.plot(valid_K, valid_aics, label='AIC', marker='s', linewidth=2)
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Information Criterion Score")
        plt.title("GMM Model Selection: AIC & BIC Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Find optimal K values
        optimal_bic_k = valid_K[np.argmin(valid_bics)]
        optimal_aic_k = valid_K[np.argmin(valid_aics)]
        
        print(f"\n=== RESULTS ===")
        print(f"Optimal K by BIC: {optimal_bic_k}")
        print(f"Optimal K by AIC: {optimal_aic_k}")
        
        plt.axvline(x=optimal_bic_k, color='blue', linestyle='--', alpha=0.7, label=f'BIC optimal: K={optimal_bic_k}')
        plt.axvline(x=optimal_aic_k, color='orange', linestyle='--', alpha=0.7, label=f'AIC optimal: K={optimal_aic_k}')
        plt.legend()
        
        plt.show()
    else:
        print("No successful GMM runs to plot.")

if __name__ == "__main__":
    asyncio.run(main())