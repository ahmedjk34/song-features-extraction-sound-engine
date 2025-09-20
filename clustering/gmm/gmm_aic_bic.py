import sys
import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt

# --- Add parent directory to sys.path ---
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- Imports from parent directory ---
from clustering_util import prepare_data_pipeline
from gmm import gmm_em

def compute_bic(X, log_likelihood, n_params):
    N = X.shape[0]
    bic = -2 * log_likelihood + n_params * np.log(N)
    return bic

def compute_aic(log_likelihood, n_params):
    aic = -2 * log_likelihood + 2 * n_params
    return aic

async def main():
    # --- Use reusable data pipeline ---
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return

    K_range = range(2, 16)
    bics, aics, logL = [], [], []
    for K in K_range:
        means, covariances, weights, responsibilities, ll = gmm_em(reduced_vectors, K)
        D = reduced_vectors.shape[1]
        n_params = K * D                   # means
        n_params += K * D * (D + 1) // 2   # covariances (symmetric)
        n_params += K - 1                  # mixing coeffs (sum to 1)
        bics.append(compute_bic(reduced_vectors, ll, n_params))
        aics.append(compute_aic(ll, n_params))
        logL.append(ll)
        print(f"K={K}: BIC={bics[-1]:.2f}, AIC={aics[-1]:.2f}, logL={ll:.2f}")

    # --- Plot BIC and AIC curves ---
    plt.plot(list(K_range), bics, label='BIC', marker='o')
    plt.plot(list(K_range), aics, label='AIC', marker='o')
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.title("GMM Model Selection (AIC & BIC)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())