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
from k_means import kmeans

async def main():
    # --- Use reusable data pipeline ---
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return

    # --- K-Means + Elbow Method ---
    inertias = []
    for K in range(2, 16):
        _, _, inertia = kmeans(reduced_vectors, n_clusters=K)
        inertias.append(inertia)

    # --- Plot Elbow ---
    plt.plot(range(2, 16), inertias, 'o-')
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method (using PCA-reduced features)")
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
