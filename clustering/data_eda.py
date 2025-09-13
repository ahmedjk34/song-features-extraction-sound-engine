import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def data_eda(songs, tsne_sample_size=500, corr_threshold=0.95, random_state=42, plot=True):
    """
    Perform exploratory data analysis on song feature vectors.
    - Computes correlation matrix, identifies highly correlated features.
    - Analyzes variance to find low-information dimensions.
    - Prints group-wise statistics.
    - Generates t-SNE visualization on a random subset.
    """
    if not songs or len(songs) == 0:
        print("[WARN] No songs to analyze in EDA.")
        return

    # 1. Prepare feature matrix
    feature_matrix = np.array([song['feature_vector'] for song in songs])
    n_samples, n_features = feature_matrix.shape
    print(f"[INFO] EDA on {n_samples} songs, {n_features} features.")

    # 2. Correlation matrix & redundancy
    df = pd.DataFrame(feature_matrix)
    corr = df.corr()
    print("[INFO] Correlation matrix computed.")

    # Flag highly correlated pairs
    high_corr = np.where((corr.values > corr_threshold) & (corr.values < 1.0))
    found_corr = False
    for i, j in zip(*high_corr):
        if i < j:
            print(f"[WARN] Features {i} & {j} are highly correlated (corr={corr.iloc[i,j]:.3f})")
            found_corr = True
    if not found_corr:
        print("[INFO] No highly correlated feature pairs found above threshold.")

    # 3. Feature variance (low-information detection)
    variances = np.var(feature_matrix, axis=0)
    low_var_idx = np.where(variances < 1e-5)[0]
    if len(low_var_idx) > 0:
        print(f"[WARN] Low-variance (near-constant) features: {low_var_idx.tolist()}")
    else:
        print("[INFO] No low-variance features found.")

    # 4. Group statistics (mean, std, min, max per group)
    groups = {
        "vggish": (0, 128),
        "mfcc": (128, 142),
        "spectral": (142, 200),
        "rhythmic": (200, 223)
    }
    for group, (start, end) in groups.items():
        group_data = feature_matrix[:, start:end]
        print(f"[GROUP: {group.upper()}] mean={group_data.mean():.3f}, std={group_data.std():.3f}, min={group_data.min():.3f}, max={group_data.max():.3f}")

    # 5. t-SNE visualization (on random subset)
    if plot:
        # Only plot if matplotlib is available and enough data
        if n_samples >= 2:
            subset_size = min(tsne_sample_size, n_samples)
            idx = np.random.RandomState(random_state).choice(n_samples, subset_size, replace=False)
            sample = feature_matrix[idx]
            print(f"[INFO] Running t-SNE on random subset of {subset_size} samples...")
            tsne = TSNE(n_components=2, random_state=random_state)
            emb = tsne.fit_transform(sample)
            plt.figure(figsize=(7, 7))
            plt.scatter(emb[:, 0], emb[:, 1], alpha=0.6, s=10)
            plt.title("t-SNE projection of feature vectors")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.tight_layout()
            plt.show()
        else:
            print("[WARN] Not enough samples for t-SNE plot.")

    print("[INFO] EDA complete.")