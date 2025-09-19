# K-Means Clustering Evaluation on Music Feature Data

## Overview

This project applies K-Means clustering to a dataset of music tracks, each represented by a high-dimensional feature vector derived from audio analysis and metadata. The feature vectors originally comprised **223 features per song**.  
**Dimensionality reduction via PCA** (Principal Component Analysis) was performed prior to clustering to mitigate the challenges of high-dimensional data and retain the most informative components.

## Clustering Methodology

- **Algorithm:** K-Means
- **Feature Vector Size:** 223 features (reduced with PCA before clustering)
- **Cluster Count (k):** Selected using the Elbow Method
- **Evaluation Metrics:**
  - **Silhouette Score**
  - **Cluster Size Distribution**

## Results

### 1. Cluster Size Distribution

| Cluster | Number of Songs |
| ------- | --------------: |
| 0       |            1445 |
| 1       |             119 |
| 2       |            1513 |
| 3       |            2115 |
| 4       |            1682 |
| 5       |            1089 |

- Clusters are somewhat imbalanced. For example, cluster 1 is much smaller than others, and cluster 3 is the largest.

### 2. Silhouette Score

- **Silhouette Score:** `0.0053`

**Interpretation:**  
The silhouette score, ranging from -1 to 1, measures both how close each point in a cluster is to points in its own cluster vs. those in other clusters.

- A score **close to 1** means well-separated clusters.
- A score **around 0** (like 0.0053 here) means clusters are highly overlapping or poorly defined.

## Why Are the Scores Low?

- **Even with PCA, Music Data Remains Complex:**  
  While PCA helps reduce dimensionality and noise, music data often lacks clear, hard boundaries between categories—musical characteristics may blend and overlap, making distinct clusters difficult.
- **Feature Quality:**  
  Not all extracted features may be equally relevant for perceptual grouping. Some features may introduce noise or redundancy, even after PCA.
- **K-Means Limitations:**  
  K-Means assumes clusters are spherical and equally sized, which may not fit the true structure of music data.
- **Cluster Structure in Music:**  
  Musical genres, moods, or styles often overlap, so clusters may not be well-separated in any feature space.

## Graphical Representation

![K-means Plot](../../images/k-means.png)

So the visualization supports what we mean: with these complicated yet overlapping music features (even after PCA), K-Means doesn’t find well-separated groups in this music dataset.

## Recommendations

- Consider further feature engineering or selection to focus on the most musically meaningful features.
- Try alternative clustering algorithms (GMM, DBSCAN, Agglomerative) that may find structure K-Means misses.
- Inspect cluster quality visually (e.g., 2D/3D projections from PCA or t-SNE).
- Experiment with different numbers of principal components in PCA and different values of k.

## Conclusion

The low silhouette score suggests that, for this dataset and these features (even after PCA), K-Means is not able to find well-separated clusters. This is a common challenge with complex, high-dimensional data like music. Further exploration with other methods and feature engineering is recommended.
