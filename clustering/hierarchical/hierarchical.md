Here’s a polished **README.md** based on your code and results:

```markdown
# Hierarchical Clustering Model Selection Results

## Overview

This experiment evaluates **hierarchical clustering** performance on **7,963 music tracks**, each represented by **50 PCA-reduced audio features**.  
The goal was to determine the optimal number of clusters (_k_) for meaningful music categorization.

## Method

- **Dataset**: 7,963 total tracks
- **Sampling**: 1,000 randomly selected tracks for efficient computation
- **Clustering Algorithm**: Agglomerative hierarchical clustering
- **Linkage**: Average linkage with Euclidean distance
- **Evaluation Range**: k = 2 → 14 clusters
- **Metrics**:
  - **Silhouette Score** – measures cohesion & separation
  - **Calinski-Harabasz Index (CH)** – separation vs. compactness
  - **Davies-Bouldin Index (DB)** – lower is better

## Results

### Optimal k by Different Criteria

- **Silhouette Score** → **k = 2** (best score: **0.322**)
- **Calinski-Harabasz Index** → **k = 3** (best score: **21.8**)
- **Davies-Bouldin Index** → **k = 8** (best score: **1.099**)

### Performance by k Value

k= 2: Silhouette= 0.322, CH= 18.5, DB= 1.255
k= 3: Silhouette= 0.273, CH= 21.8, DB= 1.482
k= 4: Silhouette= 0.265, CH= 15.3, DB= 1.262
k= 5: Silhouette= 0.236, CH= 12.1, DB= 1.132
k= 6: Silhouette= 0.188, CH= 11.6, DB= 1.288
k= 7: Silhouette= 0.174, CH= 10.2, DB= 1.163
k= 8: Silhouette= 0.162, CH= 9.1, DB= 1.099

## Recommendation

We recommend **k = 3 clusters** as the optimal balance:

- Achieves the **highest CH index** (21.8) → best separation
- Silhouette score of **0.273** → acceptable cohesion/separation tradeoff
- Avoids trivial 2-cluster split
- Provides **more interpretable and meaningful groupings** for music categorization

## Final Clustering Applied

- Applied **k = 2 clustering** to the **full dataset (7,963 tracks)**
- Result: **2 clusters**, but with **balance score = 0.000**, indicating **heavily skewed distribution** (poor cluster balance).

➡️ While k=2 scored best on silhouette, **k=3 is recommended** for practical applications.

## Visualization

![Model Selection Results](../../images/hierarchical_model_selection.png)

_The plot shows evaluation metrics across different k values. Red lines indicate optimal points for each criterion._

## Files Generated

- `hierarchical_model_selection.png` → Visualization of evaluation metrics
- `final_hierarchical_labels.csv` → Cluster assignments for the full dataset

---
```

Do you want me to also **add a short “Interpretation” section** at the end (e.g., what these clusters might correspond to musically—like moods, genres, or energy levels)? That would make the README more useful for non-technical readers.
