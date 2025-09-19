import asyncio
from data_extraction import get_songs_from_db, verify_features, data_quality_report
from data_eda import data_eda
from feature_preprocessing import process_and_return_vector
from sklearn.decomposition import PCA
import numpy as np

async def main():
    # Data extraction and verification
    songs = await get_songs_from_db()
    if songs is None or len(songs) == 0:
        return
    
    verified_songs = verify_features(songs)
    data_quality_report(verified_songs)
    data_eda(verified_songs)

    # Feature processing
    processed_vectors = [
        process_and_return_vector(song, verified_songs)
        for song in verified_songs
    ]
    processed_vectors = np.array(processed_vectors)

    # --- PCA Reduction ---
    # Choose number of components to preserve >90% variance but not too tiny dimension
    pca = PCA(n_components=0.93, svd_solver='full')  # Keep 93% of variance, adjust as you wish!
    reduced_vectors = pca.fit_transform(processed_vectors)
    print(f"PCA reduced shape: {reduced_vectors.shape} (from {processed_vectors.shape[1]} features)")

    # Log 5 sample reduced vectors
    print("Sample (first 5 rows) of PCA-reduced features:")
    for i, row in enumerate(reduced_vectors[:5]):
        print(f"Row {i}: {np.round(row, 3)}")

if __name__ == "__main__":
    asyncio.run(main())