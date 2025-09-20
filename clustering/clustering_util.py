from data_extraction import get_songs_from_db, verify_features, data_quality_report
from data_eda import data_eda
from feature_preprocessing import process_and_return_vector, get_feature_stats_after_weighting
from sklearn.decomposition import PCA
import numpy as np

async def prepare_data_pipeline():
    """
    Fetches songs, verifies features, performs EDA, processes feature vectors,
    and applies PCA to reduce dimensionality.
    
    Returns:
        reduced_vectors (np.ndarray): PCA-reduced feature vectors
        pca (PCA object): fitted PCA object
        verified_songs (list): list of verified song dicts
    """
    # Data extraction
    songs = await get_songs_from_db()
    if songs is None or len(songs) == 0:
        print("No songs found in DB.")
        return None, None, None
    
    # Feature verification
    verified_songs = verify_features(songs)
    data_quality_report(verified_songs)
    data_eda(verified_songs)

    # Feature processing
    processed_vectors = np.array([
        process_and_return_vector(song, verified_songs)
        for song in verified_songs
    ])

    # OPTIONAL: Debug check to verify weights are working
    print("\n=== DEBUGGING: Checking if weights are preserved ===")
    get_feature_stats_after_weighting(verified_songs)
    print("=== End Debug Check ===\n")

    # PCA Reduction
    pca = PCA(n_components=0.93, svd_solver='full')
    reduced_vectors = pca.fit_transform(processed_vectors)
    
    print(f"PCA reduced shape: {reduced_vectors.shape} (from {processed_vectors.shape[1]} features)")
    print("Sample (first 5 rows) of PCA-reduced features:")
    for i, row in enumerate(reduced_vectors[:5]):
        print(f"Row {i}: {np.round(row, 3)}")

    return reduced_vectors, pca, verified_songs


