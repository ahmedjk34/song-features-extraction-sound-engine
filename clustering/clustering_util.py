from data_extraction import get_songs_from_db, verify_features, data_quality_report
from data_eda import data_eda
from feature_preprocessing import process_and_return_vector
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

    # Data extraction (keep same)
    songs = await get_songs_from_db()
    if songs is None or len(songs) == 0:
        print("No songs found in DB.")
        return None, None, None
    
    # Feature verification (keep same)
    verified_songs = verify_features(songs)
    data_quality_report(verified_songs)
    data_eda(verified_songs)
    
    # NEW:
    processed_vectors = process_and_return_vector(verified_songs, scaling_method='robust')
    
    # More conservative PCA (MODIFY THIS):
    pca = PCA(n_components=min(50, processed_vectors.shape[1] // 2), svd_solver='full')
    reduced_vectors = pca.fit_transform(processed_vectors)
    
    print(f"PCA reduced shape: {reduced_vectors.shape} (from {processed_vectors.shape[1]} features)")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return reduced_vectors, pca, verified_songs