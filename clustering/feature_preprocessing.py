import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import RobustScaler

# --- Feature group boundaries and weights (based on your flattening order) ---
# 0: duration_sec
# 1: segments
# 2: global_bpm
# 3-6: bpm_segments (mean, std, min, max)
# 7-10: chroma_stft (mean, std, min, max)
# 11-14: spectral_centroid (mean, std, min, max)
# 15-18: spectral_bandwidth (mean, std, min, max)
# 19-22: rolloff (mean, std, min, max)
# 23-26: zero_crossing_rate (mean, std, min, max)
# 27-30: rms (mean, std, min, max)
# 31-56: mfcc_mean (13 pairs: mean, std)
# 57-82: mfcc_std  (13 pairs: mean, std)
# 83-86: onset_env_mean (mean, std, min, max)
# 87-90: onset_count (mean, std, min, max)
# 91-94: loudness (mean, std, min, max)
# 95-222: vggish_emb (128 dims)

# Feature group slices and weights
VGGISH_SLICE = slice(95, 223)    # 128 dims
MFCC_SLICE   = slice(31, 83)     # 52 dims (26 means+stds)
SPECTRAL_SLICE = slice(11, 31)   # 20 dims
RHYTHMIC_SLICE = slice(3, 11)    # 8 dims

GROUPS = {
    "vggish":   {"slice": VGGISH_SLICE,   "weight": 1.2},
    "mfcc":     {"slice": MFCC_SLICE,     "weight": 1.05},
    "spectral": {"slice": SPECTRAL_SLICE, "weight": 0.85},
    "rhythmic": {"slice": RHYTHMIC_SLICE, "weight": 1.1}
}

def clean_feature_vector(vec):
    """Ensure the vector is valid, numeric, nan/inf-free, and length 223."""
    vec = np.asarray(vec, dtype=np.float64)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    if vec.size < 223:
        vec = np.pad(vec, (0, 223-vec.size), mode='constant')
    elif vec.size > 223:
        vec = vec[:223]
    return vec

def process_and_return_vector(all_songs, groups=GROUPS, scaling_method='standard'):
    """
    FIXED: Robust feature preprocessing that handles extreme values properly.
    
    Args:
        all_songs: List of song dictionaries
        groups: Feature group configuration
        scaling_method: 'standard', 'robust', or 'minmax'
    
    Returns:
        processed_vectors: Properly scaled and weighted feature matrix
    """
    # Step 1: Clean all feature vectors
    raw_vectors = np.array([clean_feature_vector(s['feature_vector']) for s in all_songs])
    print(f"Raw feature matrix shape: {raw_vectors.shape}")
    print(f"Raw features - min: {raw_vectors.min():.3f}, max: {raw_vectors.max():.3f}, std: {raw_vectors.std():.3f}")
    
    # Step 2: Apply appropriate scaling to handle extreme ranges
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'robust':
        # Robust scaler is better for features with outliers
        scaler = RobustScaler()
    else:  # minmax
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-2, 2))
    
    scaled_vectors = scaler.fit_transform(raw_vectors)
    print(f"After scaling - min: {scaled_vectors.min():.3f}, max: {scaled_vectors.max():.3f}, std: {scaled_vectors.std():.3f}")
    
    # Step 3: Apply weights CAREFULLY (moderate weighting to avoid extreme values)
    weighted_vectors = np.copy(scaled_vectors)
    
    for group, cfg in groups.items():
        sl = cfg["slice"]
        # Use moderate weights to avoid numerical instability
        moderate_weight = 1.0 + 0.1 * (cfg["weight"] - 1.0)  # Scale down weights
        weighted_vectors[:, sl] = scaled_vectors[:, sl] * moderate_weight
        
        group_data = weighted_vectors[:, sl]
        print(f"{group} (weight {moderate_weight:.2f}): "
              f"range=[{group_data.min():.3f}, {group_data.max():.3f}], "
              f"std={group_data.std():.3f}")
    
    # Step 4: Final check and clipping for extreme values
    weighted_vectors = np.clip(weighted_vectors, -10, 10)
    print(f"Final weighted features - min: {weighted_vectors.min():.3f}, max: {weighted_vectors.max():.3f}")
    
    return weighted_vectors
