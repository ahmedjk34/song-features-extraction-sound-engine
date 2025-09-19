import numpy as np

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

# Feature group slices and new, slightly lower weights
VGGISH_SLICE = slice(95, 223)    # 128 dims
MFCC_SLICE   = slice(31, 83)     # 52 dims (26 means+stds)
SPECTRAL_SLICE = slice(11, 31)   # 20 dims (spectral_centroid, bandwidth, rolloff, zcr, rms: 5x4)
RHYTHMIC_SLICE = slice(3, 11)    # 8 dims (bpm_segments, chroma_stft: 2x4)

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

def min_max_scale(X, feature_range=(-10, 10)):
    """Scale features in X to the given feature_range along axis 0."""
    X = np.asarray(X, dtype=np.float64)
    X_min = np.nanmin(X, axis=0)
    X_max = np.nanmax(X, axis=0)
    denom = np.where(X_max - X_min == 0, 1, X_max - X_min)
    X_scaled = (X - X_min) / denom
    scale = feature_range[1] - feature_range[0]
    X_scaled = X_scaled * scale + feature_range[0]
    return X_scaled

def process_and_return_vector(song, all_songs, groups=GROUPS):
    """
    Clean, min-max scale (1-10), and weight the features for a single song.
    - all_songs: list of all song dicts (for scaling context)
    Returns: np.ndarray (shape: (223,))
    """
    # Clean all feature vectors for scaling context
    X = np.array([clean_feature_vector(s['feature_vector']) for s in all_songs])
    X_scaled = min_max_scale(X, feature_range=(1, 10))
    # Find this song's index (by object equality, fallback to clean directly)
    try:
        i = all_songs.index(song)
        x = X_scaled[i]
    except Exception:
        x = min_max_scale([clean_feature_vector(song['feature_vector'])], feature_range=(1, 10))[0]

    # Weight groups and ensure after weighting, values remain in 1–10
    x_weighted = np.copy(x)
    for group, cfg in groups.items():
        sl = cfg["slice"]
        w = cfg["weight"]
        x_weighted[sl] = x[sl] * w
        # Rescale weighted part if it leaves 1–10 range
        if x_weighted[sl].min() < 1 or x_weighted[sl].max() > 10:
            x_weighted[sl] = min_max_scale(x_weighted[sl][None, :], (1, 10))[0]

    return x_weighted