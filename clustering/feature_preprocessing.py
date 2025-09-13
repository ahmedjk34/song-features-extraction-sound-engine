import numpy as np
from sklearn.preprocessing import StandardScaler

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
SPECTRAL_SLICE = slice(11, 31)   # 20 dims (spectral_centroid, bandwidth, rolloff, zcr, rms: 5x4)
RHYTHMIC_SLICE = slice(3, 11)    # 8 dims (bpm_segments, chroma_stft: 2x4)
# (You may adjust boundaries later?)

GROUPS = {
    "vggish":   {"slice": VGGISH_SLICE,   "weight": 1.5},
    "mfcc":     {"slice": MFCC_SLICE,     "weight": 1.2},
    "spectral": {"slice": SPECTRAL_SLICE, "weight": 1.0},
    "rhythmic": {"slice": RHYTHMIC_SLICE, "weight": 1.5}
}

def fit_feature_group_scalers(X, groups=GROUPS):
    """Fit a StandardScaler for each group, returns dict of scalers."""
    scalers = {}
    for group, cfg in groups.items():
        scaler = StandardScaler()
        scaler.fit(X[:, cfg["slice"]])
        scalers[group] = scaler
    return scalers

def transform_features(X, scalers, groups=GROUPS):
    """Normalize and weight each group, returns new array."""
    X_out = np.zeros_like(X)
    for group, cfg in groups.items():
        X_out[:, cfg["slice"]] = scalers[group].transform(X[:, cfg["slice"]]) * cfg["weight"]
    return X_out

def process_and_store_features(songs, save_path="normalized_vectors.npy"):
    """
    Given a list of song dicts with 'feature_vector', normalize and weight features by group.
    Saves both original and normalized vectors to disk as numpy array of dicts.
    """
    X = np.array([song['feature_vector'] for song in songs])
    scalers = fit_feature_group_scalers(X)
    X_norm = transform_features(X, scalers)
    paired = [{"original": orig, "normalized": norm} for orig, norm in zip(X, X_norm)]
    np.save(save_path, paired)
    print(f"[INFO] Saved {len(paired)} original+normalized vector pairs to {save_path}")
    return paired
