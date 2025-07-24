def flatten_audio_features(audio_features):
    fv = []

    # 1. Scalars
    fv.append(audio_features["duration_sec"])
    fv.append(audio_features["segments"])
    fv.append(audio_features["global_bpm"])

    # 2. bpm_segments
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["bpm_segments"][key])

    # 3. chroma_stft
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["chroma_stft"][key])

    # 4. spectral_centroid
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["spectral_centroid"][key])

    # 5. spectral_bandwidth
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["spectral_bandwidth"][key])

    # 6. rolloff
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["rolloff"][key])

    # 7. zero_crossing_rate
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["zero_crossing_rate"][key])

    # 8. rms
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["rms"][key])

    # 9. mfcc_mean (13 pairs: mean, std)
    for coeff in audio_features["mfcc_mean"]:
        fv.append(coeff["mean"])
        fv.append(coeff["std"])

    # 10. mfcc_std (13 pairs: mean, std)
    for coeff in audio_features["mfcc_std"]:
        fv.append(coeff["mean"])
        fv.append(coeff["std"])

    # 11. onset_env_mean
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["onset_env_mean"][key])

    # 12. onset_count
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["onset_count"][key])

    # 13. loudness
    for key in ["mean", "std", "min", "max"]:
        fv.append(audio_features["loudness"][key])

    # 14. vggish_emb
    for val in audio_features["vggish_emb"]:
        fv.append(val)

    return fv

# Usage:
# flat_vector = flatten_audio_features(audio_features)
# # flat_vector is a list of 223 floats, ready for SQL storage



import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def parse_feature_vector(vec, artist_id=None, idx=None):
    """Parse a feature vector from stringified JSON or list to a numpy array of length 223."""
    try:
        if isinstance(vec, str):
            vec = json.loads(vec)
        vec = np.asarray(vec, dtype=np.float64)
        if vec.size == 223:
            return vec
        else:
            logger.warning(f"Vector for artist {artist_id}, row {idx} is not length 223 (size={vec.size}). Skipped.")
            return None
    except Exception as e:
        logger.warning(f"Could not parse vector for artist {artist_id}, row {idx}: {e}")
        return None

def average_feature_vectors(vectors):
    """Average a list of numpy arrays (each shape (223,))."""
    if not vectors:
        return None
    stacked = np.stack(vectors)
    avg = stacked.mean(axis=0)
    return avg

def vector_to_json_str(vec):
    """Converts a numpy array (or list) to a JSON string, rounding floats for compactness."""
    return json.dumps([round(float(x), 8) for x in vec])