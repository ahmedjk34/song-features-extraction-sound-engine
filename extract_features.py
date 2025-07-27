import time
import librosa
import numpy as np
import torch
from torchvggish import vggish, vggish_input
import logging
from typing import Tuple, List, Dict, Any

# Constants
VGGISH_EMBED_SIZE = 128
N_MFCC = 13

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_audio(audio_path: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file using librosa.
    Args:
        audio_path: Path to the audio file.
        sr: Target sample rate.
    Returns:
        y: Audio time series.
        sr: Sample rate.
    Raises:
        RuntimeError: If loading fails.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"librosa.load failed: {e}")

# Load VGGish model once (CPU)
vggish_model = vggish()
vggish_model.eval()

def get_vggish_embedding(y_seg: np.ndarray, sr: int) -> List[float]:
    """
    Extracts VGGish embedding from an audio segment.
    Args:
        y_seg: Audio segment.
        sr: Sample rate.
    Returns:
        List of floats representing the embedding.
    """
    try:
        # VGGish expects mono, 16kHz float32
        if sr != 16000:
            y_seg = librosa.resample(y_seg, orig_sr=sr, target_sr=16000)
            sr = 16000
        if np.max(np.abs(y_seg)) > 0:
            y_seg = y_seg / np.max(np.abs(y_seg))

        # Ensure y_seg is a numpy array, not a tensor
        if torch.is_tensor(y_seg):
            y_seg = y_seg.detach().cpu().numpy()
        y_seg = y_seg.astype(np.float32)

        # Log mel patches
        examples = vggish_input.waveform_to_examples(y_seg, sr)
        if examples.shape[0] == 0:
            # Too short for VGGish, return zeros
            return [0.0] * VGGISH_EMBED_SIZE

        # Ensure examples is a tensor
        if not torch.is_tensor(examples):
            examples = torch.from_numpy(examples)

        with torch.no_grad():
            emb = vggish_model(examples)

        # Mean pooling over patches and convert to numpy
        emb_mean = emb.mean(dim=0)
        result = emb_mean.detach().cpu().numpy()
        return result.tolist()
    except Exception as e:
        logging.error(f"VGGish embedding error: {e}")
        return [0.0] * VGGISH_EMBED_SIZE

def calculate_features(y_seg: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Calculates audio features for a segment.
    Args:
        y_seg: Audio segment.
        sr: Sample rate.
    Returns:
        Dictionary of features.
    Raises:
        RuntimeError: If feature extraction fails.
    """
    try:
        bpm, _ = librosa.beat.beat_track(y=y_seg, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y_seg, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_seg, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y_seg, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y_seg)
        rms = librosa.feature.rms(y=y_seg)
        mfccs = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=N_MFCC)
        onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr)
        onset_count = len(librosa.onset.onset_detect(y=y_seg, sr=sr))
        loudness = librosa.amplitude_to_db(y_seg, ref=np.max)
        loudness_mean = float(np.mean(loudness))
        loudness_std = float(np.std(loudness))
        loudness_max = float(np.max(loudness))
        loudness_min = float(np.min(loudness))
        try:
            vggish_emb = get_vggish_embedding(y_seg, sr)
        except Exception as vgg_err:
            logging.error(f"VGGish error: {vgg_err}")
            vggish_emb = [0.0] * VGGISH_EMBED_SIZE
        return {
            "bpm": float(bpm),
            "chroma_stft": float(np.mean(chroma_stft)),
            "spectral_centroid": float(np.mean(spectral_centroid)),
            "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
            "rolloff": float(np.mean(rolloff)),
            "zero_crossing_rate": float(np.mean(zero_crossing_rate)),
            "rms": float(np.mean(rms)),
            "mfcc_mean": [float(np.mean(m)) for m in mfccs],
            "mfcc_std": [float(np.std(m)) for m in mfccs],
            "onset_env_mean": float(np.mean(onset_env)),
            "onset_count": float(onset_count),
            "loudness_mean": loudness_mean,
            "loudness_std": loudness_std,
            "loudness_max": loudness_max,
            "loudness_min": loudness_min,
            "vggish_emb": vggish_emb if isinstance(vggish_emb, list) else vggish_emb.tolist(),
        }
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")

def aggregate_features(features_per_segment: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    """
    Aggregates a feature across segments.
    Args:
        features_per_segment: List of feature dicts.
        key: Feature key.
    Returns:
        Dict with mean, std, min, max.
    """
    vals = [seg[key] for seg in features_per_segment if key in seg]
    if not vals:
        return {}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def aggregate_mfcc(features_per_segment: List[Dict[str, Any]], key: str) -> List[Dict[str, float]]:
    """
    Aggregates MFCC features across segments.
    Args:
        features_per_segment: List of feature dicts.
        key: MFCC key.
    Returns:
        List of dicts with mean and std for each MFCC coefficient.
    """
    mfcc_arrs = [seg[key] for seg in features_per_segment if key in seg]
    if not mfcc_arrs:
        return []
    mfcc_arrs = np.array(mfcc_arrs)
    return [
        {
            "mean": float(np.mean(mfcc_arrs[:, i])),
            "std": float(np.std(mfcc_arrs[:, i])),
        }
        for i in range(mfcc_arrs.shape[1])
    ]

def aggregate_vggish(features_per_segment: List[Dict[str, Any]]) -> List[float]:
    """
    Aggregates VGGish embeddings across segments.
    Args:
        features_per_segment: List of feature dicts.
    Returns:
        Mean embedding as a list.
    """
    vgg_arrs = [seg["vggish_emb"] for seg in features_per_segment if "vggish_emb" in seg]
    if not vgg_arrs:
        return [0.0] * VGGISH_EMBED_SIZE
    vgg_arrs = np.array(vgg_arrs)
    return vgg_arrs.mean(axis=0).tolist()

def extract_features(
    audio_path: str,
    segment_duration: float = 10.0,
    min_rms: float = 0.01
) -> Dict[str, Any]:
    """
    Extracts audio features for the entire audio and for segments.
    Returns a dict with aggregated statistics for each feature.
    BPM is computed globally and for valid segments.
    """
    try:
        y, sr = load_audio(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        total_segments = max(1, int(duration // segment_duration))
        features_per_segment = []

        # Compute global BPM for the entire song
        global_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)

        for seg in range(total_segments):
            start_sample = int(seg * segment_duration * sr)
            end_sample = int(min((seg + 1) * segment_duration * sr, len(y)))
            y_seg = y[start_sample:end_sample]

            if len(y_seg) < sr:  # Skip too-short segments
                continue

            try:
                features = calculate_features(y_seg, sr)
                # Only accept BPM from segments with sufficient energy (rms)
                if features["rms"] >= min_rms:
                    features_per_segment.append(features)
            except RuntimeError as e:
                logging.error(f"Segment {seg} error: {e}")
                continue  # Skip problematic segment

        return {
            "duration_sec": float(duration),
            "segments": total_segments,
            "global_bpm": float(global_bpm),
            "bpm_segments": aggregate_features(features_per_segment, "bpm"),
            "chroma_stft": aggregate_features(features_per_segment, "chroma_stft"),
            "spectral_centroid": aggregate_features(features_per_segment, "spectral_centroid"),
            "spectral_bandwidth": aggregate_features(features_per_segment, "spectral_bandwidth"),
            "rolloff": aggregate_features(features_per_segment, "rolloff"),
            "zero_crossing_rate": aggregate_features(features_per_segment, "zero_crossing_rate"),
            "rms": aggregate_features(features_per_segment, "rms"),
            "mfcc_mean": aggregate_mfcc(features_per_segment, "mfcc_mean"),
            "mfcc_std": aggregate_mfcc(features_per_segment, "mfcc_std"),
            "onset_env_mean": aggregate_features(features_per_segment, "onset_env_mean"),
            "onset_count": aggregate_features(features_per_segment, "onset_count"),
            "loudness": {
                "mean": aggregate_features(features_per_segment, "loudness_mean").get("mean", 0.0) if features_per_segment else 0.0,
                "std": aggregate_features(features_per_segment, "loudness_std").get("mean", 0.0) if features_per_segment else 0.0,
                "min": aggregate_features(features_per_segment, "loudness_min").get("min", 0.0) if features_per_segment else 0.0,
                "max": aggregate_features(features_per_segment, "loudness_max").get("max", 0.0) if features_per_segment else 0.0,
            },
            "vggish_emb": aggregate_vggish(features_per_segment),
        }
    except RuntimeError as e:
        return {"error": str(e)}