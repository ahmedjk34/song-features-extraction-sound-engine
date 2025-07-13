import librosa
import numpy as np

def load_audio(audio_path: str, sr: int = 44100):
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"librosa.load failed: {e}")

def calculate_features(y_seg, sr):
    try:
        bpm, _ = librosa.beat.beat_track(y=y_seg, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y_seg, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_seg, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y_seg, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y_seg)
        rms = librosa.feature.rms(y=y_seg)
        mfccs = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13)

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
        }
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")

def aggregate_features(features_per_segment, key):
    vals = [seg[key] for seg in features_per_segment if key in seg]
    if not vals:
        return {}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def aggregate_mfcc(features_per_segment, key):
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

def extract_features(audio_path: str, segment_duration: float = 10.0, min_rms: float = 0.01):
    """
    Extracts audio features for the entire audio and for segments.
    Returns a dict with aggregated statistics for each feature.
    BPM is computed globally and for valid segments. since mean BPM can vary significantly across segments,
    """
    try:
        y, sr = load_audio(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        total_segments = max(1, int(duration // segment_duration))
        features_per_segment = []

        # 1. Compute global BPM for the entire song
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
                continue  # Skip problematic segment

        return {
            "duration_sec": float(duration),
            "segments": total_segments,
            "global_bpm": float(global_bpm),  # Main BPM for the song
            "bpm_segments": aggregate_features(features_per_segment, "bpm"),  # Variability (optional)
            "chroma_stft": aggregate_features(features_per_segment, "chroma_stft"),
            "spectral_centroid": aggregate_features(features_per_segment, "spectral_centroid"),
            "spectral_bandwidth": aggregate_features(features_per_segment, "spectral_bandwidth"),
            "rolloff": aggregate_features(features_per_segment, "rolloff"),
            "zero_crossing_rate": aggregate_features(features_per_segment, "zero_crossing_rate"),
            "rms": aggregate_features(features_per_segment, "rms"),
            "mfcc_mean": aggregate_mfcc(features_per_segment, "mfcc_mean"),
            "mfcc_std": aggregate_mfcc(features_per_segment, "mfcc_std"),
        }
    except RuntimeError as e:
        return {"error": str(e)}