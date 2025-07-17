import time
import librosa
import numpy as np
import torch
from torchvggish import vggish, vggish_input

def load_audio(audio_path: str, sr: int = 44100):
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"librosa.load failed: {e}")

# Load VGGish model once (CPU)
vggish_model = vggish()
vggish_model.eval()

def get_vggish_embedding(y_seg, sr):
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
            return [0.0] * 128
        
        # Ensure examples is a tensor
        if not torch.is_tensor(examples):
            examples = torch.from_numpy(examples)
        
        with torch.no_grad():
            emb = vggish_model(examples)
        
        # Mean pooling over patches and convert to numpy
        emb_mean = emb.mean(dim=0)
        # Convert tensor to numpy array
        result = emb_mean.detach().cpu().numpy()
        return result.tolist()  # Return as list to match expected format
        
    except Exception as e:
        print(f"VGGish embedding error: {e}")
        return [0.0] * 128
    
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
        # Rhythm/tempo metrics
        onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr)
        onset_count = len(librosa.onset.onset_detect(y=y_seg, sr=sr))
        # Loudness and dynamic range
        loudness = librosa.amplitude_to_db(y_seg, ref=np.max)
        loudness_mean = float(np.mean(loudness))
        loudness_std = float(np.std(loudness))
        loudness_max = float(np.max(loudness))
        loudness_min = float(np.min(loudness))
        # VGGish
        try:
            vggish_emb = get_vggish_embedding(y_seg, sr)
        except Exception as vgg_err:
            print(f"VGGish error: {vgg_err}")
            vggish_emb = [0.0] * 128
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

def aggregate_vggish(features_per_segment):
    vgg_arrs = [seg["vggish_emb"] for seg in features_per_segment if "vggish_emb" in seg]
    if not vgg_arrs:
        return [0.0] * 128
    vgg_arrs = np.array(vgg_arrs)
    return vgg_arrs.mean(axis=0).tolist()

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
                print(f"Segment {seg} error: {e}")
                continue  # Skip problematic segment

        return {
            "duration_sec": float(duration),
            "segments": total_segments,
            "global_bpm": float(global_bpm),  # Main BPM for the song
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
                "mean": aggregate_features(features_per_segment, "loudness_mean")["mean"] if features_per_segment else 0.0,
                "std": aggregate_features(features_per_segment, "loudness_std")["mean"] if features_per_segment else 0.0,
                "min": aggregate_features(features_per_segment, "loudness_min")["min"] if features_per_segment else 0.0,
                "max": aggregate_features(features_per_segment, "loudness_max")["max"] if features_per_segment else 0.0,
            },
            "vggish_emb": aggregate_vggish(features_per_segment),  # Now a single array (mean only)
        }
    except RuntimeError as e:
        return {"error": str(e)}