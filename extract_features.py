import librosa
import numpy as np
import json

def extract_features(audio_path: str):
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # Duration and tempo
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Chroma features (harmony)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma_stft)

    # Spectral features (timbre-related)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    # RMS energy
    rms = np.mean(librosa.feature.rms(y=y))

    # MFCCs (e.g. first 13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = [float(np.mean(m)) for m in mfccs]

    return {
        "duration_sec": float(duration),
        "bpm": float(tempo),
        "chroma_stft": float(chroma_mean),
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "rolloff": float(rolloff),
        "zero_crossing_rate": float(zero_crossing_rate),
        "rms": float(rms),
        "mfcc": mfccs_mean,
    }