from essentia.standard import MusicExtractor
import numpy as np

def safe_get(pool, key, default=0.0):
    """Safely get a value from Essentia's Pool"""
    try:
        return pool[key]
    except:
        return default

def get_stat(pool, key, default=0.0):
    """Safely get a statistical feature from the Pool"""
    try:
        return float(pool[key])
    except:
        return default

def extract_spotify_features(audio_path: str):
    """Extract Spotify-style audio features using Essentia"""
    
    extractor = MusicExtractor(
        lowlevelStats=['mean', 'var', 'stdev', 'median', 'min', 'max'],
        rhythmStats=['mean', 'var', 'stdev', 'median', 'min', 'max'],
        tonalStats=['mean', 'var', 'stdev', 'median', 'min', 'max']
    )
    
    try:
        features, meta = extractor(audio_path)
        
        # Extract basic rhythm features
        bpm = safe_get(features, 'rhythm.bpm', 120.0)
        
        # Extract tonal features
        key_key = safe_get(features, 'tonal.key_key', 'C')
        key_scale = safe_get(features, 'tonal.key_scale', 'major')
        
        # Extract loudness
        loudness = get_stat(features, 'lowlevel.loudness.mean', -20.0)
        
        # Extract duration
        duration = safe_get(features, 'metadata.audio_properties.length', 0.0)
        
        # Approximate danceability using rhythm and spectral features
        spectral_centroid = get_stat(features, 'lowlevel.spectral_centroid.mean', 1000.0)
        spectral_rolloff = get_stat(features, 'lowlevel.spectral_rolloff.mean', 2000.0)
        
        danceability = min(1.0, max(0.0, (bpm - 60) / 140 * 0.7 + 0.3))
        
        # Approximate energy using spectral features
        energy = min(1.0, max(0.0, (spectral_centroid - 500) / 3000))
        
        # Approximate valence using tonal features
        valence = 0.6 if key_scale == 'major' else 0.4
        
        # Approximate acousticness (inverse of spectral complexity)
        acousticness = max(0.0, min(1.0, 1.0 - (spectral_rolloff - 1000) / 10000))
        
        # Approximate instrumentalness using spectral features
        spectral_complexity = get_stat(features, 'lowlevel.spectral_complexity.mean', 0.5)
        instrumentalness = min(1.0, max(0.0, spectral_complexity))
        
        # Approximate liveness using dynamic range
        dynamic_range = get_stat(features, 'lowlevel.dynamic_complexity', 0.5)
        liveness = min(1.0, max(0.0, dynamic_range))
        
        # Get time signature (default to 4 if not found)
        time_signature = safe_get(features, 'rhythm.time_signature', 4)
        
        # Return Spotify-compatible features
        return {
            "danceability": float(danceability),
            "valence": float(valence),
            "energy": float(energy),
            "acousticness": float(acousticness),
            "instrumentalness": float(instrumentalness),
            "liveness": float(liveness),
            "bpm": float(bpm),
            "tempo": float(bpm),
            "key": str(key_key),
            "mode": str(key_scale),
            "loudness": float(loudness),
            "duration_sec": float(duration),
            "time_signature": int(time_signature),
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_complexity": float(spectral_complexity)
        }
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return {
            "danceability": 0.5,
            "valence": 0.5,
            "energy": 0.5,
            "acousticness": 0.5,
            "instrumentalness": 0.5,
            "liveness": 0.1,
            "bpm": 120.0,
            "tempo": 120.0,
            "key": "C",
            "mode": "major",
            "loudness": -20.0,
            "duration_sec": 0.0,
            "time_signature": 4,
            "spectral_centroid": 1000.0,
            "spectral_rolloff": 2000.0,
            "spectral_complexity": 0.5,
            "error": str(e)
        }

def extract_detailed_features(audio_path: str):
    """Extract detailed audio features using Essentia"""
    
    extractor = MusicExtractor(
        lowlevelStats=['mean', 'var', 'stdev', 'median', 'min', 'max'],
        rhythmStats=['mean', 'var', 'stdev', 'median', 'min', 'max'],
        tonalStats=['mean', 'var', 'stdev', 'median', 'min', 'max']
    )
    
    try:
        features, meta = extractor(audio_path)
        
        return {
            "essentia_features": dict(features),  # convert Pool to dict
            "metadata": dict(meta),
            "spotify_compatible": extract_spotify_features(audio_path)
        }
        
    except Exception as e:
        print(f"Error extracting detailed features: {e}")
        return {
            "error": str(e),
            "spotify_compatible": extract_spotify_features(audio_path)
        }
