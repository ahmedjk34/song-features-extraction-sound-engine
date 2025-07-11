from essentia.standard import MusicExtractor

def extract_spotify_features(audio_path: str):
    """Extract Spotify-style audio features using Essentia"""
    extractor = MusicExtractor(highlevelStats=True)
    features, meta = extractor(audio_path)
    
    # Extract and format features similar to Spotify's API
    return {
        "danceability": float(features['highlevel']['danceability']['all']['mean']),
        "valence": float(features['highlevel']['valence']['all']['mean']),
        "energy": float(features['highlevel']['energy']['all']['mean']),
        "acousticness": float(features['highlevel']['acousticness']['all']['mean']),
        "instrumentalness": float(features['highlevel']['instrumentalness']['all']['mean']),
        "liveness": float(features['highlevel']['liveness']['all']['mean']),
        "bpm": float(features['rhythm']['bpm']),
        "tempo": float(features['rhythm']['bpm']),
        "key": features['tonal']['key_key'],
        "mode": features['tonal']['key_scale'],
        "loudness": float(features['lowlevel']['loudness']['mean']),
        "duration_sec": float(features['metadata']['audio_properties']['length']),
        "time_signature": int(features['rhythm']['time_signature'])
    }