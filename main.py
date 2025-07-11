import tempfile
import os
from download_audio import download_audio
from extract_features import extract_spotify_features

def analyze_song(song: str, artist: str):
    """Analyze a song by downloading it and extracting audio features"""
    print(f"Analyzing: {song} by {artist}")
    
    # Create an ephemeral temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Download the audio
        print("Downloading audio...")
        download_audio(song, artist, temp_path)
        
        # Extract features
        print("Extracting features...")
        features = extract_spotify_features(temp_path)
        
        # Display results
        print("\n🎧 Spotify-style Audio Features:")
        for k, v in features.items():
            if isinstance(v, float):
                print(f"{k:20}: {v:.4f}")
            else:
                print(f"{k:20}: {v}")
        
        return features
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# Run an example
if __name__ == "__main__":
    analyze_song("Starboy", "The Weeknd")