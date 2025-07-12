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
        downloaded_path = download_audio(song, artist, temp_path)
        
        # Verify download
        if not os.path.exists(downloaded_path):
            raise FileNotFoundError(f"Download failed: {downloaded_path} not found")
        
        file_size = os.path.getsize(downloaded_path)
        if file_size == 0:
            raise ValueError("Downloaded file is empty")
        
        print(f"Audio downloaded successfully: {file_size} bytes")
        
        # Extract features
        print("Extracting features...")
        features = extract_spotify_features(downloaded_path)
        
        if not features:
            raise ValueError("Feature extraction returned no results")
        
        # Display results
        print("\n🎧 Spotify-style Audio Features:")
        print("=" * 50)
        for k, v in features.items():
            if isinstance(v, float):
                print(f"{k:20}: {v:.4f}")
            else:
                print(f"{k:20}: {v}")
        print("=" * 50)
        
        return features
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        return None
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print("Temporary file cleaned up.")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temporary file: {cleanup_error}")

def main():
    """Main entry point"""
    try:
        print("🎵 Audio Feature Extraction Tool")
        print("=" * 50)
        
        # Run analysis
        features = analyze_song("Starboy", "The Weeknd")
        
        if features:
            print("\n✅ Analysis completed successfully!")
        else:
            print("\n❌ Analysis failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

# Run an example
if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)