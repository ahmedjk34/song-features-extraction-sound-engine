import os
import json
from download_song import download_song
from extract_features import extract_features
from songs import songs

def main():
    results = []
    for i, (song, artist) in enumerate(songs):
        filename = f"temp_{i}.webm"
        try:
            print(f"🎵 Downloading: {song} - {artist}")
            download_song(song, artist, filename)
            
            print(f"🎧 Extracting features for: {song}")
            features = extract_features(filename)
            
            results.append({
                "song": song,
                "artist": artist,
                "audio_features": features
            })
            
        except Exception as e:
            print(f"❌ Failed: {song} - {artist} -> {e}")
            
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    with open("extracted_features.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ Done. Output saved to extracted_features.json")

if __name__ == "__main__":
    main()