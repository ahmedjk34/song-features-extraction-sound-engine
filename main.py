import os
import json
from download_song import download_song
from extract_features import extract_features
from songs import songs


def log_error(song, artist, error, log_file="error_log.json"):
    """Logs errors to a JSON file."""
    entry = {"song": song, "artist": artist, "error": error}
    logs = []

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)

    logs.append(entry)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def process_song(song, artist, index, temp_dir="temp_files"):
    """Downloads a song, extracts features, and handles cleanup."""
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.join(temp_dir, f"temp_{index}.webm")

    try:
        print(f"🎵 Downloading: {song} - {artist}")
        download_song(song, artist, filename)

        print(f"🎧 Extracting features for: {song}")
        features = extract_features(filename)

        if "error" in features:
            log_error(song, artist, features["error"])
            return None

        return {
            "song": song,
            "artist": artist,
            "audio_features": features,
        }

    except Exception as e:
        print(f"❌ Failed: {song} - {artist} -> {e}")
        log_error(song, artist, str(e))
        return None

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def save_results(results, output_file="extracted_features.json"):
    """Saves the extracted features to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Done. Output saved to {output_file}")


def main():
    """Main function to process all songs."""
    results = []

    for i, (song, artist) in enumerate(songs):
        result = process_song(song, artist, i)
        if result:
            results.append(result)

    save_results(results)


if __name__ == "__main__":
    main()