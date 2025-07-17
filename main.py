import os
import json
import time
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


def process_song(song, artist, index, temp_dir="temp_files", global_start=None):
    """Downloads a song, extracts features, and handles cleanup."""
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.join(temp_dir, f"temp_{index}.webm")

    try:
        start_time = time.perf_counter()
        elapsed = lambda: time.perf_counter() - global_start if global_start else 0

        print(f"[+{elapsed():.2f}s] 🎵 Downloading: {song} - {artist}")
        download_song(song, artist, filename)
        print(f"[+{elapsed():.2f}s] ✅ Download complete: {song} - {artist}")

        print(f"[+{elapsed():.2f}s] 🎧 Extracting features for: {song}")
        features = extract_features(filename)
        print(f"[+{elapsed():.2f}s] ✅ Feature extraction complete: {song}")

        if "error" in features:
            log_error(song, artist, features["error"])
            return None

        duration = time.perf_counter() - start_time
        print(f"[+{elapsed():.2f}s] ⏱️ Process duration for this song: {duration:.2f} seconds")

        return {
            "song": song,
            "artist": artist,
            "audio_features": features,
        }

    except Exception as e:
        print(f"[+{elapsed():.2f}s] ❌ Failed: {song} - {artist} -> {e}")
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
    start_all = time.perf_counter()

    for i, (song, artist) in enumerate(songs):
        result = process_song(song, artist, i, global_start=start_all)
        if result:
            results.append(result)

    save_results(results)
    total_duration = time.perf_counter() - start_all
    print(f"[+{total_duration:.2f}s] 🎉 All songs processed in {total_duration:.2f} seconds")


if __name__ == "__main__":
    main()