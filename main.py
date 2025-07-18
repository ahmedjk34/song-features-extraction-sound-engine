import os
import time
import json
from download_song import download_song
from extract_features import extract_features
from util import flatten_audio_features

import asyncio
from libsql_client import create_client

from dotenv import load_dotenv
load_dotenv()  # Load .env at the very top!


def log_error(song_id, artist_id, item_name, artist, error, log_file="error_log.json"):
    """Logs errors to a JSON file. Will handle empty or corrupted log file gracefully."""
    entry = {
        "song_id": song_id, 
        "artist_id": artist_id, 
        "item_name": item_name, 
        "artist": artist, 
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, Exception):
            logs = []
    logs.append(entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print(f"❌ Error logged for item ID {song_id}: {error}")
    
async def process_and_save_item(client, song_id, artist_id, song_name, artist_name, 
                               item_index, total_items, temp_dir="temp_files", global_start=None):
    """Downloads an audio file, extracts features, and saves to DB."""
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.join(temp_dir, f"temp_audio_{song_id}.webm")

    try:
        start_time = time.perf_counter()
        elapsed = lambda: time.perf_counter() - global_start if global_start else 0

        print(f"\n{'='*80}")
        print(f"🎵 PROCESSING AUDIO ITEM {item_index}/{total_items} [{(item_index/total_items)*100:.1f}%]")
        print(f"📀 Track: {song_name}")
        print(f"🎤 Artist: {artist_name}")
        print(f"🆔 Item ID: {song_id} | Artist ID: {artist_id}")
        print(f"⏱️  Global Elapsed: {elapsed():.2f}s")
        print(f"{'='*80}")

        print(f"[+{elapsed():.2f}s] 📥 Starting audio download...")
        download_start = time.perf_counter()
        await asyncio.to_thread(download_song, song_name, artist_name, filename)
        download_duration = time.perf_counter() - download_start
        
        # Check file size
        file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
        file_size_mb = file_size / (1024 * 1024)

        # PATCH: Check for missing or empty file before feature extraction!
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            log_error(song_id, artist_id, song_name, artist_name, f"No audio file found at {filename}")
            print(f"[+{elapsed():.2f}s] ❌ Failed: No WAV found at expected path: {filename}")
            return False

        print(f"[+{elapsed():.2f}s] ✅ Download complete! ({download_duration:.2f}s, {file_size_mb:.2f}MB)")

        print(f"[+{elapsed():.2f}s] 🎧 Extracting audio features...")
        feature_start = time.perf_counter()
        features = extract_features(filename)
        feature_duration = time.perf_counter() - feature_start
        print(f"[+{elapsed():.2f}s] ✅ Feature extraction complete! ({feature_duration:.2f}s)")

        if "error" in features:
            print(f"[+{elapsed():.2f}s] ❌ Feature extraction failed: {features['error']}")
            log_error(song_id, artist_id, song_name, artist_name, features["error"])
            return False

        # Flatten features for SQL/ML use
        print(f"[+{elapsed():.2f}s] 🔄 Flattening feature vector...")
        flat_vector = flatten_audio_features(features)
        vector_length = len(flat_vector) if isinstance(flat_vector, list) else "unknown"
        print(f"[+{elapsed():.2f}s] ✅ Feature vector flattened (length: {vector_length})")

        # Insert into songs table
        print(f"[+{elapsed():.2f}s] 💾 Saving to database...")
        db_start = time.perf_counter()
        await client.execute(
            "INSERT INTO songs (song_id, artist_id, feature_vector) VALUES (?, ?, ?)",
            [song_id, artist_id, str(flat_vector)]
        )
        db_duration = time.perf_counter() - db_start
        
        total_duration = time.perf_counter() - start_time
        print(f"[+{elapsed():.2f}s] ✅ Database save complete! ({db_duration:.2f}s)")
        print(f"[+{elapsed():.2f}s] 🎉 AUDIO ITEM PROCESSED SUCCESSFULLY!")
        print(f"⏱️  Total processing time: {total_duration:.2f}s")
        print(f"📊 Breakdown: Download: {download_duration:.2f}s | Features: {feature_duration:.2f}s | DB: {db_duration:.2f}s")

        return True

    except Exception as e:
        elapsed_time = elapsed()
        print(f"[+{elapsed_time:.2f}s] ❌ PROCESSING FAILED: {song_name} - {artist_name}")
        print(f"[+{elapsed_time:.2f}s] 💥 Error: {e}")
        log_error(song_id, artist_id, song_name, artist_name, str(e))
        return False

    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"[+{elapsed():.2f}s] 🗑️  Temporary file cleaned up: {filename}")
            except Exception as rm_err:
                print(f"[+{elapsed():.2f}s] ⚠️  Error deleting temp file {filename}: {rm_err}")

async def main():
    """Main function to process audio items one at a time from songs_from_playlists."""
    print("🚀 STARTING AUDIO PROCESSING PIPELINE")
    print("="*80)
    
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    if not url or not auth_token:
        print("❌ Missing database credentials! Check your .env file.")
        return

    print(f"🔗 Connecting to database...")
    
    async with create_client(url, auth_token=auth_token) as client:
        print("✅ Database connection established!")
        
        # Get all items from songs_from_playlists table
        print("📋 Fetching items from source table...")
        playlist_items = await client.execute("SELECT song_id, artist_id, song_name, artist_name FROM songs_from_playlists")
        
        total_items_in_source = len(playlist_items)
        print(f"📊 Found {total_items_in_source} audio items in source table")

        # Get all existing song_ids in one query (MUCH FASTER!)
        print("🔍 Fetching all existing processed item IDs...")
        existing_ids_result = await client.execute("SELECT song_id FROM songs")
        existing_ids = {row[0] for row in existing_ids_result}  # Convert to set for fast lookup
        print(f"✅ Found {len(existing_ids)} already processed items")

        # Check which items need processing
        print("\n🔄 Checking all items and showing already processed ones...")
        print("="*80)
        items_to_process = []
        items_already_processed = 0
        
        for entry in playlist_items:
            song_id, artist_id, song_name, artist_name = entry
            if song_id in existing_ids:
                items_already_processed += 1
                # SHOW ALL ALREADY PROCESSED ITEMS
                print(f"⏭️  [{items_already_processed:4d}] SKIP: {song_name} - {artist_name} (ID: {song_id})")
            else:
                items_to_process.append(entry)

        items_to_download = len(items_to_process)
        
        print("\n" + "="*80)
        print("📈 PROCESSING SUMMARY:")
        print(f"📀 Total items in source: {total_items_in_source}")
        print(f"✅ Already processed: {items_already_processed}")
        print(f"📥 To download & process: {items_to_download}")
        print(f"📊 Progress: {items_already_processed}/{total_items_in_source} ({(items_already_processed/total_items_in_source)*100:.1f}% complete)")
        print("="*80)

        if items_to_download == 0:
            print("🎉 All audio items already processed! Nothing to do.")
            return

        # Show items that will be processed
        print(f"\n📋 ITEMS TO BE PROCESSED ({items_to_download} total):")
        print("-"*80)
        for i, (song_id, artist_id, song_name, artist_name) in enumerate(items_to_process, 1):
            print(f"📥 [{i:4d}] TODO: {song_name} - {artist_name} (ID: {song_id})")
        print("-"*80)

        # Process remaining items
        start_all = time.perf_counter()
        successful_processes = 0
        failed_processes = 0
        
        for i, entry in enumerate(items_to_process, 1):
            song_id, artist_id, song_name, artist_name = entry
            
            print(f"\n⏰ Starting item {i}/{items_to_download} at {time.strftime('%H:%M:%S')}")
            
            success = await process_and_save_item(
                client, song_id, artist_id, song_name, artist_name, 
                i, items_to_download, global_start=start_all
            )
            
            if success:
                successful_processes += 1
            else:
                failed_processes += 1
            
            # Progress update
            completed = successful_processes + failed_processes
            overall_completed = items_already_processed + completed
            overall_progress = (overall_completed / total_items_in_source) * 100
            
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"✅ Successful processes this session: {successful_processes}")
            print(f"❌ Failed processes this session: {failed_processes}")
            print(f"📈 Overall progress: {overall_completed}/{total_items_in_source} ({overall_progress:.1f}%)")
            
            if i < items_to_download:
                remaining = items_to_download - i
                avg_time_per_item = (time.perf_counter() - start_all) / i
                estimated_remaining_time = remaining * avg_time_per_item
                print(f"⏱️  Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")

        total_duration = time.perf_counter() - start_all
        
        print("\n" + "="*80)
        print("🎉 PROCESSING COMPLETE!")
        print("="*80)
        print(f"⏱️  Total session time: {total_duration/60:.1f} minutes ({total_duration:.2f} seconds)")
        print(f"✅ Successfully processed: {successful_processes} items")
        print(f"❌ Failed to process: {failed_processes} items")
        print(f"📊 Success rate: {(successful_processes/(successful_processes+failed_processes)*100):.1f}%" if (successful_processes+failed_processes) > 0 else "N/A")
        print(f"🎵 Average time per item: {total_duration/items_to_download:.2f} seconds" if items_to_download > 0 else "N/A")
        print(f"📈 Final overall progress: {items_already_processed + successful_processes}/{total_items_in_source} ({((items_already_processed + successful_processes)/total_items_in_source)*100:.1f}%)")
        
        if failed_processes > 0:
            print(f"⚠️  Check error_log.json for details on {failed_processes} failed processes")
        
        print("="*80)


if __name__ == "__main__":
    asyncio.run(main())