import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import json
import asyncio
import re
import time
from libsql_client import create_client

from dotenv import load_dotenv
load_dotenv()  # <-- Load .env at the very top!

# Replace with your actual client ID/secret and redirect URI
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI_DEV"),
    scope="playlist-read-private playlist-read-collaborative"
))

async def get_latest_data_from_turso():
    """Query the last record from data_from_site table"""
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    async with create_client(url, auth_token=auth_token) as client:
        result = await client.execute(
            "SELECT data FROM data_from_site WHERE id = 1"
        )
        
        if result.rows:
            data_json = result.rows[0][0]
            print("✅ Data retrieved from data_from_site table.")
            return json.loads(data_json) if isinstance(data_json, str) else data_json
        else:
            print("❌ No data found in data_from_site table.")
            return None

async def save_songs_batch_to_database(songs_batch):
    """Save multiple songs to the database in a batch"""
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    async with create_client(url, auth_token=auth_token) as client:
        try:
            saved_count = 0
            for song in songs_batch:
                await client.execute(
                    """INSERT OR IGNORE INTO songs_from_playlists 
                       (song_id, artist_id, song_name, artist_name) 
                       VALUES (?, ?, ?, ?)""",
                    [song['song_id'], song['artist_id'], song['song_name'], song['artist_name']]
                )
                saved_count += 1
            
            return saved_count
        except Exception as e:
            print(f"❌ Error saving batch of songs: {e}")
            return 0

def extract_playlist_id_from_url(url):
    """Extract playlist ID from Spotify API URL"""
    pattern = r'https://api\.spotify\.com/v1/playlists/([a-zA-Z0-9]+)/tracks'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_all_playlist_tracks_safe(sp, playlist_id, sleep_between=0.2):
    """Get all tracks from a playlist with slight rate limiting"""
    tracks = []
    offset = 0
    total_downloaded = 0

    try:
        while True:
            response = sp.playlist_items(playlist_id, offset=offset, limit=50)
            items = response['items']
            
            if not items:
                break
                
            tracks.extend(items)
            offset += len(items)
            total_downloaded += len(items)
            
            print(f"     📥 Playlist {playlist_id}: Downloaded {total_downloaded} tracks so far...")
            
            # If playlist is very large, consider limiting
            if len(tracks) > 1000:  # Limit very large playlists
                print(f"     ⚠️  Large playlist detected, limiting to first 1000 tracks")
                break

            time.sleep(sleep_between)  # Add slight rate limiting
                
    except Exception as e:
        print(f"❌ Error fetching tracks for playlist {playlist_id}: {e}")
        return []
    
    return tracks

async def process_playlist_tracks(playlist_urls, max_playlists=None, sleep_between_playlists=0.5):
    """Process playlist URLs with optional limit and slight rate limiting between playlists"""
    all_songs_data = []
    total_saved = 0
    
    # Limit number of playlists if specified
    if max_playlists:
        playlist_urls = playlist_urls[:max_playlists]
        print(f"🚦 Processing limited to first {max_playlists} playlists")
    
    print(f"📊 Processing {len(playlist_urls)} playlists total...")
    
    for i, url in enumerate(playlist_urls, 1):
        playlist_id = extract_playlist_id_from_url(url)
        
        if not playlist_id:
            print(f"❌ Could not extract playlist ID from URL: {url}")
            continue
            
        print(f"\n🎵 [{i}/{len(playlist_urls)}] Processing playlist: {playlist_id}")
        
        try:
            playlist_info = sp.playlist(
                playlist_id, 
                fields="name,description,owner.display_name,tracks.total"
            )
            
            playlist_name = playlist_info.get('name', 'Unknown Playlist')
            playlist_owner = playlist_info.get('owner', {}).get('display_name', 'Unknown Owner')
            total_tracks = playlist_info.get('tracks', {}).get('total', 0)
            
            print(f"   📝 Playlist: '{playlist_name}' by {playlist_owner} ({total_tracks} tracks)")

            # Get all tracks from this playlist
            tracks = get_all_playlist_tracks_safe(sp, playlist_id)
            
            if tracks:
                print(f"   ✅ Retrieved {len(tracks)} tracks")
                playlist_songs = []
                
                # Process each track
                for item in tracks:
                    if item['track'] and item['track']['name'] and item['track']['id']:
                        track = item['track']
                        
                        song_data = {
                            'song_id': track['id'],
                            'artist_id': track['artists'][0]['id'] if track['artists'] else 'unknown',
                            'song_name': track['name'],
                            'artist_name': track['artists'][0]['name'] if track['artists'] else 'Unknown Artist'
                        }
                        
                        playlist_songs.append(song_data)
                        all_songs_data.append(song_data)
                
                # Save to database
                if playlist_songs:
                    saved_count = await save_songs_batch_to_database(playlist_songs)
                    total_saved += saved_count
                    print(f"   💾 Saved {saved_count} songs to database")
                    
            else:
                print(f"   ❌ No tracks found in playlist {playlist_id}")
                
        except Exception as e:
            print(f"   ❌ Error processing playlist {playlist_id}: {e}")
            continue
        
        # Progress update
        print(f"🕐 Progress: Finished playlist {i}/{len(playlist_urls)}")
        print(f"---------------------------------------------")
        time.sleep(sleep_between_playlists)  # Add slight rate limiting between playlist downloads
    
    return all_songs_data, total_saved

async def create_table_if_not_exists():
    """Create the songs_from_playlists table if it doesn't exist"""
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    async with create_client(url, auth_token=auth_token) as client:
        try:
            await client.execute("""
                CREATE TABLE IF NOT EXISTS songs_from_playlists (
                  song_id TEXT NOT NULL PRIMARY KEY,
                  artist_id TEXT NOT NULL,
                  song_name TEXT NOT NULL,
                  artist_name TEXT NOT NULL
                )
            """)
            print("✅ Table songs_from_playlists ready")
        except Exception as e:
            print(f"❌ Error creating table: {e}")

async def main():
    """Main function with slight rate limiting and enhanced logging"""
    await create_table_if_not_exists()
    
    latest_data = await get_latest_data_from_turso()
    
    if latest_data and isinstance(latest_data, list):
        print(f"\n🎯 Found {len(latest_data)} playlist URLs in database")
        
        # Uncomment this line to limit processing for testing:
        # max_playlists = 20  # Process only first 20 playlists for testing
        max_playlists = None  # Process all playlists
        
        all_songs, total_saved = await process_playlist_tracks(latest_data, max_playlists)
        
        print(f"\n📊 Final Summary:")
        print(f"   Total unique songs found: {len(all_songs)}")
        print(f"   Total songs saved to database: {total_saved}")
        
    else:
        print("❌ No valid playlist URLs found in database.")

if __name__ == "__main__":
    asyncio.run(main())