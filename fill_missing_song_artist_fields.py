import os
import asyncio
import aiohttp
import json
from libsql_client import create_client
from dotenv import load_dotenv

# ANSI color codes for pretty CMD logging
class Log:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def log_info(msg):
    print(f"{Log.OKBLUE}[INFO]{Log.ENDC} {msg}")

def log_success(msg):
    print(f"{Log.OKGREEN}[SUCCESS]{Log.ENDC} {msg}")

def log_warn(msg):
    print(f"{Log.WARNING}[WARN]{Log.ENDC} {msg}")

def log_fail(msg):
    print(f"{Log.FAIL}[FAIL]{Log.ENDC} {msg}")

load_dotenv()  # Load .env at the very top!

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

API_DELAY_SECONDS = 0.3  # 300ms between requests

async def get_spotify_token(session):
    url = "https://accounts.spotify.com/api/token"
    data = {"grant_type": "client_credentials"}
    auth = aiohttp.BasicAuth(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    async with session.post(url, data=data, auth=auth) as resp:
        if resp.status == 200:
            token_data = await resp.json()
            log_success("Obtained Spotify access token.")
            return token_data["access_token"]
        else:
            log_fail(f"Failed to get Spotify token (HTTP {resp.status})")
            return None

async def get_spotify_metadata(song_id, session, token):
    url = f"https://api.spotify.com/v1/tracks/{song_id}"
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, headers=headers) as resp:
        if resp.status == 200:
            data = await resp.json()
            song_name = data.get("name")
            artists = data.get("artists", [])
            artist_name = artists[0]["name"] if artists else None
            sub_artists = json.dumps([a for a in artists[1:]]) if len(artists) > 1 else None
            explicit = data.get("explicit")
            popularity = data.get("popularity")
            duration_ms = data.get("duration_ms")
            album_obj = data.get("album")
            album = json.dumps(album_obj) if album_obj else None

            log_info(f"Fetched metadata for song_id {song_id}: name='{song_name}', artist='{artist_name}', sub_artists={sub_artists}")
            return {
                "song_name": song_name,
                "artist_name": artist_name,
                "sub_artists": sub_artists,
                "explicit": explicit,
                "popularity": popularity,
                "duration_ms": duration_ms,
                "album": album
            }
        elif resp.status == 401:
            log_warn(f"Spotify token expired while fetching song_id {song_id}.")
            return "TOKEN_EXPIRED"
        else:
            log_fail(f"Spotify API error for song_id {song_id}: HTTP {resp.status}")
            return None

async def fill_missing_fields():
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    if not url or not auth_token:
        log_fail("Missing database credentials! Check your .env file.")
        return
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        log_fail("Missing Spotify credentials! Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env.")
        return

    log_info("Connecting to database...")

    async with create_client(url, auth_token=auth_token) as client, aiohttp.ClientSession() as session:
        log_success("Database connection established.")
        token = await get_spotify_token(session)
        if not token:
            log_fail("Could not obtain Spotify access token.")
            return

        # Find songs with any new field missing
        songs_missing_query = """
            SELECT song_id
            FROM songs
            WHERE song_name IS NULL
               OR artist_name IS NULL
               OR explicit IS NULL
               OR popularity IS NULL
               OR duration_ms IS NULL
               OR album IS NULL
        """
        log_info("Searching for songs with missing metadata fields...")
        songs_missing = await client.execute(songs_missing_query)

        to_update = []
        for row in songs_missing.rows:
            to_update.append(('songs', row['song_id']))

        log_info(f"Found {len(to_update)} songs needing metadata updates.")

        for i, (table, song_id) in enumerate(to_update, 1):
            log_info(f"[{i}/{len(to_update)}] Processing song_id: {song_id}...")
            meta = await get_spotify_metadata(song_id, session, token)
            if meta == "TOKEN_EXPIRED":
                log_warn("Refreshing Spotify token...")
                token = await get_spotify_token(session)
                if not token:
                    log_fail("Could not refresh Spotify token. Aborting.")
                    break
                meta = await get_spotify_metadata(song_id, session, token)

            await asyncio.sleep(API_DELAY_SECONDS)
            if meta and isinstance(meta, dict):
                update_query = f"""
                    UPDATE {table}
                    SET song_name = COALESCE(song_name, ?),
                        artist_name = COALESCE(artist_name, ?),
                        sub_artists = COALESCE(sub_artists, ?),
                        explicit = COALESCE(explicit, ?),
                        popularity = COALESCE(popularity, ?),
                        duration_ms = COALESCE(duration_ms, ?),
                        album = COALESCE(album, ?)
                    WHERE song_id = ?
                """
                await client.execute(update_query, [
                    meta["song_name"],
                    meta["artist_name"],
                    meta["sub_artists"],
                    meta["explicit"],
                    meta["popularity"],
                    meta["duration_ms"],
                    meta["album"],
                    song_id
                ])
                log_success(f"Updated {song_id} in {table}.")
            else:
                log_warn(f"Could not fetch info for {song_id}. Skipping.")

    log_info("Metadata fill operation complete.")

if __name__ == "__main__":
    asyncio.run(fill_missing_fields())