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

async def get_artist_metadata(artist_id, session, token):
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, headers=headers) as resp:
        if resp.status == 200:
            data = await resp.json()
            artist_name = data.get("name")
            followers = data.get("followers", {}).get("total")
            genres = json.dumps(data.get("genres", [])) if data.get("genres") else None
            images = json.dumps(data.get("images", [])) if data.get("images") else None
            popularity = data.get("popularity")

            log_info(
                f"Fetched metadata for artist_id {artist_id}: name='{artist_name}', followers='{followers}', genres={genres}, popularity={popularity}"
            )
            return {
                "artist_name": artist_name,
                "followers": followers,
                "genres": genres,
                "images": images,
                "popularity": popularity,
            }
        elif resp.status == 401:
            log_warn(f"Spotify token expired while fetching artist_id {artist_id}.")
            return "TOKEN_EXPIRED"
        else:
            log_fail(f"Spotify API error for artist_id {artist_id}: HTTP {resp.status}")
            return None

async def fill_missing_artist_fields():
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

        # Find artists with any new field missing
        artists_missing_query = """
            SELECT artist_id
            FROM artists_vector
            WHERE artist_name IS NULL
               OR followers IS NULL
               OR genres IS NULL
               OR images IS NULL
               OR popularity IS NULL
        """
        log_info("Searching for artists with missing metadata fields...")
        artists_missing = await client.execute(artists_missing_query)

        to_update = []
        for row in artists_missing.rows:
            to_update.append(('artists_vector', row['artist_id']))

        log_info(f"Found {len(to_update)} artists needing metadata updates.")

        for i, (table, artist_id) in enumerate(to_update, 1):
            log_info(f"[{i}/{len(to_update)}] Processing artist_id: {artist_id}...")
            meta = await get_artist_metadata(artist_id, session, token)
            if meta == "TOKEN_EXPIRED":
                log_warn("Refreshing Spotify token...")
                token = await get_spotify_token(session)
                if not token:
                    log_fail("Could not refresh Spotify token. Aborting.")
                    break
                meta = await get_artist_metadata(artist_id, session, token)

            await asyncio.sleep(API_DELAY_SECONDS)
            if meta and isinstance(meta, dict):
                update_query = f"""
                    UPDATE {table}
                    SET artist_name = COALESCE(artist_name, ?),
                        followers = COALESCE(followers, ?),
                        genres = COALESCE(genres, ?),
                        images = COALESCE(images, ?),
                        popularity = COALESCE(popularity, ?)
                    WHERE artist_id = ?
                """
                await client.execute(update_query, [
                    meta["artist_name"],
                    meta["followers"],
                    meta["genres"],
                    meta["images"],
                    meta["popularity"],
                    artist_id
                ])
                log_success(f"Updated {artist_id} in {table}.")
            else:
                log_warn(f"Could not fetch info for {artist_id}. Skipping.")

    log_info("Artist metadata fill operation complete.")

if __name__ == "__main__":
    asyncio.run(fill_missing_artist_fields())