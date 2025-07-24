import os
import asyncio
import aiohttp
from libsql_client import create_client
from dotenv import load_dotenv

load_dotenv()  # Load .env at the very top!

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

API_DELAY_SECONDS = 0.3  # 300ms between requests

async def get_spotify_token(session):
    """Get Spotify access token using Client Credentials Flow."""
    url = "https://accounts.spotify.com/api/token"
    data = {"grant_type": "client_credentials"}
    auth = aiohttp.BasicAuth(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    async with session.post(url, data=data, auth=auth) as resp:
        if resp.status == 200:
            token_data = await resp.json()
            return token_data["access_token"]
        else:
            print("❌ Failed to get Spotify token:", resp.status)
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
            return song_name, artist_name
        elif resp.status == 401:
            # Token expired
            return "TOKEN_EXPIRED", "TOKEN_EXPIRED"
        else:
            print(f"Spotify API error for {song_id}: {resp.status}")
            return None, None

async def fill_missing_fields():
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    if not url or not auth_token:
        print("❌ Missing database credentials! Check your .env file.")
        return
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("❌ Missing Spotify credentials! Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env.")
        return

    print(f"🔗 Connecting to database...")

    async with create_client(url, auth_token=auth_token) as client, aiohttp.ClientSession() as session:
        print("✅ Database connection established!")
        # Get initial Spotify token
        token = await get_spotify_token(session)
        if not token:
            print("❌ Could not obtain Spotify access token.")
            return

        # Find missing fields in 'songs'
        songs_missing_query = """
            SELECT song_id FROM songs
            WHERE song_name IS NULL OR artist_name IS NULL
        """
        songs_missing = await client.execute(songs_missing_query)

        to_update = []
        for row in songs_missing.rows:
            to_update.append(('songs', row['song_id']))

        print(f"Found {len(to_update)} songs needing metadata updates.")

        for table, song_id in to_update:
            song_name, artist_name = await get_spotify_metadata(song_id, session, token)
            if song_name == "TOKEN_EXPIRED":
                token = await get_spotify_token(session)
                if not token:
                    print("❌ Could not refresh Spotify token.")
                    break
                song_name, artist_name = await get_spotify_metadata(song_id, session, token)

            await asyncio.sleep(API_DELAY_SECONDS)
            if song_name and artist_name:
                update_query = f"""
                    UPDATE {table}
                    SET song_name = ?, artist_name = ?
                    WHERE song_id = ?
                """
                await client.execute(update_query, [song_name, artist_name, song_id])
                print(f"✅ Updated {song_id} in {table}")
            else:
                print(f"❌ Could not fetch info for {song_id}")

if __name__ == "__main__":
    asyncio.run(fill_missing_fields())