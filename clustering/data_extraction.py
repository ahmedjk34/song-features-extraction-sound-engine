import os

from dotenv import load_dotenv
from libsql_client import create_client

import asyncio
import aiohttp

import numpy as np


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

load_dotenv() 


async def get_songs_from_db():
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    if not url or not auth_token:
        log_fail("Missing database credentials! Check your .env file.")
        return

    log_info("Connecting to database...")

    async with create_client(url, auth_token=auth_token) as client, aiohttp.ClientSession() as session:
        log_success("Database connection established.")

        # Find artists with any new field missing
        songs_query = """
            SELECT * from songs limit 2
        """
        log_info("Fetching songs from database...")
        songs = await client.execute(songs_query)

        if not songs:
            log_warn("No songs found in the database.")
            return []
        
        log_success(f"Fetched {len(songs)} songs from the database.")

    return songs.rows

def verify_features(songs):
    original_count = len(songs)
    log_info(f"Verifying feature vectors for {original_count} songs...")

    verified_songs = []
    for song in songs:
        feature_vector = song['feature_vector']

        #convert string to numpy array if needed
        if isinstance(feature_vector, str):
            # Convert string representation to numpy array
            feature_vector = np.fromstring(feature_vector.strip("[]"), sep=",")

        
        if feature_vector is None or len(feature_vector) != 223:
            log_warn(f"Song ID {song['song_id']} has invalid feature vector.")
        else:
            verified_songs.append(song)

    verified_count = len(verified_songs)
    log_success(f"Verification complete. {verified_count}/{original_count} songs have valid feature vectors.")
    return verified_songs

async def main():
    songs = await get_songs_from_db()
    if songs is None:
        return
    verified_songs = verify_features(songs)




if __name__ == "__main__":
    asyncio.run(main())
