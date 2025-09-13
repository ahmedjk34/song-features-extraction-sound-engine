import os

from dotenv import load_dotenv
from libsql_client import create_client

import asyncio
import aiohttp

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


async def getSongsFromDB():
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
            SELECT * from songs
        """
        log_info("Fetching songs from database...")
        songs = await client.execute(songs_query)

        if not songs:
            log_warn("No songs found in the database.")
            return []
        
        log_success(f"Fetched {len(songs)} songs from the database.")

        return songs



async def main():
    songs = await getSongsFromDB()


if __name__ == "__main__":
    asyncio.run(main())
