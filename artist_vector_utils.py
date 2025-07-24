import os
import asyncio
import logging
from libsql_client import create_client
from dotenv import load_dotenv
from util import parse_feature_vector, average_feature_vectors, vector_to_json_str

# Load environment variables
load_dotenv()

DB_URL = os.getenv("TURSO_DATABASE_URL")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

async def fetch_artists_with_min_songs(client, min_count=5):
    query = """
        SELECT artist_id FROM artists_vector
        WHERE song_count >= ?
    """
    result = await client.execute(query, [min_count])
    artist_ids = [row['artist_id'] for row in result.rows]
    logger.info(f"Found {len(artist_ids)} artists with >= {min_count} songs.")
    return artist_ids

async def fetch_artist_song_vectors(client, artist_id):
    query = """
        SELECT feature_vector FROM songs
        WHERE artist_id = ?
          AND feature_vector IS NOT NULL
    """
    result = await client.execute(query, [artist_id])
    vectors = []
    for idx, row in enumerate(result.rows):
        vec = parse_feature_vector(row['feature_vector'], artist_id, idx)
        if vec is not None:
            vectors.append(vec)
    logger.info(f"Artist {artist_id}: {len(vectors)} valid feature vectors collected.")
    return vectors

async def update_artist_vector(client, artist_id, avg_vector):
    vec_str = vector_to_json_str(avg_vector)
    query = """
        UPDATE artists_vector
        SET feature_vector = ?
        WHERE artist_id = ?
    """
    await client.execute(query, [vec_str, artist_id])
    logger.info(f"Updated artist_id {artist_id} with averaged feature_vector.")

async def process():
    if not DB_URL or not AUTH_TOKEN:
        logger.error("Missing database credentials! Check your .env file.")
        return

    async with create_client(DB_URL, auth_token=AUTH_TOKEN) as client:
        artist_ids = await fetch_artists_with_min_songs(client, 5)
        for artist_id in artist_ids:
            try:
                vectors = await fetch_artist_song_vectors(client, artist_id)
                if len(vectors) == 0:
                    logger.warning(f"Artist {artist_id} has no valid vectors. Skipping.")
                    continue
                avg = average_feature_vectors(vectors)
                await update_artist_vector(client, artist_id, avg)
            except Exception as e:
                logger.error(f"Error processing artist {artist_id}: {e}")

if __name__ == "__main__":
    asyncio.run(process())