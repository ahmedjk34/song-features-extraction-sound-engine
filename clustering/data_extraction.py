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
        return None
    
    log_info("Connecting to database...")
    
    async with create_client(url, auth_token=auth_token) as client, aiohttp.ClientSession() as session:
        log_success("Database connection established.")
        
        # Find artists with any new field missing
        songs_query = """
        SELECT * from songs
        """
        
        log_info("Fetching songs from database...")
        result = await client.execute(songs_query)
        
        if not result.rows:
            log_warn("No songs found in the database.")
            return []
        
        # Convert rows to dictionaries using column names
        songs = []
        for row in result.rows:
            # Create dictionary by zipping column names with row values
            song_dict = {col: row[i] for i, col in enumerate(result.columns)}
            songs.append(song_dict)
        
        log_success(f"Fetched {len(songs)} songs from the database.")
        return songs

def verify_features(songs):
    original_count = len(songs)
    log_info(f"Verifying feature vectors for {original_count} songs...")
    
    verified_songs = []
    for song in songs:
        feature_vector = song['feature_vector']
        
        # Convert string to numpy array if needed
        if isinstance(feature_vector, str):
            # Convert string representation to numpy array
            feature_vector = np.fromstring(feature_vector.strip("[]"), sep=",")
        
        if feature_vector is None or len(feature_vector) != 223:
            log_warn(f"Song ID {song['song_id']} has invalid feature vector.")
        else:
            # Update the song dictionary with the processed feature vector
            song_copy = song.copy()  # Make a copy to avoid modifying original
            song_copy['feature_vector'] = feature_vector
            verified_songs.append(song_copy)
    
    verified_count = len(verified_songs)
    log_success(f"Verification complete. {verified_count}/{original_count} songs have valid feature vectors.")
    return verified_songs

def data_quality_report(songs):
    if not songs:
        log_warn("No songs to analyze for data quality.")
        return
    
    feature_matrix = np.array([song['feature_vector'] for song in songs])
    report = {}
    
    for i in range(feature_matrix.shape[1]):
        col = feature_matrix[:, i]
        report[f"feature_{i}"] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "missing": int(np.sum(np.isnan(col))),
        }
    
    # Print summary for each feature (or group as needed)
    for feat, stats in report.items():
        print(f"{feat}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}, missing={stats['missing']}")

async def main():
    songs = await get_songs_from_db()
    if songs is None or len(songs) == 0:
        return
    
    verified_songs = verify_features(songs)
    data_quality_report(verified_songs)

if __name__ == "__main__":
    asyncio.run(main())