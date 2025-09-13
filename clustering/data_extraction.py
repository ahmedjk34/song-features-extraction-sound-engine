# utils.py
import os
from dotenv import load_dotenv
from libsql_client import create_client
import asyncio
import aiohttp
import numpy as np

# Load environment variables immediately on import
load_dotenv()

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


def log_info(msg: str):
    print(f"{Log.OKBLUE}[INFO]{Log.ENDC} {msg}")


def log_success(msg: str):
    print(f"{Log.OKGREEN}[SUCCESS]{Log.ENDC} {msg}")


def log_warn(msg: str):
    print(f"{Log.WARNING}[WARN]{Log.ENDC} {msg}")


def log_fail(msg: str):
    print(f"{Log.FAIL}[FAIL]{Log.ENDC} {msg}")


async def get_songs_from_db():
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    if not url or not auth_token:
        log_fail("Missing database credentials! Check your .env file.")
        return None
    
    log_info("Connecting to database...")
    
    async with create_client(url, auth_token=auth_token) as client, aiohttp.ClientSession() as session:
        log_success("Database connection established.")
        
        songs_query = "SELECT * from songs"
        log_info("Fetching songs from database...")
        result = await client.execute(songs_query)
        
        if not result.rows:
            log_warn("No songs found in the database.")
            return []
        
        # Convert rows to dictionaries using column names
        songs = [{col: row[i] for i, col in enumerate(result.columns)} for row in result.rows]
        
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
            feature_vector = np.fromstring(feature_vector.strip("[]"), sep=",")
        
        if feature_vector is None or len(feature_vector) != 223:
            log_warn(f"Song ID {song['song_id']} has invalid feature vector.")
        else:
            song_copy = song.copy()
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
    
    # Print summary for each feature
    for feat, stats in report.items():
        print(f"{feat}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
              f"min={stats['min']:.3f}, max={stats['max']:.3f}, missing={stats['missing']}")


# Define what gets exported if someone does `from utils import *`
__all__ = [
    "Log",
    "log_info",
    "log_success",
    "log_warn",
    "log_fail",
    "get_songs_from_db",
    "verify_features",
    "data_quality_report",
]
