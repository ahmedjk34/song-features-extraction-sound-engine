import os
from dotenv import load_dotenv
from libsql_client import create_client
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Optional
import time

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

async def get_songs_from_db(max_retries: int = 3, batch_size: Optional[int] = None) -> Optional[List[Dict]]:
    """
    Get songs from database with retry logic and optional batching
    
    Args:
        max_retries: Maximum number of retry attempts
        batch_size: If specified, fetch songs in batches (useful for large datasets)
    """
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    if not url or not auth_token:
        log_fail("Missing database credentials! Check your .env file.")
        return None

    for attempt in range(max_retries):
        try:
            log_info(f"Connecting to database... (attempt {attempt + 1}/{max_retries})")
            
            # Configure timeout and connection settings
            timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=60)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            ) as session:
                client = create_client(url, auth_token=auth_token)
                
                log_success("Database connection established.")
                
                if batch_size:
                    return await _fetch_songs_in_batches(client, batch_size)
                else:
                    return await _fetch_all_songs(client)
                    
        except (aiohttp.ClientPayloadError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            log_warn(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                log_info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                log_fail("All retry attempts exhausted.")
                return None
        except Exception as e:
            log_fail(f"Unexpected error: {str(e)}")
            return None

async def _fetch_all_songs(client) -> List[Dict]:
    """Fetch all songs at once"""
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

async def _fetch_songs_in_batches(client, batch_size: int) -> List[Dict]:
    """Fetch songs in batches to reduce memory usage and network load"""
    log_info("Getting total song count...")
    count_result = await client.execute("SELECT COUNT(*) as total FROM songs")
    total_songs = count_result.rows[0][0] if count_result.rows else 0
    
    if total_songs == 0:
        log_warn("No songs found in the database.")
        return []
    
    log_info(f"Found {total_songs} songs. Fetching in batches of {batch_size}...")
    
    all_songs = []
    offset = 0
    
    while offset < total_songs:
        batch_query = f"SELECT * FROM songs LIMIT {batch_size} OFFSET {offset}"
        log_info(f"Fetching batch: {offset + 1}-{min(offset + batch_size, total_songs)} of {total_songs}")
        
        try:
            result = await client.execute(batch_query)
            if result.rows:
                batch_songs = [{col: row[i] for i, col in enumerate(result.columns)} for row in result.rows]
                all_songs.extend(batch_songs)
                log_success(f"Fetched batch of {len(batch_songs)} songs")
            
            offset += batch_size
            
            # Small delay between batches to be nice to the server
            await asyncio.sleep(0.1)
            
        except Exception as e:
            log_warn(f"Error fetching batch at offset {offset}: {str(e)}")
            # Continue with next batch instead of failing completely
            offset += batch_size
            continue
    
    log_success(f"Fetched total of {len(all_songs)} songs from the database.")
    return all_songs

def verify_features(songs: List[Dict]) -> List[Dict]:
    """Verify and clean feature vectors"""
    original_count = len(songs)
    log_info(f"Verifying feature vectors for {original_count} songs...")
    
    verified_songs = []
    
    for song in songs:
        try:
            feature_vector = song['feature_vector']
            
            # Convert string to numpy array if needed
            if isinstance(feature_vector, str):
                # Handle different string formats
                feature_str = feature_vector.strip()
                if feature_str.startswith('[') and feature_str.endswith(']'):
                    feature_str = feature_str[1:-1]  # Remove brackets
                
                # Try different separators
                try:
                    feature_vector = np.fromstring(feature_str, sep=",")
                except ValueError:
                    try:
                        feature_vector = np.fromstring(feature_str, sep=" ")
                    except ValueError:
                        log_warn(f"Song ID {song.get('song_id', 'unknown')} has unparseable feature vector format.")
                        continue
            
            # Convert to numpy array if it isn't already
            if not isinstance(feature_vector, np.ndarray):
                feature_vector = np.array(feature_vector)
            
            # Check for valid feature vector
            if feature_vector is None or len(feature_vector) != 223:
                log_warn(f"Song ID {song.get('song_id', 'unknown')} has invalid feature vector length: {len(feature_vector) if feature_vector is not None else 'None'}")
                continue
            
            # Check for NaN or infinite values
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                log_warn(f"Song ID {song.get('song_id', 'unknown')} has NaN or infinite values in feature vector.")
                continue
            
            # Create a copy and update the feature vector
            song_copy = song.copy()
            song_copy['feature_vector'] = feature_vector
            verified_songs.append(song_copy)
            
        except Exception as e:
            log_warn(f"Error processing song ID {song.get('song_id', 'unknown')}: {str(e)}")
            continue
    
    verified_count = len(verified_songs)
    log_success(f"Verification complete. {verified_count}/{original_count} songs have valid feature vectors.")
    
    if verified_count == 0:
        log_fail("No songs with valid feature vectors found!")
    
    return verified_songs

def data_quality_report(songs: List[Dict]):
    """Generate a data quality report for the feature vectors"""
    if not songs:
        log_warn("No songs to analyze for data quality.")
        return
    
    log_info("Generating data quality report...")
    
    try:
        feature_matrix = np.array([song['feature_vector'] for song in songs])
        
        log_info(f"Feature matrix shape: {feature_matrix.shape}")
        log_info(f"Total songs: {len(songs)}")
        
        # Overall statistics
        overall_stats = {
            "total_features": feature_matrix.shape[1],
            "total_songs": feature_matrix.shape[0],
            "matrix_mean": float(np.mean(feature_matrix)),
            "matrix_std": float(np.std(feature_matrix)),
            "matrix_min": float(np.min(feature_matrix)),
            "matrix_max": float(np.max(feature_matrix)),
            "total_missing": int(np.sum(np.isnan(feature_matrix)))
        }
        
        log_success("Overall Statistics:")
        for key, value in overall_stats.items():
            print(f"  {key}: {value}")
        
        # Check for features with unusual distributions
        problematic_features = []
        for i in range(feature_matrix.shape[1]):
            col = feature_matrix[:, i]
            std_val = np.std(col)
            
            # Flag features with very low variance (might be constant)
            if std_val < 1e-10:
                problematic_features.append(f"feature_{i} (low variance: {std_val:.2e})")
        
        if problematic_features:
            log_warn("Features with potential issues:")
            for feat in problematic_features[:10]:  # Show first 10
                print(f"  {feat}")
            if len(problematic_features) > 10:
                print(f"  ... and {len(problematic_features) - 10} more")
        
        log_success("Data quality report completed.")
        
    except Exception as e:
        log_fail(f"Error generating data quality report: {str(e)}")

# Test function to verify database connectivity
async def test_connection():
    """Test database connectivity"""
    log_info("Testing database connection...")
    try:
        songs = await get_songs_from_db(max_retries=1, batch_size=10)  # Small batch for testing
        if songs:
            log_success(f"Connection test successful. Found {len(songs)} songs (limited to 10 for test).")
            return True
        else:
            log_fail("Connection test failed - no songs retrieved.")
            return False
    except Exception as e:
        log_fail(f"Connection test failed: {str(e)}")
        return False

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
    "test_connection",
]