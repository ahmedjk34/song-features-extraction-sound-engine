import subprocess
import os

def download_audio(song_name: str, artist_name: str, output_path: str):
    """Download audio from YouTube using yt-dlp"""
    query = f"ytsearch1:{song_name} {artist_name} lyrics"
    command = [
        "yt-dlp",
        "-x", 
        "--audio-format", "mp3", 
        "--quiet",
        "-o", output_path,
        query
    ]
    
    try:
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio: {e}")
        raise
    except FileNotFoundError:
        print("yt-dlp not found. Make sure it's installed.")
        raise