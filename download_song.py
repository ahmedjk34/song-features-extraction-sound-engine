import yt_dlp
import os

#We used ffmpeg for conversion to .WAV, if we want to upload it to let's say railway, we need to bundle it with docker and change the explicit path

import yt_dlp
import os
import logging

# Setup basic logging
logging.basicConfig(
    filename='downloaded_songs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def download_song(song_name, artist_name, output_filename="audio.wav"):
    query = f"ytsearch1:{song_name} {artist_name} lyrics"

    ydl_opts_info = {
        'quiet': True,
        'skip_download': True,
    }

    # First, extract info to check duration
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(query, download=False)
        if 'entries' in info and info['entries']:
            video_info = info['entries'][0]
        else:
            print("❌ No results found!")
            logging.warning(f"No results found for: {song_name} by {artist_name}")
            return

        duration = video_info.get('duration', 0)
        if duration > 15 * 60:
            print(f"❌ Skipped: Video is longer than 15 minutes ({duration/60:.2f} min)")
            logging.info(f"Skipped (too long): {song_name} by {artist_name} | Duration: {duration} seconds | URL: {video_info.get('webpage_url', '')}")
            return

        video_url = video_info['webpage_url']

    # Now, download with processing
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': [
            '-ar', '44100',        # 44.1 kHz - CD quality, standard for music
            '-ac', '2',            # STEREO - crucial for music analysis
            '-sample_fmt', 's16',  # 16-bit PCM
        ],
        'prefer_ffmpeg': True,
        'quiet': False,
        'ffmpeg_location': 'C:/ffmpeg/bin/',  # Adjust as needed
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    if os.path.exists("temp_audio.wav"):
        os.rename("temp_audio.wav", output_filename)
        print(f"✅ Saved: {output_filename}")
        logging.info(f"Downloaded: {song_name} by {artist_name} | Duration: {duration} seconds | URL: {video_url} | Saved as: {output_filename}")
    else:
        print("❌ Failed: No WAV found")
        logging.error(f"Failed to save WAV for: {song_name} by {artist_name} | URL: {video_url}")
