import yt_dlp
import os

#We used ffmpeg for conversion to .WAV, if we want to upload it to let's say railway, we need to bundle it with docker and change the explicit path

def download_song(song_name, artist_name, output_filename="audio.wav"):
    query = f"ytsearch1:{song_name} {artist_name} lyrics"

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
        'ffmpeg_location': 'C:/ffmpeg/bin/',  # Most common location
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])

    if os.path.exists("temp_audio.wav"):
        os.rename("temp_audio.wav", output_filename)
        print(f"✅ Saved: {output_filename}")
    else:
        print("❌ Failed: No WAV found")

# Example:
download_song("CRY", "cigs after sex", "audio.wav")
