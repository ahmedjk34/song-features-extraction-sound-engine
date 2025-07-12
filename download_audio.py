import subprocess
import os
import tempfile
import shutil

def download_audio(song_name: str, artist_name: str, output_path: str):
    """Download audio from YouTube using yt-dlp"""
    query = f"ytsearch1:{song_name} {artist_name}"
    
    # Clean up any existing file first
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Use a temporary directory to avoid conflicts
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = os.path.join(temp_dir, "audio.%(ext)s")
        
        # Simplified command that avoids the exit code 101 issue
        command = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "0",  # Best quality
            "--no-warnings",
            "--ignore-errors",
            "--no-playlist",
            "--no-check-certificates",
            "--prefer-ffmpeg",
            "--ffmpeg-location", "/usr/bin/ffmpeg",
            "-o", temp_output,
            query
        ]
        
        try:
            print(f"Searching for: {query}")
            result = subprocess.run(
                command, 
                check=True,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            # Find the downloaded file in temp directory
            downloaded_files = []
            for file in os.listdir(temp_dir):
                if file.startswith("audio."):
                    downloaded_files.append(os.path.join(temp_dir, file))
            
            if not downloaded_files:
                raise FileNotFoundError("No audio file was downloaded")
            
            # Use the first (and should be only) downloaded file
            downloaded_file = downloaded_files[0]
            
            # Move the file to the final output path
            shutil.move(downloaded_file, output_path)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Download successful: {os.path.getsize(output_path)} bytes")
                return output_path
            else:
                raise FileNotFoundError(f"Final output file not found at {output_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"yt-dlp error (exit code {e.returncode}):")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            
            # Try alternative approach with direct URL download
            try:
                print("Trying alternative download method...")
                
                # Step 1: Get the direct URL first
                url_command = [
                    "yt-dlp",
                    "--get-url",
                    "-f", "bestaudio/best",
                    "--no-warnings",
                    query
                ]
                
                url_result = subprocess.run(
                    url_command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                direct_url = url_result.stdout.strip()
                if not direct_url:
                    raise Exception("Could not get direct URL")
                
                print(f"Got direct URL, downloading...")
                
                # Step 2: Download directly with minimal options
                direct_command = [
                    "yt-dlp",
                    "-x",
                    "--audio-format", "mp3",
                    "--no-warnings",
                    "--no-playlist",
                    "-o", temp_output,
                    direct_url
                ]
                
                subprocess.run(direct_command, check=True, capture_output=True, text=True, timeout=120)
                
                # Find and move the downloaded file
                downloaded_files = []
                for file in os.listdir(temp_dir):
                    if file.startswith("audio."):
                        downloaded_files.append(os.path.join(temp_dir, file))
                
                if not downloaded_files:
                    raise FileNotFoundError("Alternative download failed to create file")
                
                downloaded_file = downloaded_files[0]
                shutil.move(downloaded_file, output_path)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"Alternative download successful: {os.path.getsize(output_path)} bytes")
                    return output_path
                else:
                    raise FileNotFoundError("Alternative download failed")
                    
            except subprocess.CalledProcessError as e2:
                print(f"Alternative method failed: {e2}")
                if e2.stderr:
                    print(f"Alternative stderr: {e2.stderr}")
                # Try one more approach with just basic download
                try:
                    print("Trying basic download method...")
                    basic_command = [
                        "yt-dlp",
                        "--extract-audio",
                        "--audio-format", "mp3",
                        "--output", temp_output,
                        query
                    ]
                    
                    subprocess.run(basic_command, check=True, capture_output=True, text=True, timeout=120)
                    
                    # Find and move the downloaded file
                    downloaded_files = []
                    for file in os.listdir(temp_dir):
                        if file.endswith(('.mp3', '.m4a', '.webm', '.opus')):
                            downloaded_files.append(os.path.join(temp_dir, file))
                    
                    if not downloaded_files:
                        raise FileNotFoundError("Basic download failed")
                    
                    downloaded_file = downloaded_files[0]
                    
                    # Convert to MP3 if needed
                    if not downloaded_file.endswith('.mp3'):
                        print("Converting to MP3...")
                        convert_command = [
                            "ffmpeg",
                            "-i", downloaded_file,
                            "-acodec", "mp3",
                            "-ab", "192k",
                            "-y",
                            output_path
                        ]
                        subprocess.run(convert_command, check=True, capture_output=True, text=True)
                    else:
                        shutil.move(downloaded_file, output_path)
                    
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"Basic download successful: {os.path.getsize(output_path)} bytes")
                        return output_path
                    else:
                        raise FileNotFoundError("Basic download failed to create output")
                        
                except Exception as e3:
                    print(f"All download methods failed. Last error: {e3}")
                    raise Exception(f"All download methods failed. Last error: {e3}")
                    
            except Exception as e3:
                print(f"Alternative method error: {e3}")
                raise Exception(f"Alternative download failed: {e3}")
                
        except subprocess.TimeoutExpired:
            print("Download timed out")
            raise Exception("Download timed out after 2 minutes")
        except FileNotFoundError:
            print("yt-dlp not found. Make sure it's installed.")
            raise
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            raise