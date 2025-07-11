# YouTube to Spotify-Style Audio Features Extractor

A Python pipeline that takes a **song name + artist**, searches YouTube for the first matching video, downloads the audio **temporarily**, and extracts Spotify-style audio features using [Essentia](https://essentia.upf.edu/).

---

## 🚀 Features

- Searches YouTube for the given song and artist using `yt-dlp`
- Downloads the first matching audio result as a **temporary file** (no permanent storage)
- Extracts rich Spotify-like audio features such as:
  - Danceability, Valence, Energy
  - Acousticness, Instrumentalness, Liveness
  - Tempo (BPM), Key, Mode, Loudness
  - Duration, Time Signature
- Easy to extend for recommendation systems or music analysis projects

---

## 🛠️ Technologies & Libraries

| Library/Tool                               | Purpose                           |
| ------------------------------------------ | --------------------------------- |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | YouTube audio search & download   |
| [Essentia](https://essentia.upf.edu/)      | Audio feature extraction          |
| Python 3.8+                                | Core programming language         |
| `tempfile`                                 | Ephemeral temporary file handling |

---

## ⚙️ Setup & Installation

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/yt-to-spotify-features.git
cd yt-to-spotify-features
```

````

### 2. Create and activate Conda environment (recommended)

```bash
conda create -n yt-audio python=3.8 -y
conda activate yt-audio
```

### 3. Install dependencies

```bash
pip install yt-dlp
conda install -c conda-forge essentia -y
```

> Note: Essentia installation might take a while as it includes C++ binaries.

---

## 📂 Project Structure

```
yt-to-spotify-features/
├── download_audio.py     # YouTube audio search & download module
├── extract_features.py   # Essentia audio features extraction module
├── main.py               # Orchestrates full pipeline (download + extract)
└── README.md             # This file
```

---

## 🔧 Usage

Run the main pipeline with a song and artist of your choice:

```bash
python main.py
```

By default, the example runs:

```python
analyze_song("Starboy", "The Weeknd")
```

You can modify the `main.py` file or import `analyze_song` into your own scripts.

---

## 🧩 How It Works

1. **Search & Download**
   Uses `yt-dlp` to search YouTube for the query `"song name + artist"` and download the audio of the first result as an ephemeral `.mp3` file.

2. **Feature Extraction**
   Passes the temporary audio file to Essentia's `MusicExtractor` to extract Spotify-like audio features.

3. **Cleanup**
   Temporary audio files are automatically deleted after processing to avoid disk clutter.

---

## 🎯 Spotify-Style Audio Features Extracted

- `danceability`
- `valence`
- `energy`
- `acousticness`
- `instrumentalness`
- `liveness`
- `tempo` / `bpm`
- `key`
- `mode`
- `loudness`
- `duration_sec`
- `time_signature`

---

## 🧪 Sample Output

```
Analyzing: Starboy by The Weeknd

🎧 Spotify-style Audio Features:
danceability        : 0.836
valence             : 0.561
energy              : 0.742
acousticness        : 0.003
instrumentalness    : 0.0001
liveness            : 0.128
bpm                 : 93.24
tempo               : 93.24
key                 : G#
mode                : minor
loudness            : -5.63
duration_sec        : 231.92
time_signature      : 4
```

---

## 🚀 Next Steps / Extensions

- Build a **music recommender system** using these extracted features
- Create a **web UI** (Streamlit, Flask, React+FastAPI) to allow interactive queries
- Batch process playlists or large song libraries
- Integrate with Spotify API for metadata enrichment
- Add support for **real-time streaming analysis** (advanced)

---

## ⚠️ Notes & Caveats

- Requires stable internet connection for YouTube download
- Essentia currently supports only **local audio files**; thus, temp files are used as a workaround
- The project focuses on feature extraction, not on storing or recommending songs yet

---

## 📜 License

MIT License © 2025 Your Name

---

## 💬 Contact

For questions or contributions, open an issue or contact:

**Your Name** — [your.email@example.com](mailto:your.email@example.com)
GitHub: [https://github.com/yourusername](https://github.com/yourusername)

---

Thanks for checking out the project! 🎵🚀
````
