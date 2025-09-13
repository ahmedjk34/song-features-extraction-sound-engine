# Audio Feature Extraction & Artist Profiling System

## Project Overview

### Goal

This system implements an automated pipeline for extracting rich audio features from songs to enable music analysis, recommendation systems, and audio similarity matching. It bridges the gap between raw audio content and meaningful numerical representations that can be used in machine learning applications.

### Main Functionalities

- **Automated Song Download**: Downloads audio from YouTube using song name and artist
- **Multi-dimensional Feature Extraction**: Extracts 200+ audio features including:
  - Traditional audio features (MFCCs, spectral features, tempo)
  - Deep learning embeddings (VGGish neural network features)
  - Rhythmic and harmonic characteristics
- **Database Integration**: Stores processed features in Turso database for scalable access
- **Artist Profile Generation**: Creates averaged feature vectors for artists based on their songs
- **Batch Processing**: Handles large playlists with error recovery and progress tracking

### Problems It Solves

- **Music Recommendation**: Enables content-based filtering using audio similarity
- **Genre Classification**: Provides features for automatic music genre detection
- **Audio Analysis**: Supports research in music information retrieval
- **Playlist Generation**: Enables creation of playlists based on audio characteristics
- **Artist Similarity**: Quantifies musical similarity between artists

## Pipeline & Architecture

### Overview of the Processing Pipeline

```
YouTube URL/Search → Audio Download → Feature Extraction → Database Storage → Artist Profiling
```

### Detailed Pipeline Steps

#### 1. **Song Collection** (`get_spotify_playlist_songs.py`)

- **Purpose**: Fetches song metadata from Spotify playlists
- **Input**: Spotify playlist URLs or IDs
- **Output**: Database table `songs_from_playlists` with song names, artists, and IDs
- **Why Necessary**: Provides structured input for the processing pipeline
- **Connection**: Seeds the main processing pipeline with songs to analyze

#### 2. **Audio Download** (`download_song.py`)

- **Purpose**: Downloads audio files from YouTube based on song/artist names
- **Input**: Song name, artist name, output filename
- **Output**: WAV audio file (44.1kHz, stereo)
- **Why Necessary**: Obtains raw audio data for feature extraction
- **Key Features**:
  - Uses yt-dlp for robust YouTube downloading
  - Converts to standardized WAV format using FFmpeg
  - Filters out videos longer than 15 minutes
  - Cross-platform FFmpeg path handling
- **Connection**: Provides audio files for the feature extraction step

#### 3. **Feature Extraction** (`extract_features.py`)

- **Purpose**: Extracts comprehensive audio features from downloaded audio
- **Input**: WAV audio file path
- **Output**: JSON object with 200+ audio features
- **Why Necessary**: Converts raw audio into numerical representations for analysis
- **Processing Method**:
  - Segments audio into 10-second chunks
  - Extracts features from each segment
  - Aggregates features across segments (mean, std, min, max)
  - Handles edge cases (silent segments, short audio)

#### 4. **Database Storage** (`main.py`)

- **Purpose**: Orchestrates the entire pipeline and stores results
- **Input**: Song metadata from playlist table
- **Output**: Populated `songs` table with feature vectors
- **Why Necessary**: Provides persistent storage and batch processing coordination
- **Features**:
  - Async processing for efficiency
  - Progress tracking and error recovery
  - Duplicate detection and skip logic
  - Comprehensive logging

#### 5. **Artist Profiling** (`artist_vector_utils.py`, `average_artists_vectors.py`)

- **Purpose**: Creates representative feature vectors for artists
- **Input**: Multiple song feature vectors per artist
- **Output**: Averaged artist profile vectors
- **Why Necessary**: Enables artist-level analysis and recommendations
- **Logic**: Automatically updates artist profiles when ≥5 songs are processed

## Setup & Installation

### Prerequisites

- Python 3.8+
- FFmpeg installed and accessible
- Turso database account
- Spotify API credentials (for playlist fetching)

### Dependencies Installation

```bash
pip install -r requirements.txt
```

### Required Dependencies

```
librosa==0.10.1          # Audio processing
torch==2.1.0             # PyTorch for VGGish
torchvggish==0.1         # VGGish embeddings
yt-dlp==2023.7.6         # YouTube downloading
spotipy==2.22.1          # Spotify API
libsql-client==0.5.0     # Database client
python-dotenv==1.0.0     # Environment variables
numpy==1.24.3            # Numerical computing
```

### Environment Setup

Create a `.env` file in the project root:

```env
TURSO_DATABASE_URL=your_turso_database_url
TURSO_AUTH_TOKEN=your_turso_auth_token
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
```

### FFmpeg Installation

- **Windows**: Download from https://ffmpeg.org and place in `C:/ffmpeg/bin/`
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### Database Schema

The system expects these tables in your Turso database:

```sql
-- Songs from playlists (input)
CREATE TABLE songs_from_playlists (
    song_id TEXT PRIMARY KEY,
    artist_id TEXT,
    song_name TEXT,
    artist_name TEXT
);

-- Processed songs (output)
CREATE TABLE songs (
    song_id TEXT PRIMARY KEY,
    artist_id TEXT,
    feature_vector TEXT,  -- JSON array
    song_name TEXT,
    artist_name TEXT
);

-- Artist profiles
CREATE TABLE artists_vector (
    artist_id TEXT PRIMARY KEY,
    feature_vector TEXT,  -- JSON array
    song_count INTEGER
);
```

## Usage Instructions

### 1. Collect Songs from Spotify Playlists

```bash
python get_spotify_playlist_songs.py
```

This populates the `songs_from_playlists` table with song metadata.

### 2. Run the Main Processing Pipeline

```bash
python main.py
```

This will:

- Download audio for each unprocessed song
- Extract audio features
- Store results in the database
- Update artist profiles automatically

### 3. Fill Missing Artist Information (Optional)

```bash
python fill_missing_artists_fields.py
python fill_missing_song_artist_fields.py
```

These scripts enrich the database with additional metadata.

### 4. Manual Artist Vector Updates

```bash
python average_artists_vectors.py
```

Recalculates all artist profile vectors based on their songs.

## Module Documentation

### Core Modules

#### `main.py` - Pipeline Orchestrator

**Purpose**: Main entry point that coordinates the entire processing pipeline

**Key Functions**:

- `process_and_save_item()`: Downloads, processes, and saves a single song
- `update_artist_vector_if_needed()`: Updates artist profiles when enough songs are processed
- `log_error()`: Records processing errors to JSON log file

**Flow**:

1. Connects to database
2. Fetches unprocessed songs from `songs_from_playlists`
3. For each song: download → extract features → save to database
4. Updates artist profiles automatically
5. Provides detailed progress tracking and error reporting

#### `extract_features.py` - Audio Feature Extraction Engine

**Purpose**: Converts audio files into numerical feature representations

**Key Functions**:

- `extract_features()`: Main extraction function, returns aggregated features
- `calculate_features()`: Extracts features from a single audio segment
- `get_vggish_embedding()`: Generates deep learning embeddings using VGGish model
- `aggregate_features()`: Combines segment-level features into song-level statistics

**Feature Categories**:

1. **Temporal Features**: BPM, onset detection, zero-crossing rate
2. **Spectral Features**: Centroid, bandwidth, rolloff, chroma
3. **Timbral Features**: MFCCs (13 coefficients), RMS energy
4. **Loudness Features**: Mean, std, min, max loudness
5. **Deep Learning Features**: 128-dimensional VGGish embeddings

**Processing Strategy**:

- Segments audio into 10-second chunks
- Processes each segment independently
- Aggregates using statistical measures (mean, std, min, max)
- Handles variable-length audio gracefully

#### `download_song.py` - Audio Acquisition

**Purpose**: Downloads and preprocesses audio from YouTube

**Key Functions**:

- `download_song()`: Main download function with duration filtering

**Features**:

- YouTube search using song + artist + "lyrics" for accuracy
- Duration filtering (skips videos >15 minutes)
- Standardized output format (44.1kHz WAV)
- Cross-platform FFmpeg integration
- Comprehensive logging

#### `util.py` - Feature Processing Utilities

**Purpose**: Provides utilities for feature vector manipulation

**Key Functions**:

- `flatten_audio_features()`: Converts nested feature dict to flat vector
- `parse_feature_vector()`: Reconstructs features from flattened vector
- `average_feature_vectors()`: Computes mean of multiple feature vectors
- `vector_to_json_str()`: Serializes vectors for database storage

#### `artist_vector_utils.py` - Artist Profile Management

**Purpose**: Manages artist-level feature aggregation

**Key Functions**:

- `fetch_artist_song_vectors()`: Retrieves all feature vectors for an artist
- `update_artist_vector()`: Saves computed artist profile to database

**Logic**: Creates representative artist profiles by averaging song features

### Support Modules

#### `get_spotify_playlist_songs.py` - Playlist Processing

**Purpose**: Extracts song metadata from Spotify playlists

- Handles multiple playlist formats
- Fetches track and artist information
- Populates source database table

#### `fill_missing_artists_fields.py` & `fill_missing_song_artist_fields.py`

**Purpose**: Data enrichment modules

- Fills gaps in artist and song metadata
- Ensures database completeness for analysis

## Code Flow

### Main Processing Sequence

1. **Initialization**

   - Load environment variables
   - Connect to Turso database
   - Initialize VGGish model (CPU-based for compatibility)

2. **Batch Processing Loop**

   ```
   For each unprocessed song:
   ├── Download audio from YouTube
   ├── Extract comprehensive features
   ├── Store in database
   ├── Update artist profile (if conditions met)
   └── Clean up temporary files
   ```

3. **Feature Extraction Workflow**

   ```
   Audio File → Segment into chunks → Extract per-segment features → Aggregate statistics → Return feature vector
   ```

4. **Artist Profile Updates**
   - Triggered when artist has ≥5 processed songs
   - Computes average feature vector across all artist's songs
   - Stores in `artists_vector` table

### Error Handling Strategy

- Graceful failure handling at each pipeline stage
- Comprehensive error logging to `error_log.json`
- Automatic cleanup of temporary files
- Progress preservation (can resume after interruption)

## Extracted Features Table

| Feature Category  | Feature Name       | Dimensions | Description                                      |
| ----------------- | ------------------ | ---------- | ------------------------------------------------ |
| **Global**        | duration_sec       | 1          | Total audio duration in seconds                  |
| **Global**        | segments           | 1          | Number of 10-second segments processed           |
| **Temporal**      | global_bpm         | 1          | Overall tempo of the song                        |
| **Temporal**      | bpm_segments       | 4          | BPM statistics (mean, std, min, max) per segment |
| **Temporal**      | onset_env_mean     | 4          | Onset strength statistics                        |
| **Temporal**      | onset_count        | 4          | Number of note onsets statistics                 |
| **Temporal**      | zero_crossing_rate | 4          | Zero-crossing rate statistics                    |
| **Spectral**      | chroma_stft        | 4          | Chromagram (pitch class) statistics              |
| **Spectral**      | spectral_centroid  | 4          | Spectral centroid statistics                     |
| **Spectral**      | spectral_bandwidth | 4          | Spectral bandwidth statistics                    |
| **Spectral**      | rolloff            | 4          | Spectral rolloff statistics                      |
| **Timbral**       | rms                | 4          | RMS energy statistics                            |
| **Timbral**       | mfcc_mean          | 26         | MFCC mean (13 coeffs × 2: mean, std)             |
| **Timbral**       | mfcc_std           | 26         | MFCC std (13 coeffs × 2: mean, std)              |
| **Loudness**      | loudness           | 4          | Loudness statistics (mean, std, min, max)        |
| **Deep Learning** | vggish_emb         | 128        | VGGish neural network embeddings                 |
| **TOTAL**         |                    | **223**    | **Total feature dimensions**                     |

### Detailed Breakdown:

```
3    # Global features (duration_sec, segments, global_bpm)
+ 4    # bpm_segments (mean, std, min, max)
+ 4    # chroma_stft (mean, std, min, max)
+ 4    # spectral_centroid (mean, std, min, max)
+ 4    # spectral_bandwidth (mean, std, min, max)
+ 4    # rolloff (mean, std, min, max)
+ 4    # zero_crossing_rate (mean, std, min, max)
+ 4    # rms (mean, std, min, max)
+ 26   # mfcc_mean (13 coefficients × 2 values each: mean, std)
+ 26   # mfcc_std (13 coefficients × 2 values each: mean, std)
+ 4    # onset_env_mean (mean, std, min, max)
+ 4    # onset_count (mean, std, min, max)
+ 4    # loudness (mean, std, min, max)
+ 128  # vggish_emb (128-dimensional embedding)
─────
= 223 total features
```

### Feature Aggregation Strategy

- **Per-segment extraction**: Each 10-second segment generates one feature vector
- **Statistical aggregation**: Features aggregated using mean, std, min, max across segments
- **Robust processing**: Segments with insufficient energy (RMS < 0.01) are excluded from BPM calculation
- **Deep learning integration**: VGGish embeddings averaged across all valid segments

## Configuration & Environment Variables

### Required Environment Variables

- `TURSO_DATABASE_URL`: Connection string for Turso database
- `TURSO_AUTH_TOKEN`: Authentication token for database access
- `SPOTIPY_CLIENT_ID`: Spotify API client ID
- `SPOTIPY_CLIENT_SECRET`: Spotify API client secret

### Configurable Parameters

#### In `extract_features.py`:

- `segment_duration`: Duration of audio segments (default: 10.0 seconds)
- `min_rms`: Minimum RMS threshold for valid segments (default: 0.01)
- `N_MFCC`: Number of MFCC coefficients (default: 13)
- `VGGISH_EMBED_SIZE`: VGGish embedding dimensions (default: 128)

#### In `download_song.py`:

- Maximum video duration: 15 minutes (900 seconds)
- Audio format: 44.1kHz, stereo WAV
- Search query format: "ytsearch1:{song} {artist} lyrics"

#### In `main.py`:

- Artist profile update threshold: 5 songs minimum
- Temporary file directory: "temp_files"
- Error log file: "error_log.json"

## Additional Notes

### Audio Processing Insights

1. **Segmented Processing**: The system processes audio in 10-second segments to:

   - Handle memory limitations efficiently
   - Capture temporal variations within songs
   - Enable robust statistical aggregation
   - Handle variable-length audio files

2. **VGGish Integration**: Uses Google's VGGish model for deep audio embeddings:

   - Pre-trained on AudioSet dataset
   - Captures high-level audio patterns
   - Complements traditional signal processing features
   - Runs on CPU for broad compatibility

3. **Feature Robustness**: Multiple strategies ensure reliable feature extraction:
   - Energy-based segment filtering
   - Graceful handling of silent periods
   - Fallback values for failed extractions
   - Statistical aggregation reduces noise

### Performance Optimizations

1. **Async Processing**: Database operations and I/O are asynchronous
2. **Batch Duplicate Detection**: Single query to check all existing songs
3. **Model Reuse**: VGGish model loaded once and reused
4. **Memory Management**: Temporary files cleaned after each song
5. **Progress Tracking**: Detailed timing and progress information

### Design Decisions

1. **Database Choice**: Turso (libsql) chosen for:

   - Serverless scalability
   - SQL compatibility
   - Edge deployment capabilities

2. **Feature Storage**: JSON arrays in database for:

   - Flexibility in feature vector changes
   - Easy serialization/deserialization
   - Compatibility with various ML frameworks

3. **Artist Profiling**: Automatic updates ensure:
   - Fresh artist representations
   - Scalable processing
   - Consistent data quality

This sound engine provides a robust foundation for music analysis applications, combining traditional signal processing with modern deep learning approaches to create comprehensive audio representations suitable for recommendation systems, genre classification, and music research.
