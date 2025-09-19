# Audio Feature Extraction & Clustering System

## Project Overview

### Goal

This system implements an automated pipeline for extracting rich audio features from songs and applying machine learning clustering to enable music analysis, recommendation systems, and audio similarity matching. It bridges the gap between raw audio content and meaningful numerical representations that can be used in machine learning applications, with advanced clustering capabilities for music discovery and analysis.

### Main Functionalities

- **Automated Song Download**: Downloads audio from YouTube using song name and artist
- **Multi-dimensional Feature Extraction**: Extracts 200+ audio features including:
  - Traditional audio features (MFCCs, spectral features, tempo)
  - Deep learning embeddings (VGGish neural network features)
  - Rhythmic and harmonic characteristics
- **Advanced Clustering Pipeline**: Machine learning clustering with PCA dimensionality reduction
- **Database Integration**: Stores processed features and cluster assignments in Turso database
- **Artist Profile Generation**: Creates averaged feature vectors for artists based on their songs
- **Batch Processing**: Handles large playlists with error recovery and progress tracking
- **K-Means Clustering**: Automated music similarity clustering with elbow method optimization

### Problems It Solves

- **Music Recommendation**: Enables content-based filtering using audio similarity and clustering
- **Genre Classification**: Provides features for automatic music genre detection
- **Audio Analysis**: Supports research in music information retrieval
- **Playlist Generation**: Enables creation of playlists based on audio characteristics and cluster membership
- **Artist Similarity**: Quantifies musical similarity between artists and clusters
- **Music Discovery**: Identifies similar songs through cluster analysis
- **Content Organization**: Automatically groups songs by acoustic similarity

## Pipeline & Architecture

### Overview of the Processing Pipeline

```
YouTube URL/Search → Audio Download → Feature Extraction → Database Storage → Artist Profiling → Clustering Analysis
```

### Detailed Pipeline Steps

#### 1. **Song Collection** (`get_spotify_playlist_songs.py`)

- **Purpose**: Fetches song metadata from Spotify playlists
- **Input**: Spotify playlist URLs or IDs
- **Output**: Database table `songs_from_playlists` with song names, artists, and IDs
- **Why Necessary**: Provides structured input for the processing pipeline
- **Connection**: Seeds the main processing pipeline with songs to analyze

#### 2. **Audio Download** (`download_song.py`)

- **Purpose**: Downloads and preprocesses audio from YouTube
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

#### 6. **Clustering Pipeline** (`clustering/`)

- **Purpose**: Groups songs by acoustic similarity using machine learning
- **Input**: Feature vectors from processed songs
- **Output**: Cluster assignments and similarity metrics
- **Why Necessary**: Enables music discovery and recommendation through similarity grouping
- **Key Components**:
  - **Data Pipeline** (`clustering_util.py`): Unified extraction, verification, and PCA processing
  - **Feature Preprocessing** (`feature_preprocessing.py`): Intelligent scaling and weighting
  - **K-Means Implementation** (`k-means/k_means.py`): Custom clustering with multiple initializations
  - **Elbow Method** (`k-means/k_means_elbow.py`): Optimal cluster number selection
  - **Database Integration**: Stores cluster assignments in `song_clusters` table

## Clustering System

### Machine Learning Pipeline

The clustering system processes the 223-dimensional feature vectors through several stages:

#### 1. **Data Preparation Pipeline** (`clustering/clustering_util.py`)

```python
async def prepare_data_pipeline():
    # Data extraction and verification
    songs = await get_songs_from_db()
    verified_songs = verify_features(songs)
    
    # Feature processing with intelligent weighting
    processed_vectors = [process_and_return_vector(song, verified_songs) 
                        for song in verified_songs]
    
    # PCA dimensionality reduction (93% variance retention)
    pca = PCA(n_components=0.93, svd_solver='full')
    reduced_vectors = pca.fit_transform(processed_vectors)
    
    return reduced_vectors, pca, verified_songs
```

#### 2. **Enhanced Feature Preprocessing**

- **Intelligent Group Weighting**:
  - VGGish embeddings: 1.2x weight (deep learning features)
  - MFCC features: 1.05x weight (timbral characteristics)
  - Spectral features: 0.85x weight (frequency domain)
  - Rhythmic features: 1.1x weight (tempo and chroma)

- **Min-Max Scaling**: Features scaled to 1-10 range for optimal clustering
- **Data Cleaning**: NaN/infinity handling with robust error recovery

#### 3. **PCA Dimensionality Reduction**

- **Variance Retention**: Maintains 93% of original feature variance
- **Dimension Reduction**: Typically reduces from 223 to ~130 dimensions
- **Performance Benefits**: Faster clustering and reduced storage requirements
- **Noise Reduction**: Eliminates redundant feature correlations

#### 4. **K-Means Clustering Implementation**

```python
def kmeans(X, n_clusters=5, max_iters=100, tol=1e-4, n_init=10):
    # Multiple random initializations for robustness
    # Convergence monitoring with tolerance controls
    # Returns centroids, labels, and inertia for quality assessment
```

#### 5. **Elbow Method Optimization**

- **Automated K Selection**: Tests cluster numbers from 2-15
- **Inertia Analysis**: Computes within-cluster sum of squares
- **Visual Validation**: Generates elbow plots for optimal K identification
- **Integration Ready**: Seamlessly works with the main clustering pipeline

### Database Schema for Clustering

```sql
CREATE TABLE song_clusters (
    song_id TEXT NOT NULL,
    algorithm TEXT CHECK(algorithm IN ('kmeans', 'gmm', 'hierarchical', 'dbscan')),
    
    -- K-Means specific fields
    kmeans_cluster_id INT,
    kmeans_distance REAL,
    
    -- Future algorithm support
    gmm_cluster_id INT,
    gmm_probabilities TEXT,  -- JSON format
    hier_level1_id INT,      -- Hierarchical clustering
    hier_level2_id INT,
    dbscan_cluster_id INT,   -- DBSCAN clustering
    
    confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (song_id, algorithm),
    FOREIGN KEY (song_id) REFERENCES songs(song_id)
);
```

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
scikit-learn>=1.3.0      # Machine learning (PCA, preprocessing)
matplotlib>=3.7.0        # Plotting for elbow method
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

Run the complete database setup using the provided schema:

```bash
# Execute the SQL schema file
sqlite3 your_database.db < tables.sql
```

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

-- Clustering results
CREATE TABLE song_clusters (
    song_id TEXT NOT NULL,
    algorithm TEXT,
    kmeans_cluster_id INT,
    kmeans_distance REAL,
    confidence REAL,
    PRIMARY KEY (song_id, algorithm)
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

### 3. Run Clustering Analysis

#### Complete Clustering Pipeline
```bash
python clustering/main.py
```

This executes the unified data pipeline with PCA processing.

#### Find Optimal Number of Clusters
```bash
python clustering/k-means/k_means_elbow.py
```

This generates an elbow plot to determine the optimal K value for clustering.

#### Programmatic Clustering Usage
```python
import asyncio
from clustering.clustering_util import prepare_data_pipeline
from clustering.k_means.k_means import kmeans

async def run_clustering():
    # Prepare data with PCA
    reduced_vectors, pca, songs = await prepare_data_pipeline()
    
    # Perform K-Means clustering
    centroids, labels, inertia = kmeans(reduced_vectors, n_clusters=8)
    
    # Use cluster assignments for analysis
    for i, song in enumerate(songs):
        cluster_id = labels[i]
        print(f"{song['song_name']} → Cluster {cluster_id}")

asyncio.run(run_clustering())
```

### 4. Fill Missing Artist Information (Optional)

```bash
python fill_missing_artists_fields.py
python fill_missing_song_artist_fields.py
```

These scripts enrich the database with additional metadata.

### 5. Manual Artist Vector Updates

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

### Clustering Modules

#### `clustering/clustering_util.py` - Unified Data Pipeline

**Purpose**: Provides a streamlined pipeline for clustering preparation

**Key Functions**:

- `prepare_data_pipeline()`: Complete data extraction, verification, EDA, and PCA processing

**Features**:

- Integrated data quality reporting
- Automatic PCA dimensionality reduction
- Comprehensive error handling
- Async processing support

#### `clustering/feature_preprocessing.py` - Advanced Feature Processing

**Purpose**: Intelligent feature scaling and weighting for clustering

**Key Functions**:

- `process_and_return_vector()`: Single-song feature processing with weighting
- `clean_feature_vector()`: Data sanitization and normalization
- `min_max_scale()`: Custom scaling to 1-10 range

**Feature Group Weights**:

```python
GROUPS = {
    "vggish":   {"slice": slice(95, 223),   "weight": 1.2},  # Deep learning
    "mfcc":     {"slice": slice(31, 83),    "weight": 1.05}, # Timbral
    "spectral": {"slice": slice(11, 31),    "weight": 0.85}, # Frequency
    "rhythmic": {"slice": slice(3, 11),     "weight": 1.1}   # Tempo/Chroma
}
```

#### `clustering/k-means/k_means.py` - K-Means Implementation

**Purpose**: Custom K-Means clustering optimized for audio features

**Key Functions**:

- `kmeans()`: Main clustering function with multiple initializations
- `compute_inertia()`: Quality metric calculation

**Features**:

- Multiple random initializations (n_init=10)
- Convergence monitoring with tolerance
- Inertia-based quality assessment
- Robust centroid updates

#### `clustering/k-means/k_means_elbow.py` - Elbow Method

**Purpose**: Automated optimal cluster number selection

**Features**:

- Tests K values from 2-15
- Generates elbow plots
- Uses PCA-reduced features
- Integrated with main pipeline

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
- Enhanced conditional update logic

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

4. **Clustering Pipeline Workflow**

   ```
   Feature Vectors → Data Verification → EDA → Feature Preprocessing → PCA Reduction → K-Means Clustering → Store Results
   ```

5. **Artist Profile Updates**
   - Triggered when artist has ≥5 processed songs
   - Computes average feature vector across all artist's songs
   - Stores in `artists_vector` table

### Error Handling Strategy

- Graceful failure handling at each pipeline stage
- Comprehensive error logging to `error_log.json`
- Automatic cleanup of temporary files
- Progress preservation (can resume after interruption)
- Robust data validation throughout clustering pipeline

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

### Clustering Feature Processing

- **Intelligent weighting**: Different feature groups weighted based on importance for clustering
- **Min-max scaling**: All features scaled to 1-10 range for consistent clustering performance
- **PCA reduction**: Dimensionality reduced while maintaining 93% of variance
- **Quality control**: Comprehensive data validation and cleaning before clustering

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

#### In `clustering/feature_preprocessing.py`:

- **Feature group weights**:
  - VGGish: 1.2x
  - MFCC: 1.05x
  - Spectral: 0.85x
  - Rhythmic: 1.1x
- **Scaling range**: 1-10 (min-max scaling)
- **PCA variance retention**: 93%

#### In `clustering/k-means/k_means.py`:

- `n_clusters`: Number of clusters (default: 5)
- `max_iters`: Maximum iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-4)
- `n_init`: Number of random initializations (default: 10)

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

### Clustering System Insights

1. **PCA Benefits**: Dimensionality reduction provides multiple advantages:

   - **Performance**: Faster clustering on reduced dimensions
   - **Storage**: More efficient database storage
   - **Noise Reduction**: Eliminates redundant feature correlations
   - **Visualization**: Enables 2D/3D cluster visualization

2. **Feature Weighting Strategy**: Different weights optimize clustering quality:

   - **VGGish**: Higher weight for deep learning semantic features
   - **MFCC**: Moderate weight for timbral characteristics
   - **Spectral**: Lower weight to balance frequency domain features
   - **Rhythmic**: Higher weight for tempo and harmonic content

