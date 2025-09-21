CREATE TABLE data_from_site (
id INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT NOT NULL,
    created_at NUMERIC DEFAULT (CURRENT_TIMESTAMP)
);

CREATE TABLE songs_from_playlists (
    song_id TEXT NOT NULL PRIMARY KEY,
    artist_id TEXT NOT NULL,
    song_name TEXT NOT NULL,
    artist_name TEXT NOT NULL
);

CREATE TABLE artists_vector (
        artist_id INT PRIMARY KEY,
        feature_vector VECTOR(223),
        song_count INT DEFAULT 0,
        artist_name TEXT NOT NULL DEFAULT '',
        followers INT,
        genres TEXT,
        images TEXT,
        popularity INT
);

CREATE TABLE songs (
        song_id TEXT PRIMARY KEY,
        artist_id INT NOT NULL,
        feature_vector VECTOR(223),
        song_name TEXT,
        artist_name TEXT,
        explicit BOOLEAN,
        popularity INT,
        duration_ms INT,
        sub_artists TEXT,
        album TEXT,
        FOREIGN KEY (artist_id) REFERENCES artists_vector(artist_id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
);

CREATE TABLE song_clusters (
    song_id TEXT NOT NULL,
    algorithm TEXT CHECK(algorithm IN ('kmeans', 'gmm', 'hierarchical', 'dbscan')),
    
    -- K-Means
    kmeans_cluster_id INT,
    kmeans_distance REAL,
    confidence REAL, -- for k-means only, others confidence can be derived from probabilities
    kmeans_cluster_size INT, -- number of points in this cluster
    kmeans_silhouette_score REAL, -- silhouette score for this point/cluster
    
    -- GMM
    gmm_cluster_id INT,
    gmm_probabilities TEXT, -- store JSON as text
    
    -- Hierarchical
    hier_level1_id INT, -- broad clusters
    hier_level2_id INT, -- fine clusters
    hier_distance REAL,
    
    -- DBSCAN
    dbscan_cluster_id INT, -- -1 for noise
    dbscan_is_core BOOLEAN,
    is_noise_point BOOLEAN, -- derived from dbscan_cluster_id == -1
    eps REAL, -- DBSCAN epsilon parameter
    min_samples INTEGER, -- DBSCAN min_samples parameter
    dbscan_cluster_size INTEGER, -- number of points in this DBSCAN cluster
    dbscan_silhouette_score REAL, -- silhouette score for DBSCAN clustering
    dbscan_n_clusters INTEGER, -- total number of DBSCAN clusters found
    dbscan_n_noise INTEGER, -- total number of noise points in DBSCAN
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (song_id, algorithm),
    FOREIGN KEY (song_id) REFERENCES songs(song_id)
);