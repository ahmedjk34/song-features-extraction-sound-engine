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
    song_id INT PRIMARY KEY,
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
