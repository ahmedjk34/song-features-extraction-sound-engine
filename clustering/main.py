import asyncio
from data_extraction import get_songs_from_db, verify_features, data_quality_report
from data_eda import data_eda
from feature_preprocessing import process_and_store_features

async def main():
    # Part 1 of the pipeline: Data Extraction and Verification
    songs = await get_songs_from_db()
    if songs is None or len(songs) == 0:
        return
    
    verified_songs = verify_features(songs)
    data_quality_report(verified_songs)

    # Part 2 of the pipeline: Exploratory Data Analysis
    data_eda(verified_songs)


    # part 3 of the pipeline: Feature Preprocessing and Normalization
    process_and_store_features(verified_songs, save_path="normalized_vectors.npy")


if __name__ == "__main__":
    asyncio.run(main())