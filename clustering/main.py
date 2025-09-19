import asyncio
from clustering_util import prepare_data_pipeline

async def main():
    reduced_vectors, pca, verified_songs = await prepare_data_pipeline()
    if reduced_vectors is None:
        return

    # Now you can continue using reduced_vectors for clustering, ML, etc.
    print("Data pipeline ready! Ready for next steps.")


if __name__ == "__main__":
    asyncio.run(main())
