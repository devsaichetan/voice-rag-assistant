import json
import os
import time

import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    INDEX_DIR,
    EMBEDDING_MODEL,
    TOP_K
)


def load_index():
    """Load the FAISS index."""

    index_path = os.path.join(
        INDEX_DIR,
        "faiss.index"
    )

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    return faiss.read_index(index_path)


def load_metadata():
    """Load metadata corresponding to FAISS vectors."""

    metadata_path = os.path.join(
        INDEX_DIR,
        "metadata.json"
    )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def retrieve(
    query,
    model,
    index,
    metadata,
    top_k=TOP_K
):

    start_time = time.perf_counter()

    # Convert query into vector
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieval_time = (
        time.perf_counter() - start_time
    ) * 1000

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        result = metadata[
            index_position
        ].copy()

        result["score"] = float(score)

        results.append(result)

    return results, retrieval_time


def main():

    print("Loading FAISS index...")

    index = load_index()

    print(
        f"FAISS contains "
        f"{index.ntotal} vectors."
    )

    print("Loading metadata...")

    metadata = load_metadata()

    print(
        f"Loaded {len(metadata)} metadata records."
    )

    print("Loading embedding model...")

    model = load_embedding_model()

    print("\nRetrieval system ready!")

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print("Please enter a question.")

        return

    results, retrieval_time = retrieve(
        query,
        model,
        index,
        metadata
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Text:\n"
            f"{result['text']}"
        )

        print("-" * 70)

    print(
        f"\nRetrieval time: "
        f"{retrieval_time:.2f} ms"
    )


if __name__ == "__main__":
    main()