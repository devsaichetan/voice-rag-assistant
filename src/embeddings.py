import json
import os

import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    PROCESSED_DATA_DIR,
    INDEX_DIR,
    EMBEDDING_MODEL
)


def load_chunks():
    path = os.path.join(
        PROCESSED_DATA_DIR,
        "chunks.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def create_embeddings(texts, model):

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype("float32")


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def save_index(index):

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )

    index_path = os.path.join(
        INDEX_DIR,
        "faiss.index"
    )

    faiss.write_index(
        index,
        index_path
    )

    print(
        f"FAISS index saved to: {index_path}"
    )


def save_metadata(chunks):

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )

    metadata_path = os.path.join(
        INDEX_DIR,
        "metadata.json"
    )

    metadata = []

    for chunk in chunks:

        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "query_id": chunk["query_id"],
            "passage_id": chunk["passage_id"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"],
            "url": chunk["url"],
            "is_selected": chunk["is_selected"]
        })

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False
        )

    print(
        f"Metadata saved to: {metadata_path}"
    )


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        texts,
        model
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    index = create_faiss_index(
        embeddings
    )

    print(
        f"FAISS index contains "
        f"{index.ntotal} vectors."
    )

    save_index(index)

    save_metadata(chunks)

    print(
        "\nEmbedding pipeline completed successfully!"
    )


if __name__ == "__main__":
    main()