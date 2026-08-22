from datasets import load_dataset
import json
import os

from src.config import (
    DATASET_NAME,
    DATASET_CONFIG,
    DATASET_SPLIT,
    PROCESSED_DATA_DIR
)


MAX_SAMPLES = 1000


def load_ms_marco():
    print("Loading MS MARCO...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT
    )

    print(f"Total samples available: {len(dataset)}")

    return dataset


def extract_passages(dataset, max_samples=MAX_SAMPLES):

    documents = []

    print(f"\nProcessing first {max_samples} samples...")

    limit = min(max_samples, len(dataset))

    for sample in dataset.select(range(limit)):

        query = sample["query"]
        query_id = sample["query_id"]

        passages = sample["passages"]

        passage_texts = passages["passage_text"]
        selected = passages["is_selected"]
        urls = passages["url"]

        for i, text in enumerate(passage_texts):

            text = text.strip()

            if not text:
                continue

            document = {
                "query_id": query_id,
                "query": query,
                "passage_id": i,
                "text": text,
                "is_selected": selected[i],
                "url": urls[i]
            }

            documents.append(document)

    return documents


def save_documents(documents):

    os.makedirs(
        PROCESSED_DATA_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        PROCESSED_DATA_DIR,
        "documents.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved {len(documents)} passages.")
    print(f"File: {output_path}")


def main():

    dataset = load_ms_marco()

    documents = extract_passages(dataset)

    save_documents(documents)

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()