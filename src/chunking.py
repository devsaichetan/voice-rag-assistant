import json
import os
import re

from src.config import PROCESSED_DATA_DIR


TARGET_WORDS = 100
MAX_WORDS = 150
OVERLAP_SENTENCES = 1


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def create_chunks(text):

    text = clean_text(text)

    if not text:
        return []

    words = text.split()

    # Keep short passages intact
    if len(words) <= MAX_WORDS:
        return [text]

    sentences = split_into_sentences(text)

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence_words = len(sentence.split())

        if (
            current_sentences
            and current_word_count + sentence_words > MAX_WORDS
        ):

            chunks.append(
                " ".join(current_sentences)
            )

            # Sentence overlap
            current_sentences = current_sentences[
                -OVERLAP_SENTENCES:
            ]

            current_word_count = sum(
                len(s.split())
                for s in current_sentences
            )

        current_sentences.append(sentence)
        current_word_count += sentence_words

        if current_word_count >= TARGET_WORDS:

            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = current_sentences[
                -OVERLAP_SENTENCES:
            ]

            current_word_count = sum(
                len(s.split())
                for s in current_sentences
            )

    if current_sentences:

        chunk = " ".join(current_sentences)

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def process_documents(documents):

    chunked_documents = []
    chunk_id = 0

    for document in documents:

        chunks = create_chunks(
            document["text"]
        )

        for chunk_number, chunk in enumerate(chunks):

            chunked_documents.append({
                "chunk_id": chunk_id,
                "query_id": document["query_id"],
                "passage_id": document["passage_id"],
                "chunk_number": chunk_number,
                "text": chunk,
                "url": document["url"],
                "is_selected": document["is_selected"]
            })

            chunk_id += 1

    return chunked_documents


def load_documents():

    path = os.path.join(
        PROCESSED_DATA_DIR,
        "documents.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_chunks(chunks):

    output_path = os.path.join(
        PROCESSED_DATA_DIR,
        "chunks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_path


def main():

    print("Loading documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )

    print("\nCreating smart chunks...")

    chunks = process_documents(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    output_path = save_chunks(chunks)

    print("\nChunks saved to:")
    print(output_path)


if __name__ == "__main__":
    main()