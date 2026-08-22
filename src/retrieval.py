import json
import os
import time

import faiss
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.config import (
    INDEX_DIR,
    EMBEDDING_MODEL,
    TOP_K
)


# ============================================================
# Lightweight ONNX Embedding Model
# ============================================================

class ONNXEmbeddingModel:

    def __init__(self, model_dir):

        print("Loading ONNX embedding model...")

        self.model_dir = model_dir

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )

        # ----------------------------------------------------
        # ONNX model
        # ----------------------------------------------------

        model_path = os.path.join(
            model_dir,
            "model.onnx"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"ONNX model not found: {model_path}"
            )

        print(
            f"Loading ONNX model:\n{model_path}"
        )

        # ----------------------------------------------------
        # ONNX Runtime
        # ----------------------------------------------------

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_names = {
            inp.name
            for inp in self.session.get_inputs()
        }

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

        print(
            f"ONNX inputs: "
            f"{list(self.input_names)}"
        )

        print(
            f"ONNX outputs: "
            f"{self.output_names}"
        )

        print("ONNX embedding model ready!")


    # ========================================================
    # Mean Pooling
    # ========================================================

    @staticmethod
    def mean_pooling(
        token_embeddings,
        attention_mask
    ):

        mask = attention_mask[
            :, :, None
        ].astype(
            np.float32
        )

        masked_embeddings = (
            token_embeddings * mask
        )

        summed = masked_embeddings.sum(
            axis=1
        )

        counts = np.clip(
            mask.sum(axis=1),
            a_min=1e-9,
            a_max=None
        )

        return summed / counts


    # ========================================================
    # Normalize embeddings
    # ========================================================

    @staticmethod
    def normalize(
        embeddings
    ):

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        return embeddings / np.clip(
            norms,
            a_min=1e-12,
            a_max=None
        )


    # ========================================================
    # Encode
    # ========================================================

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        **kwargs
    ):

        if isinstance(
            texts,
            str
        ):

            texts = [texts]

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np"
        )

        # ----------------------------------------------------
        # Prepare ONNX inputs
        # ----------------------------------------------------

        inputs = {}

        if "input_ids" in self.input_names:

            inputs["input_ids"] = (
                encoded["input_ids"]
                .astype(np.int64)
            )

        if "attention_mask" in self.input_names:

            inputs["attention_mask"] = (
                encoded["attention_mask"]
                .astype(np.int64)
            )

        if "token_type_ids" in self.input_names:

            inputs["token_type_ids"] = (
                encoded["token_type_ids"]
                .astype(np.int64)
            )

        # ----------------------------------------------------
        # Run ONNX
        # ----------------------------------------------------

        outputs = self.session.run(
            None,
            inputs
        )

        # ----------------------------------------------------
        # Get embedding output
        # ----------------------------------------------------

        embedding = outputs[0]

        # ----------------------------------------------------
        # Case 1:
        # Model already returns sentence embeddings
        #
        # Shape:
        # [batch, embedding_dimension]
        # ----------------------------------------------------

        if embedding.ndim == 2:

            sentence_embeddings = embedding

        # ----------------------------------------------------
        # Case 2:
        # Model returns token embeddings
        #
        # Shape:
        # [batch, sequence_length, hidden_dimension]
        # ----------------------------------------------------

        elif embedding.ndim == 3:

            sentence_embeddings = (
                self.mean_pooling(
                    embedding,
                    encoded["attention_mask"]
                )
            )

        else:

            raise RuntimeError(
                "Unexpected ONNX output shape: "
                f"{embedding.shape}"
            )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        if normalize_embeddings:

            sentence_embeddings = (
                self.normalize(
                    sentence_embeddings
                )
            )

        if convert_to_numpy:

            return sentence_embeddings.astype(
                "float32"
            )

        return sentence_embeddings


# ============================================================
# Load FAISS index
# ============================================================

def load_index():

    index_path = os.path.join(
        INDEX_DIR,
        "faiss.index"
    )

    if not os.path.exists(
        index_path
    ):

        raise FileNotFoundError(
            f"FAISS index not found: "
            f"{index_path}"
        )

    print(
        f"Loading lightweight FAISS index:\n"
        f"{index_path}"
    )

    return faiss.read_index(
        index_path
    )


# ============================================================
# Load metadata
# ============================================================

def load_metadata():

    metadata_path = os.path.join(
        INDEX_DIR,
        "metadata.json"
    )

    if not os.path.exists(
        metadata_path
    ):

        raise FileNotFoundError(
            f"Metadata not found: "
            f"{metadata_path}"
        )

    print(
        f"Loading metadata:\n"
        f"{metadata_path}"
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# Load embedding model
# ============================================================

def load_embedding_model():

    return ONNXEmbeddingModel(
        EMBEDDING_MODEL
    )


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    query,
    model,
    index,
    metadata,
    top_k=TOP_K
):

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Convert query to embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(
        "float32"
    )

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieval_time = (
        time.perf_counter()
        - start_time
    ) * 1000

    results = []

    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:

            continue

        result = metadata[
            index_position
        ].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return (
        results,
        retrieval_time
    )


# ============================================================
# Test retrieval
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "LIGHTWEIGHT RETRIEVAL SYSTEM"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print(
        "\nLoading FAISS index..."
    )

    index = load_index()

    print(
        f"FAISS contains "
        f"{index.ntotal} vectors."
    )

    print(
        f"Embedding dimension: "
        f"{index.d}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print(
        "\nLoading metadata..."
    )

    metadata = load_metadata()

    print(
        f"Loaded "
        f"{len(metadata)} metadata records."
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = load_embedding_model()

    print(
        "\nRetrieval system ready!"
    )

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "Please enter a question."
        )

        return

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results, retrieval_time = retrieve(
        query,
        model,
        index,
        metadata,
        TOP_K
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL RESULTS"
    )

    print(
        "=" * 70
    )

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

        print(
            f"URL:\n"
            f"{result['url']}"
        )

        print(
            "-" * 70
        )

    print(
        f"\nRetrieval time: "
        f"{retrieval_time:.2f} ms"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()
