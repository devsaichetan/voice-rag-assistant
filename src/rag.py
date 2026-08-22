import time

from google import genai

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K,
    RERANK_TOP_N,
    SIMILARITY_THRESHOLD
)

from src.reranker import (
    load_reranker,
    rerank
)

from src.retrieval import (
    load_index,
    load_metadata,
    load_embedding_model,
    retrieve
)


def create_context(results):
    """
    Combine retrieved chunks into a context
    that will be provided to Gemini.
    """

    context_parts = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {rank}
FAISS Similarity Score: {result['score']:.4f}
Reranker Score: {result['reranker_score']:.4f}
URL: {result['url']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


def generate_answer(
    question,
    context,
    client
):
    """
    Generate a grounded answer using Gemini.
    """

    prompt = f"""
You are a grounded question-answering assistant.

Answer the user's question ONLY using the
information provided in the context below.

Do not use outside knowledge.

If the context does not contain enough information
to answer the question, say:

"I don't have enough information in the provided context."

Do not invent or hallucinate facts.

Keep the answer concise and directly relevant
to the user's question.

USER QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


def filter_results(
    results,
    threshold=SIMILARITY_THRESHOLD
):
    """
    Keep only results whose FAISS similarity
    score is above the threshold.
    """

    filtered_results = [
        result
        for result in results
        if result["score"] >= threshold
    ]

    return filtered_results


def main():

    # -------------------------
    # Check Gemini API key
    # -------------------------

    if not GEMINI_API_KEY:

        raise ValueError(
            "GEMINI_API_KEY is not set in .env"
        )

    # -------------------------
    # Load components
    # -------------------------

    print("Loading RAG components...")

    index = load_index()

    metadata = load_metadata()

    embedding_model = load_embedding_model()

    reranker_model = load_reranker()

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    print(
        f"FAISS vectors: {index.ntotal}"
    )

    print(
        f"FAISS Top-K: {TOP_K}"
    )

    print(
        f"Reranker Top-N: {RERANK_TOP_N}"
    )

    print(
        f"Similarity threshold: "
        f"{SIMILARITY_THRESHOLD}"
    )

    print("\nRAG system ready!")

    # -------------------------
    # Get user question
    # -------------------------

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print("Please enter a question.")

        return

    # ==================================================
    # STEP 1 — FAISS RETRIEVAL
    # ==================================================

    retrieval_start = time.perf_counter()

    results, _ = retrieve(
        question,
        embedding_model,
        index,
        metadata,
        TOP_K
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    ) * 1000

    print(
        f"\nFAISS retrieved "
        f"{len(results)} candidates."
    )

    # ==================================================
    # STEP 2 — CROSS-ENCODER RERANKING
    # ==================================================

    rerank_start = time.perf_counter()

    reranked_results = rerank(
        question,
        results,
        reranker_model,
        RERANK_TOP_N
    )

    rerank_time = (
        time.perf_counter()
        - rerank_start
    ) * 1000

    print(
        f"Reranker selected "
        f"{len(reranked_results)} chunks."
    )

    # ==================================================
    # STEP 3 — SIMILARITY FILTER
    # ==================================================

    filtered_results = filter_results(
        reranked_results
    )

    print(
        f"After similarity filtering: "
        f"{len(filtered_results)} chunks."
    )

    # ==================================================
    # STEP 4 — NO RELEVANT CONTEXT
    # ==================================================

    if not filtered_results:

        print("\n" + "=" * 70)

        print("RAG ANSWER")

        print("=" * 70)

        print(
            "I don't have enough information "
            "in the provided context."
        )

        print("\n" + "-" * 70)

        print(
            f"FAISS retrieval time: "
            f"{retrieval_time:.2f} ms"
        )

        print(
            f"Reranking time: "
            f"{rerank_time:.2f} ms"
        )

        print(
            f"Total processing time: "
            f"{retrieval_time + rerank_time:.2f} ms"
        )

        print("=" * 70)

        return

    # ==================================================
    # STEP 5 — BUILD CONTEXT
    # ==================================================

    context = create_context(
        filtered_results
    )

    # ==================================================
    # STEP 6 — GEMINI GENERATION
    # ==================================================

    generation_start = time.perf_counter()

    answer = generate_answer(
        question,
        context,
        client
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    ) * 1000

    # ==================================================
    # STEP 7 — DISPLAY ANSWER
    # ==================================================

    print("\n" + "=" * 70)

    print("RAG ANSWER")

    print("=" * 70)

    print(answer)

    print("\n" + "-" * 70)

    print(
        f"FAISS retrieval time: "
        f"{retrieval_time:.2f} ms"
    )

    print(
        f"Reranking time: "
        f"{rerank_time:.2f} ms"
    )

    print(
        f"Generation time: "
        f"{generation_time:.2f} ms"
    )

    print(
        f"Total time: "
        f"{retrieval_time + rerank_time + generation_time:.2f} ms"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()