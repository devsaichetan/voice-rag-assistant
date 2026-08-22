from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():

    print("Loading reranker model...")

    model = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        f"Reranker model: {RERANKER_MODEL}"
    )

    return model


def rerank(
    query,
    results,
    model,
    top_n=5
):
    """
    Rerank retrieved passages using
    a cross-encoder.
    """

    if not results:
        return []

    pairs = [
        (
            query,
            result["text"]
        )
        for result in results
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False
    )

    reranked_results = []

    for result, score in zip(
        results,
        scores
    ):

        result_copy = result.copy()

        result_copy[
            "reranker_score"
        ] = float(score)

        reranked_results.append(
            result_copy
        )

    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_n]