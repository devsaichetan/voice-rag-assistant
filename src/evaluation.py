import time
import json
import os

from src.config import (
    PROCESSED_DATA_DIR,
    TOP_K,
    RERANK_TOP_N
)

from src.retrieval import (
    load_index,
    load_metadata,
    load_embedding_model,
    retrieve
)

from src.reranker import (
    load_reranker,
    rerank
)


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


def build_evaluation_queries(
    documents,
    max_queries=100
):
    """
    Build evaluation queries using the
    MS MARCO is_selected labels.
    """

    query_data = {}

    for document in documents:

        query_id = document["query_id"]

        if query_id not in query_data:

            query_data[query_id] = {
                "query": document["query"],
                "relevant_passage_ids": set()
            }

        if document["is_selected"]:

            query_data[
                query_id
            ]["relevant_passage_ids"].add(
                document["passage_id"]
            )

    evaluation_queries = []

    for query_id, data in query_data.items():

        if not data["relevant_passage_ids"]:
            continue

        evaluation_queries.append({
            "query_id": query_id,
            "query": data["query"],
            "relevant_passage_ids":
                data["relevant_passage_ids"]
        })

        if len(evaluation_queries) >= max_queries:
            break

    return evaluation_queries


def calculate_recall_at_k(
    results,
    relevant_passage_ids,
    k
):
    """
    Recall@K = 1 when at least one relevant
    passage occurs in the first K results.
    """

    top_k_results = results[:k]

    retrieved_ids = {
        result["passage_id"]
        for result in top_k_results
    }

    if retrieved_ids.intersection(
        relevant_passage_ids
    ):

        return 1

    return 0


def calculate_mrr(
    results,
    relevant_passage_ids
):
    """
    Reciprocal rank of the first relevant result.
    """

    for rank, result in enumerate(
        results,
        start=1
    ):

        if result["passage_id"] in relevant_passage_ids:

            return 1 / rank

    return 0


def evaluate():

    print("Loading evaluation data...")

    documents = load_documents()

    evaluation_queries = build_evaluation_queries(
        documents,
        max_queries=100
    )

    print(
        f"Evaluation queries: "
        f"{len(evaluation_queries)}"
    )

    print("\nLoading retrieval components...")

    index = load_index()

    metadata = load_metadata()

    embedding_model = load_embedding_model()

    print("\nLoading reranker...")

    reranker_model = load_reranker()

    # -------------------------
    # Metric storage
    # -------------------------

    faiss_recall_5 = []
    faiss_recall_10 = []
    faiss_mrr = []

    rerank_recall_5 = []
    rerank_recall_10 = []
    rerank_mrr = []

    faiss_times = []
    rerank_times = []

    # -------------------------
    # Evaluation loop
    # -------------------------

    for i, item in enumerate(
        evaluation_queries,
        start=1
    ):

        query = item["query"]

        relevant_ids = item[
            "relevant_passage_ids"
        ]

        # ==================================================
        # FAISS
        # ==================================================

        faiss_start = time.perf_counter()

        results, _ = retrieve(
            query,
            embedding_model,
            index,
            metadata,
            TOP_K
        )

        faiss_time = (
            time.perf_counter()
            - faiss_start
        ) * 1000

        faiss_times.append(
            faiss_time
        )

        # FAISS metrics

        faiss_recall_5.append(
            calculate_recall_at_k(
                results,
                relevant_ids,
                5
            )
        )

        faiss_recall_10.append(
            calculate_recall_at_k(
                results,
                relevant_ids,
                10
            )
        )

        faiss_mrr.append(
            calculate_mrr(
                results,
                relevant_ids
            )
        )

        # ==================================================
        # RERANKER
        # ==================================================

        rerank_start = time.perf_counter()

        reranked_results = rerank(
            query,
            results,
            reranker_model,
            RERANK_TOP_N
        )

        rerank_time = (
            time.perf_counter()
            - rerank_start
        ) * 1000

        rerank_times.append(
            rerank_time
        )

        # Reranker metrics

        rerank_recall_5.append(
            calculate_recall_at_k(
                reranked_results,
                relevant_ids,
                5
            )
        )

        rerank_recall_10.append(
            calculate_recall_at_k(
                reranked_results,
                relevant_ids,
                10
            )
        )

        rerank_mrr.append(
            calculate_mrr(
                reranked_results,
                relevant_ids
            )
        )

        if i % 10 == 0:

            print(
                f"Processed "
                f"{i}/{len(evaluation_queries)}"
            )

    # ==================================================
    # FINAL METRICS
    # ==================================================

    faiss_recall_5_avg = (
        sum(faiss_recall_5)
        / len(faiss_recall_5)
    )

    faiss_recall_10_avg = (
        sum(faiss_recall_10)
        / len(faiss_recall_10)
    )

    faiss_mrr_avg = (
        sum(faiss_mrr)
        / len(faiss_mrr)
    )

    rerank_recall_5_avg = (
        sum(rerank_recall_5)
        / len(rerank_recall_5)
    )

    rerank_recall_10_avg = (
        sum(rerank_recall_10)
        / len(rerank_recall_10)
    )

    rerank_mrr_avg = (
        sum(rerank_mrr)
        / len(rerank_mrr)
    )

    faiss_latency = (
        sum(faiss_times)
        / len(faiss_times)
    )

    rerank_latency = (
        sum(rerank_times)
        / len(rerank_times)
    )

    # ==================================================
    # DISPLAY
    # ==================================================

    print("\n" + "=" * 70)

    print("RETRIEVAL COMPARISON")

    print("=" * 70)

    print(
        f"Queries evaluated: "
        f"{len(evaluation_queries)}"
    )

    print("\nFAISS ONLY")

    print(
        f"Recall@5: "
        f"{faiss_recall_5_avg:.4f}"
    )

    print(
        f"Recall@10: "
        f"{faiss_recall_10_avg:.4f}"
    )

    print(
        f"MRR: "
        f"{faiss_mrr_avg:.4f}"
    )

    print(
        f"Average latency: "
        f"{faiss_latency:.2f} ms"
    )

    print("\nFAISS + RERANKER")

    print(
        f"Recall@5: "
        f"{rerank_recall_5_avg:.4f}"
    )

    print(
        f"Recall@10: "
        f"{rerank_recall_10_avg:.4f}"
    )

    print(
        f"MRR: "
        f"{rerank_mrr_avg:.4f}"
    )

    print(
        f"Average reranking latency: "
        f"{rerank_latency:.2f} ms"
    )

    print(
        f"Average combined latency: "
        f"{faiss_latency + rerank_latency:.2f} ms"
    )

    print("\nIMPROVEMENT")

    print(
        f"Recall@5 improvement: "
        f"{rerank_recall_5_avg - faiss_recall_5_avg:+.4f}"
    )

    print(
        f"Recall@10 improvement: "
        f"{rerank_recall_10_avg - faiss_recall_10_avg:+.4f}"
    )

    print(
        f"MRR improvement: "
        f"{rerank_mrr_avg - faiss_mrr_avg:+.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate()