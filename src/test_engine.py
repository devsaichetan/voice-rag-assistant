from src.rag_engine import RAGEngine


def main():

    engine = RAGEngine()

    while True:

        question = input(
            "\nEnter your question "
            "(type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":

            print("Exiting...")

            break

        if not question:

            print("Please enter a question.")

            continue

        result = engine.ask(
            question
        )

        print("\n" + "=" * 70)

        print("ANSWER")

        print("=" * 70)

        print(
            result["answer"]
        )

        print("\nSOURCES")

        print("=" * 70)

        for source in result["sources"]:

            print(
                f"Chunk ID: "
                f"{source['chunk_id']}"
            )

            print(
                f"FAISS Score: "
                f"{source['score']:.4f}"
            )

            print(
                f"Reranker Score: "
                f"{source['reranker_score']:.4f}"
            )

            print(
                f"URL: "
                f"{source['url']}"
            )

            print("-" * 70)

        print("\nTIMING")

        print("=" * 70)

        print(
            f"Retrieval: "
            f"{result['retrieval_time_ms']:.2f} ms"
        )

        print(
            f"Reranking: "
            f"{result['rerank_time_ms']:.2f} ms"
        )

        print(
            f"Generation: "
            f"{result['generation_time_ms']:.2f} ms"
        )

        print(
            f"Total: "
            f"{result['total_time_ms']:.2f} ms"
        )

        print("=" * 70)


if __name__ == "__main__":
    main()