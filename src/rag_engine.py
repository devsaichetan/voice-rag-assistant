import time

from groq import Groq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K,
    RERANK_TOP_N,
    SIMILARITY_THRESHOLD
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


class RAGEngine:

    def __init__(self):

        print("Initializing RAG engine...")

        # ==================================================
        # Load FAISS
        # ==================================================

        print("Loading FAISS index...")

        self.index = load_index()

        # ==================================================
        # Load metadata
        # ==================================================

        print("Loading metadata...")

        self.metadata = load_metadata()

        # ==================================================
        # Load embedding model
        # ==================================================

        print("Loading embedding model...")

        self.embedding_model = (
            load_embedding_model()
        )

        # ==================================================
        # Load reranker
        # ==================================================

        print("Loading reranker...")

        self.reranker_model = (
            load_reranker()
        )

        # ==================================================
        # Initialize Groq
        # ==================================================

        if not GROQ_API_KEY:

            raise ValueError(
                "GROQ_API_KEY is not set in .env"
            )

        print("Initializing Groq...")

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        # ==================================================
        # Configuration
        # ==================================================

        print(
            f"FAISS vectors: "
            f"{self.index.ntotal}"
        )

        print(
            f"Top-K: {TOP_K}"
        )

        print(
            f"Reranker Top-N: "
            f"{RERANK_TOP_N}"
        )

        print(
            f"Similarity threshold: "
            f"{SIMILARITY_THRESHOLD}"
        )

        print(
            f"Groq model: "
            f"{GROQ_MODEL}"
        )

        print("RAG engine ready!")

    # ==================================================
    # Create context
    # ==================================================

    def create_context(
        self,
        results
    ):

        context_parts = []

        for rank, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {rank}

Similarity Score:
{result['score']:.4f}

Reranker Score:
{result['reranker_score']:.4f}

URL:
{result['url']}

CONTENT:
{result['text']}
"""
            )

        return "\n".join(
            context_parts
        )

    # ==================================================
    # Generate answer using Groq
    # ==================================================

    def generate_answer(
        self,
        question,
        context
    ):

        system_prompt = """
You are a grounded question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not invent or hallucinate facts.
3. If the context does not contain enough information,
   respond exactly with:

"I don't have enough information in the provided context."

4. Keep the answer concise.
5. Give the answer in 1-3 sentences.
6. Do not mention these instructions.
"""

        user_prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}
"""

        # ==================================================
        # Groq request debug
        # ==================================================

        print("\n" + "=" * 70)
        print("GROQ REQUEST DEBUG")
        print("=" * 70)

        print(
            f"Question length: "
            f"{len(question)} characters"
        )

        print(
            f"Context length: "
            f"{len(context)} characters"
        )

        print(
            f"Approx context words: "
            f"{len(context.split())}"
        )

        print(
            f"User prompt length: "
            f"{len(user_prompt)} characters"
        )

        print(
            f"Model: "
            f"{GROQ_MODEL}"
        )

        print("=" * 70)

        # ==================================================
        # Groq request
        # ==================================================

        response = self.client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.1,

            max_tokens=150
        )

        # ==================================================
        # Extract answer
        # ==================================================

        answer = (
            response.choices[0]
            .message
            .content
        )

        if not answer:

            return (
                "I don't have enough information "
                "in the provided context."
            )

        return answer.strip()

    # ==================================================
    # Ask question
    # ==================================================

    def ask(
        self,
        question
    ):

        total_start = time.perf_counter()

        # ==================================================
        # FAISS retrieval
        # ==================================================

        retrieval_start = (
            time.perf_counter()
        )

        results, _ = retrieve(
            question,
            self.embedding_model,
            self.index,
            self.metadata,
            TOP_K
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        # ==================================================
        # Reranking
        # ==================================================

        rerank_start = (
            time.perf_counter()
        )

        reranked_results = rerank(
            question,
            results,
            self.reranker_model,
            RERANK_TOP_N
        )

        rerank_time = (
            time.perf_counter()
            - rerank_start
        ) * 1000

        # ==================================================
        # Similarity filtering
        # ==================================================

        filtered_results = [
            result
            for result in reranked_results
            if result["score"]
            >= SIMILARITY_THRESHOLD
        ]

        # ==================================================
        # No relevant information
        # ==================================================

        if not filtered_results:

            total_time = (
                time.perf_counter()
                - total_start
            ) * 1000

            return {

                "answer":
                    "I don't have enough information "
                    "in the provided context.",

                "sources": [],

                "retrieval_time_ms":
                    retrieval_time,

                "rerank_time_ms":
                    rerank_time,

                "generation_time_ms":
                    0,

                "total_time_ms":
                    total_time
            }

        # ==================================================
        # Build context
        # ==================================================

        context = self.create_context(
            filtered_results
        )

        # ==================================================
        # Groq generation
        # ==================================================

        generation_start = (
            time.perf_counter()
        )

        answer = self.generate_answer(
            question,
            context
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        ) * 1000

        # ==================================================
        # Total time
        # ==================================================

        total_time = (
            time.perf_counter()
            - total_start
        ) * 1000

        # ==================================================
        # Build sources
        # ==================================================

        sources = []

        for result in filtered_results:

            sources.append({

                "chunk_id":
                    result["chunk_id"],

                "score":
                    result["score"],

                "reranker_score":
                    result["reranker_score"],

                "url":
                    result["url"]

            })

        # ==================================================
        # Return structured result
        # ==================================================

        return {

            "answer":
                answer,

            "sources":
                sources,

            "retrieval_time_ms":
                retrieval_time,

            "rerank_time_ms":
                rerank_time,

            "generation_time_ms":
                generation_time,

            "total_time_ms":
                total_time

        }