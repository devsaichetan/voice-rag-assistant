import time

from groq import Groq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD
)

from src.retrieval import (
    load_index,
    load_metadata,
    load_embedding_model,
    retrieve
)


class RAGEngine:

    def __init__(self):

        print("Initializing lightweight RAG engine...")

        # ==================================================
        # Load FAISS index
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

        print("Loading lightweight embedding model...")

        self.embedding_model = load_embedding_model()

        # ==================================================
        # Initialize Groq
        # ==================================================

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set."
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
            f"Top-K: "
            f"{TOP_K}"
        )

        print(
            f"Similarity threshold: "
            f"{SIMILARITY_THRESHOLD}"
        )

        print(
            f"Groq model: "
            f"{GROQ_MODEL}"
        )

        print("Lightweight RAG engine ready!")

    # ==================================================
    # Create context
    # ==================================================

    def create_context(self, results):

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

URL:
{result.get('url', 'N/A')}

CONTENT:
{result.get('text', '')}
"""
            )

        return "\n".join(context_parts)

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
        # Debug information
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

    def ask(self, question):

        total_start = time.perf_counter()

        # ==================================================
        # Validate question
        # ==================================================

        if not question or not question.strip():

            return {
                "answer":
                    "Please provide a question.",

                "sources": [],

                "retrieval_time_ms": 0,

                "rerank_time_ms": 0,

                "generation_time_ms": 0,

                "total_time_ms":
                    (
                        time.perf_counter()
                        - total_start
                    ) * 1000
            }

        question = question.strip()

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
        # Similarity filtering
        # ==================================================

        filtered_results = [

            result

            for result in results

            if result.get("score", 0)
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
                    0,

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
                    result.get(
                        "chunk_id"
                    ),

                "score":
                    result.get(
                        "score",
                        0
                    ),

                # Reranker removed.
                # Kept as None for API compatibility.
                "reranker_score":
                    None,

                "url":
                    result.get(
                        "url",
                        ""
                    )
            })

        # ==================================================
        # Return result
        # ==================================================

        return {

            "answer":
                answer,

            "sources":
                sources,

            "retrieval_time_ms":
                retrieval_time,

            "rerank_time_ms":
                0,

            "generation_time_ms":
                generation_time,

            "total_time_ms":
                total_time
        }
