import time

from google import genai

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL
)


def main():

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    question = "What is an Eiffel Tower?"

    context = """
SOURCE 1

Similarity Score:
0.8363

Reranker Score:
10.6762

URL:
example.com

CONTENT:
The Eiffel Tower is an iron lattice tower located
in Paris, France, and is one of the most well known
landmarks in the world. The tower was completed in
1889. It serves as a tourist attraction, a venue for
observations of the city of Paris, and as a location
for TV and radio broadcast antennae.

SOURCE 2

CONTENT:
Named after its designer, engineer Gustave Eiffel,
the Eiffel Tower was built between 1887 and 1889
as the entrance arch for the Exposition Universelle,
a World's Fair marking the centennial celebration
of the French Revolution.
"""

    prompt = f"""
You are a grounded question-answering assistant.

Answer ONLY using the provided context.

Do not use outside knowledge.
Do not invent facts.

If the context does not contain enough information,
say exactly:

"I don't have enough information in the provided context."

Give a concise answer in 1-3 sentences.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

    print("Sending RAG-style prompt to Gemini...")

    start = time.perf_counter()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(response.text)

    print("\n" + "=" * 60)

    print(
        f"Generation time: "
        f"{elapsed:.2f} ms"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()