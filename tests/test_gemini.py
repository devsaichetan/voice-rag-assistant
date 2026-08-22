import time

from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL
)


def main():

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set in .env"
        )

    print("Initializing Gemini...")

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    print(
        f"Model: {GEMINI_MODEL}"
    )

    prompt = """
Answer this question in one short sentence:

What is the capital of France?
"""

    print("\nSending request...")

    start = time.perf_counter()

    response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="low"
        ),
        max_output_tokens=300
    )
)

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    print("\n" + "=" * 60)
    print("GEMINI RESPONSE")
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