from src.voice_rag import process_voice


def main():

    audio_path = input(
        "Enter audio file path: "
    ).strip()

    print("\nProcessing voice...")

    result = process_voice(
        audio_path
    )

    print("\n" + "=" * 70)
    print("VOICE RAG RESULT")
    print("=" * 70)

    print("\nTRANSCRIPT:")
    print(result["transcript"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nAUDIO:")
    print(result["audio_path"])

    print("\nTIMING:")

    print(
        f"STT: "
        f"{result['timing']['stt_ms']:.2f} ms"
    )

    print(
        f"Retrieval: "
        f"{result['timing']['retrieval_ms']:.2f} ms"
    )

    print(
        f"Reranking: "
        f"{result['timing']['reranking_ms']:.2f} ms"
    )

    print(
        f"Generation: "
        f"{result['timing']['generation_ms']:.2f} ms"
    )

    print(
        f"TTS: "
        f"{result['timing']['tts_ms']:.2f} ms"
    )

    print(
        f"TOTAL: "
        f"{result['timing']['total_ms']:.2f} ms"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()