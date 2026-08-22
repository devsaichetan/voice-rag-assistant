from src.stt import transcribe_audio


def main():

    audio_path = input(
        "Enter audio file path: "
    ).strip()

    print("\nTranscribing...")

    transcript = transcribe_audio(
        audio_path,
        language_code="en-IN"
    )

    print("\n" + "=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)

    print(transcript)

    print("=" * 60)


if __name__ == "__main__":
    main()