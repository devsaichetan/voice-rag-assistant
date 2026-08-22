from src.tts import text_to_speech


def main():

    text = input(
        "Enter text to convert to speech: "
    ).strip()

    if not text:

        print("Please enter some text.")

        return

    output_path = (
        "data/raw/answer.wav"
    )

    print("\nGenerating speech...")

    result = text_to_speech(
        text=text,
        output_path=output_path,
        language_code="en-IN",
        speaker="shubh"
    )

    print("\n" + "=" * 60)
    print("TTS SUCCESS")
    print("=" * 60)

    print(
        f"Audio saved to: {result}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()