import os
import base64

from dotenv import load_dotenv
from sarvamai import SarvamAI


load_dotenv()


SARVAM_API_KEY = os.getenv(
    "SARVAM_API_KEY"
)


def get_sarvam_client():

    if not SARVAM_API_KEY:
        raise ValueError(
            "SARVAM_API_KEY is not set in .env"
        )

    return SarvamAI(
        api_subscription_key=SARVAM_API_KEY
    )


def text_to_speech(
    text,
    output_path="data/raw/answer.wav",
    language_code="en-IN",
    speaker="shubh"
):

    client = get_sarvam_client()

    response = client.text_to_speech.convert(
        text=text,
        model="bulbul:v3",
        language_code=language_code,
        speaker=speaker,
        speech_sample_rate=24000
    )

    # Sarvam returns base64 encoded audio
    combined_audio = "".join(
        response.audios
    )

    audio_bytes = base64.b64decode(
        combined_audio
    )

    with open(
        output_path,
        "wb"
    ) as audio_file:

        audio_file.write(
            audio_bytes
        )

    return output_path