import os

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


def transcribe_audio(
    audio_path,
    language_code="en-IN"
):

    client = get_sarvam_client()

    with open(
        audio_path,
        "rb"
    ) as audio_file:

        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="transcribe",
            language_code=language_code
        )

    return response.transcript