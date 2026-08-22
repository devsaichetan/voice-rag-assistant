import os
import time
import uuid

from src.stt import transcribe_audio
from src.rag_engine import RAGEngine
from src.tts import text_to_speech


# ==================================================
# Project paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

AUDIO_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)


# Make sure output directory exists
os.makedirs(
    AUDIO_OUTPUT_DIR,
    exist_ok=True
)


# ==================================================
# Load RAG engine ONCE
# ==================================================

print("Loading shared RAG engine...")

rag_engine = RAGEngine()

print("Shared RAG engine ready!")


# ==================================================
# Process voice
# ==================================================

def process_voice(
    audio_path,
    language_code="en-IN"
):

    total_start = time.perf_counter()

    # ==================================================
    # Speech-to-text
    # ==================================================

    stt_start = time.perf_counter()

    transcript = transcribe_audio(
        audio_path,
        language_code=language_code
    )

    stt_time = (
        time.perf_counter()
        - stt_start
    ) * 1000

    print("\n" + "=" * 70)
    print("TRANSCRIPTION")
    print("=" * 70)

    print(transcript)

    # ==================================================
    # RAG
    # ==================================================

    print("\nRunning RAG...")

    result = rag_engine.ask(
        transcript
    )

    # ==================================================
    # Text-to-speech
    # ==================================================

    print("\nGenerating voice answer...")

    tts_start = time.perf_counter()

    # Generate a unique filename for every request
    audio_filename = (
        f"voice_answer_"
        f"{uuid.uuid4().hex}.wav"
    )

    audio_output_path = os.path.join(
        AUDIO_OUTPUT_DIR,
        audio_filename
    )

    audio_output = text_to_speech(
        text=result["answer"],
        output_path=audio_output_path,
        language_code=language_code,
        speaker="shubh"
    )

    tts_time = (
        time.perf_counter()
        - tts_start
    ) * 1000

    # ==================================================
    # Total time
    # ==================================================

    total_time = (
        time.perf_counter()
        - total_start
    ) * 1000

    # ==================================================
    # Return result
    # ==================================================

    return {

        "transcript":
            transcript,

        "answer":
            result["answer"],

        "sources":
            result["sources"],

        "audio_path":
            audio_output,

        "audio_filename":
            audio_filename,

        "timing": {

            "stt_ms":
                stt_time,

            "retrieval_ms":
                result[
                    "retrieval_time_ms"
                ],

            "reranking_ms":
                result[
                    "rerank_time_ms"
                ],

            "generation_ms":
                result[
                    "generation_time_ms"
                ],

            "tts_ms":
                tts_time,

            "total_ms":
                total_time
        }
    }