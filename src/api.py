import os
import tempfile

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from src.voice_rag import (
    process_voice,
    rag_engine
)


# ==================================================
# Project paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

TEMPLATES_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

AUDIO_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)


# ==================================================
# FastAPI application
# ==================================================

app = FastAPI(
    title="Voice RAG API",
    description=(
        "RAG-based Question Answering API "
        "with Voice Support"
    ),
    version="1.0.0"
)


# ==================================================
# Static files
# ==================================================

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static"
)


# ==================================================
# Request model
# ==================================================

class QuestionRequest(BaseModel):

    question: str


# ==================================================
# Root endpoint
# ==================================================

@app.get("/")
def root():

    return {
        "message": "Voice RAG API is running",
        "status": "ok"
    }


# ==================================================
# Text Ask endpoint
# ==================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        result = rag_engine.ask(
            question
        )

        return {

            "question":
                question,

            "answer":
                result["answer"],

            "sources":
                result["sources"],

            "timing": {

                "retrieval_ms":
                    round(
                        result[
                            "retrieval_time_ms"
                        ],
                        2
                    ),

                "reranking_ms":
                    round(
                        result[
                            "rerank_time_ms"
                        ],
                        2
                    ),

                "generation_ms":
                    round(
                        result[
                            "generation_time_ms"
                        ],
                        2
                    ),

                "total_ms":
                    round(
                        result[
                            "total_time_ms"
                        ],
                        2
                    )
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==================================================
# Voice Ask endpoint
# ==================================================

@app.post("/voice-ask")
async def voice_ask(
    file: UploadFile = File(...)
):

    # ==================================================
    # Validate filename
    # ==================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Audio file is required."
        )

    # ==================================================
    # Allowed audio formats
    # ==================================================

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".ogg",
        ".m4a",
        ".webm",
        ".flac"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Allowed formats: "
                ".wav, .mp3, .ogg, .m4a, "
                ".webm, .flac"
            )
        )

    temp_path = None

    try:

        # ==================================================
        # Read uploaded audio
        # ==================================================

        audio_data = await file.read()

        if not audio_data:

            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty."
            )

        # ==================================================
        # Save temporary input file
        # ==================================================

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(
                audio_data
            )

            temp_path = temp_file.name

        print(
            f"\nReceived audio: "
            f"{file.filename}"
        )

        print(
            f"Temporary file: "
            f"{temp_path}"
        )

        # ==================================================
        # Voice RAG pipeline
        # ==================================================

        result = process_voice(
            temp_path
        )

        # ==================================================
        # Get generated audio filename
        # ==================================================

        audio_filename = (
            result["audio_filename"]
        )

        # ==================================================
        # Response
        # ==================================================

        return {

            "filename":
                file.filename,

            "transcript":
                result["transcript"],

            "answer":
                result["answer"],

            "sources":
                result["sources"],

            "audio_url":
                f"/voice-answer/{audio_filename}",

            "timing": {

                "stt_ms":
                    round(
                        result[
                            "timing"
                        ]["stt_ms"],
                        2
                    ),

                "retrieval_ms":
                    round(
                        result[
                            "timing"
                        ]["retrieval_ms"],
                        2
                    ),

                "reranking_ms":
                    round(
                        result[
                            "timing"
                        ]["reranking_ms"],
                        2
                    ),

                "generation_ms":
                    round(
                        result[
                            "timing"
                        ]["generation_ms"],
                        2
                    ),

                "tts_ms":
                    round(
                        result[
                            "timing"
                        ]["tts_ms"],
                        2
                    ),

                "total_ms":
                    round(
                        result[
                            "timing"
                        ]["total_ms"],
                        2
                    )
            }
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # ==================================================
        # Remove temporary uploaded input
        # ==================================================

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ==================================================
# Voice answer audio endpoint
# ==================================================

@app.get(
    "/voice-answer/{filename}"
)
def get_voice_answer(
    filename: str
):

    # --------------------------------------------------
    # Security: prevent path traversal
    # --------------------------------------------------

    safe_filename = os.path.basename(
        filename
    )

    if safe_filename != filename:

        raise HTTPException(
            status_code=400,
            detail="Invalid audio filename."
        )

    # --------------------------------------------------
    # Only allow WAV files
    # --------------------------------------------------

    if not safe_filename.lower().endswith(
        ".wav"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only WAV audio files are allowed."
        )

    audio_path = os.path.join(
        AUDIO_DIR,
        safe_filename
    )

    # --------------------------------------------------
    # Check file exists
    # --------------------------------------------------

    if not os.path.exists(
        audio_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Voice answer audio not found."
        )

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=safe_filename
    )


# ==================================================
# Frontend
# ==================================================

@app.get("/app")
def frontend():

    return FileResponse(
        os.path.join(
            TEMPLATES_DIR,
            "index.html"
        )
    )