import os
from dotenv import load_dotenv

load_dotenv()


# ==================================================
# Project paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RAW_DATA_DIR = os.path.join(
    DATA_DIR,
    "raw"
)

PROCESSED_DATA_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

INDEX_DIR = os.path.join(
    BASE_DIR,
    "indexes"
)


# ==================================================
# Dataset
# ==================================================

DATASET_NAME = "microsoft/ms_marco"

DATASET_CONFIG = "v1.1"

DATASET_SPLIT = "train"


# ==================================================
# Retrieval
# ==================================================

TOP_K = 5

# Reranker removed to reduce RAM usage
RERANK_TOP_N = 0

SIMILARITY_THRESHOLD = 0.25


# ==================================================
# Lightweight ONNX Embedding Model
# ==================================================

EMBEDDING_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "paraphrase-MiniLM-L3-v2"
)


# ==================================================
# Speech-to-Text / Sarvam
# ==================================================

STT_API_KEY = os.getenv(
    "STT_API_KEY"
)

# Also support SARVAM_API_KEY if that is
# the variable currently used in Render.
SARVAM_API_KEY = os.getenv(
    "SARVAM_API_KEY"
)


# ==================================================
# Groq LLM
# ==================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# ==================================================
# Legacy Gemini configuration
# ==================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)
