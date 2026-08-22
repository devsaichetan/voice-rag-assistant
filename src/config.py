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

TOP_K = 20

RERANK_TOP_N = 5

SIMILARITY_THRESHOLD = 0.65


# ==================================================
# Embedding model
# ==================================================

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "paraphrase-MiniLM-L3-v2"
)


# ==================================================
# Speech-to-Text
# ==================================================

STT_API_KEY = os.getenv(
    "STT_API_KEY"
)


# ==================================================
# Groq LLM
# ==================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


# ==================================================
# Legacy Gemini configuration
# ==================================================
# Keep these for now so we can switch back to Gemini
# easily if needed.

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)
