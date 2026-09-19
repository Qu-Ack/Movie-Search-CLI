from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

MOVIES_FILE = DATA_DIR / "movies.json"
GOLDEN_DATASET_FILE = DATA_DIR / "golden_dataset.json"

# Cache file paths
INDEX_CACHE_FILE = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_FILE = CACHE_DIR / "docmap.pkl"
TF_CACHE_FILE = CACHE_DIR / "tf.pkl"
EMBEDDINGS_CACHE_FILE = CACHE_DIR / "embeddings.pkl"

# Default configuration parameters
DEFAULT_ALPHA: float = 0.7
DEFAULT_TOP_N: int = 5
DEFAULT_RETRIEVAL_LIMIT: int = 50
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"


def ensure_cache_dir() -> Path:
    """Ensure that the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR
