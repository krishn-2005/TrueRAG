from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = BASE_DIR / "vector_db"
COLLECTION_NAME = "emb_a_recursive"

TOP_K = 3

LLM_MODEL = "qwen3:4b-q4_K_M"
