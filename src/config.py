from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = BASE_DIR / "vector_db"
COLLECTION_NAME = "emb_a_recursive"

TOP_K = 3

LLM_MODEL = "openai/gpt-oss-safeguard-20b"

PDF_URLS = {
    "survey_of_transformers.pdf":
        "https://github.com/krishn-2005/TrueRAG/blob/main/data/survey_of_transformers.pdf",

    "LLM_survey.pdf":
        "https://github.com/krishn-2005/TrueRAG/blob/main/data/LLM_survey.pdf",

    "eval_LLM.pdf":
        "https://github.com/krishn-2005/TrueRAG/blob/main/data/eval_LLM.pdf",
}
