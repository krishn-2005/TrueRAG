from pathlib import Path
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import VECTOR_DB_PATH, COLLECTION_NAME
from src.embeddings import get_embeddings


DATA_DIR = APP_DIR / "data"
PDF_PATHS = [
    (DATA_DIR / "survey_of_transformers.pdf", 33),
    (DATA_DIR / "LLM_survey.pdf", 36),
    (DATA_DIR / "eval_LLM.pdf", 58),
]

def load_documents():

    documents = []

    for pdf_path, page_limit in PDF_PATHS:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        documents.extend(docs[:page_limit])

    return documents

def build_vector_db():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_PATH),
    )

    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    build_vector_db()
