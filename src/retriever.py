from langchain_chroma import Chroma

from src.config import VECTOR_DB_PATH, TOP_K
from src.embeddings import get_embeddings


def get_retriever():
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="emb_a_recursive",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
    )

    return vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )