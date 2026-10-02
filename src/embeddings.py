import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from src.config import EMBEDDING_MODEL

from dotenv import load_dotenv
load_dotenv()

def get_embeddings():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN is not set."
        )

    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )
