import os
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

from src.config import LLM_MODEL

def get_llm():
  
  # hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
  groq_token = os.getenv("GROQ_API_KEY")

  if not groq_token:
    raise ValueError(
        "GROQ_API_KEY is not set."
    )
  
  # return HuggingFaceEndpoint(
  #   repo_id=LLM_MODEL,
  #   huggingfacehub_api_token=hf_token,
  #   task="text-generation",
  #   )
  
  return ChatGroq(
    model=LLM_MODEL,
    api_key=groq_token,
  )