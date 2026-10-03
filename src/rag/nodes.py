from pathlib import Path
from src.config import PDF_URLS
from src.retriever import get_retriever
from src.llm import get_llm
from src.rag.prompts import prompt
from langchain_core.output_parsers import StrOutputParser


retriever = get_retriever()
llm = get_llm()
parser = StrOutputParser()


def retrieve(state):
    question = state["question"]

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    sources = []

    for doc in retrieved_docs:
      filename = Path(
          doc.metadata.get("source", "")
      ).name

      page_label = doc.metadata.get("page_label")

      if page_label is None:
          page = doc.metadata.get("page")
          page_label = page + 1 if page is not None else None

      sources.append({
          "source": filename,
          "page": page_label,
          "url": PDF_URLS.get(filename)
      })

    return {
        "context": context,
        "sources": sources
    }

def generate(state):
    question = state["question"]
    context = state["context"]

    response = (prompt | llm | parser).invoke({
        "query": question,
        "context": context
    })

    return {
        "answer": response
    }


def check_context(state):
    context = state.get("context", "")

    if not context:
        return "no_context"

    if not str(context).strip():
        return "no_context"

    return "has_context"


def fallback(state):
    return {
        "answer": "I don't have enough information in the provided documents."
    }