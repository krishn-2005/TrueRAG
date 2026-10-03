from src.retriever import get_retriever

retriever = get_retriever()

docs = retriever.invoke("What is self-attention?")

for doc in docs:
    # print(doc.page_content[:300])
    print(doc.metadata)
    print("-" * 50)
