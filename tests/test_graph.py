from src.rag.graph import app


result = app.invoke({
    "question": "What are the recent news currently about world cup 2023?"
})


print("\nANSWER:")
print(result["answer"])


print("\nSOURCES:")

for source in result["sources"]:
  
    print(f"Source: {source['source']}")
    print(f"Page: {source['page']}")
    print(f"URL: {source['url']}")
    print("-" * 50)