from src.llm import get_llm


llm = get_llm()

response = llm.invoke(
    "Explain self-attention in 2-3 simple sentences."
)

print(response.content)