from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

Answer the user's question using only the information provided in the context.

If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Do not invent facts."""
    ),
    (
        "human",
        """Context:

{context}

Question:

{query}

Answer:"""
    )
])