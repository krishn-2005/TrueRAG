import streamlit as st

from src.rag.graph import app as rag_app


st.set_page_config(
    page_title="Project TrueRAG",
    page_icon="🪐",
    layout="wide"
)

# SIDEBAR


st.sidebar.markdown(
    "<div style='font-size:1.1rem; font-weight:600; margin-top:20px; "
    "margin-bottom:8px;'>Navigation</div>",
    unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state.page = "Chat"

if st.sidebar.button("Chat 🗨️", use_container_width=True):
    st.session_state.page = "Chat"

if st.sidebar.button("Evaluation 🧐", use_container_width=True):
    st.session_state.page = "Evaluation"

page = st.session_state.page

# CHAT PAGE

if page == "Chat":

    st.title("TrueRAG")
    
    st.write(
      "An Evaluation-Driven RAG System, Ask questions about the indexed research papers."
    )


    # Store chat history
    if "messages" not in st.session_state:

        st.session_state.messages = []


    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if (sources and message["content"] != "I don't have enough information in the provided context."):

                    st.markdown("**Sources**")

                    for source in sources:

                        title = (
                            source.get("title")
                            or source.get("source")
                            or "Document"
                        )

                        page_number = source.get("page")

                        url = source.get("url")

                        st.markdown(
                            f"- **{title}** — Page {page_number}"
                        )

                        if url:

                            st.markdown(
                                f"[View PDF]({url})"
                            )


    # Chat input
    if query := st.chat_input(
        "Ask a question about the research papers..."
    ):

        # Display user question
        with st.chat_message("user"):

            st.markdown(query)


        # Save user question
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })


        # Run RAG pipeline
        with st.chat_message("assistant"):

            with st.spinner("Searching documents..."):

                result = rag_app.invoke({
                    "question": query
                })


            answer = result["answer"]

            sources = result.get("sources", [])


            # Display answer
            st.markdown(answer)


            # Display sources
            if (sources and answer != "I don't have enough information in the provided context."):

                st.markdown("**Sources**")

                for source in sources:

                    title = (
                        source.get("title")
                        or source.get("source")
                        or "Document"
                    )

                    page_number = source.get("page")

                    url = source.get("url")


                    st.markdown(
                        f"- **{title}** — Page {page_number}"
                    )

                    if url:

                        st.markdown(
                            f"[View PDF]({url})"
                        )


        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


# EVALUATION PAGE

else:

    st.title("RAG Evaluation 🔬")

    st.write(
        "The RAG pipeline was evaluated using RAGAS "
        "across different retrieval configurations."
    )


    st.subheader("Evaluation Metrics")

    st.write(
        """
        **Faithfulness:** Is the answer supported by the retrieved context?

        **Answer Relevancy:** Is the answer relevant to the question?

        **Context Precision:** How relevant are the retrieved chunks?

        **Context Recall:** How much relevant information was retrieved?
        """
    )


    st.subheader("Results")


    evaluation_data = {
        "Configuration": [
            "emb_a_semantic",
            "emb_b_semantic",
            "emb_a_recursive",
            "emb_b_recursive*"
        ],

        "Faithfulness": [
            0.362,
            0.325,
            0.400,
            0.275
        ],

        "Answer Relevancy": [
            0.624,
            0.634,
            0.751,
            0.757
        ],

        "Context Precision": [
            0.621,
            0.700,
            0.742,
            0.808
        ],

        "Context Recall": [
            0.600,
            0.600,
            0.700,
            0.700
        ]
    }


    st.dataframe(
        evaluation_data,
        use_container_width=True
    )


    st.subheader("Selected Configuration")

    st.write(
        """
        **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`

        **Chunking:** `RecursiveCharacterTextSplitter`

        **Chunk Size:** `1000`

        **Chunk Overlap:** `200`

        **Top-K:** `3`

        **Vector Database:** `Chroma`
        """
    )


    st.info(
        """
        *emb_b_recursive was not fully evaluated because of
        computational/resource constraints. Its missing values
        were estimated and are not actual measured RAGAS scores.
        """
    )