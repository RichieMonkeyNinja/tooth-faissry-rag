import streamlit as st
from dotenv import load_dotenv

from retrieval import RELEVANCE_THRESHOLD, VECTORSTORE_DIR, answer_question, load_vectorstore


st.set_page_config(page_title="MedMCQA RAG Chatbot", page_icon=":stethoscope:", layout="wide")


@st.cache_resource
def get_vectorstore():
    load_dotenv()
    return load_vectorstore(VECTORSTORE_DIR)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None


def render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_diagnostics() -> None:
    st.subheader("Latest Diagnostics")
    result = st.session_state.latest_result

    if result is None:
        st.info("No retrieval has been run yet.")
        return

    st.write(f"Accepted: `{result['accepted']}`")
    st.write(f"Score: `{result['score']:.4f}`")
    st.write(f"Threshold: `{RELEVANCE_THRESHOLD:.2f}`")
    st.write("Retrieved chunk:")
    st.code(result["retrieved_chunk"] or "None", language="text")
    st.write("Metadata:")
    st.json(result.get("metadata", {}))

    if not result["accepted"]:
        st.error(result["reason"])


def main() -> None:
    init_session_state()

    st.title("MedMCQA Retrieval Chatbot")
    st.caption("Answers are generated only from the nearest retrieved context in the FAISS index.")

    chat_col, diagnostics_col = st.columns([2, 1], gap="large")
    vectorstore = get_vectorstore()

    with chat_col:
        render_messages()
        question = st.chat_input("Ask a medical question")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            result = answer_question(question, vectorstore)
            st.session_state.latest_result = result

            if result["accepted"]:
                assistant_message = result["answer"]
            else:
                assistant_message = "No sufficiently similar context was found in the database, so no grounded answer was generated."

            st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            with st.chat_message("assistant"):
                st.markdown(assistant_message)

    with diagnostics_col:
        render_diagnostics()


if __name__ == "__main__":
    main()
