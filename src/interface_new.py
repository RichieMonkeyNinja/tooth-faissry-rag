import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from retrieval import (
    NO_GROUNDED_ANSWER_MESSAGE,
    RERANK_ACCEPT_THRESHOLD,
    SOURCE_FILTER_ALL,
    VECTORSTORE_DIR,
    answer_question,
    load_vectorstore,
)


st.set_page_config(page_title="Tooth FAISSry: RAG Chatbot", page_icon=":stethoscope:", layout="wide")


@st.cache_resource
def get_vectorstore():
    load_dotenv()
    return load_vectorstore(VECTORSTORE_DIR)


def init_session_state() -> None:
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        new_id = create_new_chat()
        st.session_state.current_chat_id = new_id
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "editing_chat_id" not in st.session_state:
        st.session_state.editing_chat_id = None


def create_new_chat() -> str:
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    st.session_state.chats[chat_id] = {
        "title": "Chat Room",
        "messages": [],
    }
    return chat_id


def get_current_chat() -> dict:
    return st.session_state.chats[st.session_state.current_chat_id]


def render_messages() -> None:
    chat = get_current_chat()
    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_chat_history() -> None:
    #Sidebar list of historical chats
    st.header("Chats")

    if st.button("New Chat", use_container_width=True, key="new_chat_bottom"):
        new_id = create_new_chat()
        st.session_state.current_chat_id = new_id
        st.session_state.latest_result = None
        st.session_state.editing_chat_id = None
        st.rerun()

    st.divider()

    sorted_chat_ids = sorted(st.session_state.chats.keys(), reverse=True)

    for chat_id in sorted_chat_ids:
        chat = st.session_state.chats[chat_id]
        is_current = chat_id == st.session_state.current_chat_id
        is_editing = st.session_state.editing_chat_id == chat_id

        st.markdown('<div class="chat-row">', unsafe_allow_html=True)

        if is_editing:
            new_title = st.text_input(
                "Rename chat",
                value=chat["title"],
                key=f"edit_input_{chat_id}",
                label_visibility="collapsed",
            )
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("Save", key=f"save_{chat_id}", use_container_width=True):
                    if new_title.strip():
                        chat["title"] = new_title.strip()
                    st.session_state.editing_chat_id = None
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key=f"cancel_{chat_id}", use_container_width=True):
                    st.session_state.editing_chat_id = None
                    st.rerun()
        else:
            label = f"{chat['title']}" if is_current else chat["title"]

            col1, col2, col3 = st.columns([5, 2, 2], gap="small", vertical_alignment="center")
            with col1:
                if st.button(
                    label,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.latest_result = None
                    st.rerun()
            with col2:
                if st.button(":material/edit:", key=f"edit_{chat_id}", help="Rename chat", use_container_width=True):
                    st.session_state.editing_chat_id = chat_id
                    st.rerun()
            with col3:
                if st.button(":material/delete:", key=f"del_{chat_id}", help="Delete chat", use_container_width=True):
                    del st.session_state.chats[chat_id]
                    if chat_id == st.session_state.current_chat_id:
                        if st.session_state.chats:
                            st.session_state.current_chat_id = sorted(st.session_state.chats.keys(), reverse=True)[0]
                        else:
                            st.session_state.current_chat_id = create_new_chat()
                        st.session_state.latest_result = None
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


def render_retrieval_details() -> None:
    st.subheader("Latest Retrieval Details")
    result = st.session_state.latest_result

    if result is None:
        st.info("No retrieval has been run yet.")
        return

    st.write(f"Accepted: `{result['accepted']}`")
    st.write(f"Source filter: `{result.get('source_filter', SOURCE_FILTER_ALL)}`")
    st.write(f"Score: `{result['score']:.4f}`")
    st.write(f"Vector score: `{result.get('vector_score', 0.0):.4f}`")
    st.write(f"BM25 score: `{result.get('bm25_score', 0.0):.4f}`")
    st.write(f"RRF score: `{result.get('rrf_score', 0.0):.4f}`")
    rerank_score = result.get("rerank_score")
    st.write(f"Rerank score: `{rerank_score:.4f}`" if rerank_score is not None else "Rerank score: `None`")
    st.write(f"Rerank accept threshold: `{RERANK_ACCEPT_THRESHOLD:.2f}`")
    st.write("Retrieved chunk:")
    st.code(result["retrieved_chunk"] or "None", language="text")
    st.write("Metadata:")
    st.json(result.get("metadata", {}))

    if not result["accepted"]:
        st.error(result["reason"])

def render_disclaimer() -> None:
    st.markdown(
        """
        **Disclaimer**
        
        This chatbot is a research prototype designed to assist with dental questions. It retrieves information from a vector database built on the MedMCQA dataset and the California Dental Association. 
        
        Can make mistakes. Check important info.
        """
    )

def main() -> None:
    init_session_state()

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 8px;
        }

        /* Sidebar buttons: light grey base style */
        section[data-testid="stSidebar"] .stButton > button {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            font-weight: 400;
            height: 38px;           
            padding: 0 10px;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: #e4e4e4;
            color: #000000;
            border-color: #cccccc;
        }
                
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background-color: #e4e4e4;
            border-color: #cccccc;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background-color: #f0f0f0;
            border-color: #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Tooth FAISSry Chatbot: A RAG-powered Dental Knowledge Assistant")
    st.caption("Document Sources: MedMCQA; California Dental Association ")

    vectorstore = get_vectorstore()

    render_messages()
    question = st.chat_input("Ask a dental question: e.g. What are the causes of bad breath?")

    if question:
        chat = get_current_chat()

        chat["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Retrieving answer..."):
            result = answer_question(question, vectorstore, source_filter='all')
        st.session_state.latest_result = result

        if result["accepted"]:
            assistant_message = result["answer"]
        else:
            assistant_message = NO_GROUNDED_ANSWER_MESSAGE

        chat["messages"].append({"role": "assistant", "content": assistant_message})
        with st.chat_message("assistant"):
            st.markdown(assistant_message)

    with st.sidebar:
        render_chat_history()
        st.divider()
        st.header("🔬 Retrieval Information")
        render_retrieval_details()
        st.divider()
        render_disclaimer()


if __name__ == "__main__":
    main()