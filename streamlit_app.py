import streamlit as st
from src.rag_pipeline import answer_query
from src.ingest import ingest_uploaded_pdf

st.set_page_config(layout="wide")

# Initialize session state FIRST
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "store_path" not in st.session_state:
    st.session_state.store_path = None

st.title("📄 Domain RAG Assistant")

with st.sidebar:
    st.header("📂 Document Upload")

    mode = st.radio(
    "Mode",
    ["Question Answering", "Document Summary"]
                    )
    st.session_state.mode = mode

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and not st.session_state.document_ready:
        with st.spinner("Processing document..."):
            store_path = ingest_uploaded_pdf(uploaded_file)
            st.session_state.store_path = store_path
            st.session_state.document_ready = True

        st.session_state.document_ready = True
        st.success("Document processed successfully.")

    if st.session_state.document_ready:
        st.info("Document ready for querying.")
    else:
        st.warning("Upload a PDF to begin.")

        
# Hide default Streamlit UI elements
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at bottom
prompt = st.chat_input("Ask a question about the uploaded document...")

if prompt and st.session_state.document_ready:
    # 1. Add user message first
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # 2. Generate response
    response = answer_query(prompt, st.session_state.store_path,st.session_state.mode)

    # 3. Add assistant message to memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"]
    })

    # 4. Force rerun so rendering happens cleanly
    st.rerun()

elif prompt and not st.session_state.document_ready:
    st.warning("Please upload a document first.")